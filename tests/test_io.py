"""Single-file persistence for sessions and fits."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from caltrust.core.poses import Pose
from caltrust.core.session import CalibrationRecord, CalibrationSession, RobotPoses
from caltrust.errors import SerializationError
from caltrust.io import load_fit, load_session, read_bundle, save_fit, save_session, write_bundle
from caltrust.io.bundle import check_format
from caltrust.refit import RefitOptions, instrument


def test_bundle_round_trip(tmp_path):
    manifest = {"format": "x", "format_version": 1, "note": "hello"}
    arrays = {"a": np.arange(6.0).reshape(2, 3), "b": np.array([1, 2, 3])}
    path = write_bundle(str(tmp_path / "b"), manifest, arrays)
    assert path.endswith(".npz")
    back_manifest, back_arrays = read_bundle(path)
    assert back_manifest == manifest
    assert np.array_equal(back_arrays["a"], arrays["a"])


def test_bundle_appends_the_suffix_only_when_missing(tmp_path):
    named = write_bundle(str(tmp_path / "c.npz"), {"format": "x"}, {})
    assert named.endswith("c.npz") and not named.endswith("c.npz.npz")


def test_bundle_creates_missing_directories(tmp_path):
    path = write_bundle(str(tmp_path / "deep" / "nested" / "b"), {"format": "x"}, {})
    assert os.path.isfile(path)


def test_bundle_rejects_the_reserved_array_name(tmp_path):
    with pytest.raises(SerializationError, match="reserved"):
        write_bundle(str(tmp_path / "b"), {}, {"__manifest__": np.zeros(1)})


def test_bundle_rejects_an_unserialisable_manifest(tmp_path):
    with pytest.raises(SerializationError, match="not JSON-serialisable"):
        write_bundle(str(tmp_path / "b"), {"f": object()}, {})


def test_bundle_serialises_numpy_scalars(tmp_path):
    path = write_bundle(str(tmp_path / "b"), {"v": np.float64(1.5), "a": np.arange(3)}, {})
    manifest, _ = read_bundle(path)
    assert manifest["v"] == 1.5 and manifest["a"] == [0, 1, 2]


def test_reading_a_missing_file_says_so(tmp_path):
    with pytest.raises(SerializationError, match="no such file"):
        read_bundle(str(tmp_path / "absent.npz"))


def test_reading_a_non_npz_says_so(tmp_path):
    path = tmp_path / "junk.npz"
    path.write_bytes(b"not an archive")
    with pytest.raises(SerializationError, match="could not read"):
        read_bundle(str(path))


def test_reading_an_npz_without_a_manifest_says_so(tmp_path):
    path = str(tmp_path / "plain.npz")
    np.savez(path, a=np.zeros(3))
    with pytest.raises(SerializationError, match="not a caltrust bundle"):
        read_bundle(path)


def test_check_format_rejects_the_wrong_format():
    with pytest.raises(SerializationError, match="expected a 'y' bundle"):
        check_format({"format": "x", "format_version": 1}, "y", 1)


def test_check_format_rejects_a_newer_version():
    with pytest.raises(SerializationError, match="newer than this caltrust"):
        check_format({"format": "x", "format_version": 9}, "x", 1)


def test_session_round_trip_is_exact(good_capture, pinhole, tmp_path):
    robot = RobotPoses.from_matrices(
        [Pose.from_rvec_tvec([0.05 * i, 0, 0], [i, 0, 400.0]).matrix
         for i in range(good_capture.observations.n_views)],
        good_capture.observations.view_ids,
    )
    session = CalibrationSession(
        observations=good_capture.observations,
        prior=CalibrationRecord(pinhole, (1280, 720), "a.yml", 0.24),
        robot=robot,
        metadata={"rig": "cell-3"},
    )
    path = save_session(session, str(tmp_path / "s"))
    back = load_session(path)
    assert back.observations.view_ids == session.observations.view_ids
    assert back.observations.target == session.observations.target
    for a, b in zip(session.observations.views, back.observations.views):
        assert np.array_equal(a.point_ids, b.point_ids)
        assert np.array_equal(a.image_points, b.image_points)
        assert a.metadata == b.metadata
    assert back.prior.camera == pinhole and back.prior.reported_rms == 0.24
    assert all(a.allclose(b, atol=1e-9) for a, b in zip(robot.poses, back.robot.poses))
    assert back.metadata == {"rig": "cell-3"}
    assert back.created == session.created


def test_session_without_prior_or_robot_round_trips(good_session, tmp_path):
    back = load_session(save_session(good_session, str(tmp_path / "s")))
    assert back.prior is None and back.robot is None


def test_loading_a_fit_as_a_session_is_refused(good_session, tmp_path):
    fit = instrument(good_session)
    path = save_fit(fit, str(tmp_path / "f"))
    with pytest.raises(SerializationError, match="expected a 'caltrust.session'"):
        load_session(path)


def test_session_bundle_detects_a_view_count_mismatch(good_session, tmp_path):
    path = save_session(good_session, str(tmp_path / "s"))
    manifest, arrays = read_bundle(path)
    manifest["views"] = manifest["views"][:-1]
    broken = write_bundle(str(tmp_path / "broken"), manifest, arrays)
    with pytest.raises(SerializationError, match="offset spans"):
        load_session(broken)


@pytest.mark.parametrize("terms", [5, 8])
def test_fit_round_trip_is_exact(session_with_prior, tmp_path, terms):
    fit = instrument(session_with_prior, RefitOptions(distortion_terms=terms))
    back = load_fit(save_fit(fit, str(tmp_path / f"f{terms}")))
    assert back.camera.allclose(fit.camera)
    assert all(a.allclose(b, atol=1e-12) for a, b in zip(fit.poses, back.poses))
    assert np.allclose(back.covariance.intrinsic, fit.covariance.intrinsic)
    assert np.allclose(back.residuals.residuals, fit.residuals.residuals)
    assert np.allclose(back.residuals.radial.radial_rms, fit.residuals.radial.radial_rms)
    assert back.conditioning.to_dict() == fit.conditioning.to_dict()
    assert back.options == fit.options
    assert back.view_ids == fit.view_ids
    assert back.prior_rms == fit.prior_rms
    assert back.initial_guess == fit.initial_guess
    assert back.rms == pytest.approx(fit.rms)
    assert back.at_optimum == fit.at_optimum
    assert [v.robust_z for v in back.residuals.per_view] == pytest.approx(
        [v.robust_z for v in fit.residuals.per_view]
    )


def test_loading_a_session_as_a_fit_is_refused(good_session, tmp_path):
    path = save_session(good_session, str(tmp_path / "s"))
    with pytest.raises(SerializationError, match="expected a 'caltrust.fit'"):
        load_fit(path)


def test_fit_bundle_detects_a_missing_array(good_session, tmp_path):
    fit = instrument(good_session)
    path = save_fit(fit, str(tmp_path / "f"))
    manifest, arrays = read_bundle(path)
    del arrays["eq_u_view"]
    broken = write_bundle(str(tmp_path / "broken"), manifest, arrays)
    with pytest.raises(SerializationError, match="missing"):
        load_fit(broken)


def test_bundle_holds_no_pickled_objects(good_session, tmp_path):
    """A bundle must never be able to execute code when it is opened."""
    path = save_session(good_session, str(tmp_path / "s"))
    with np.load(path, allow_pickle=False) as archive:
        assert "__manifest__" in archive.files
        json.loads(str(archive["__manifest__"]))


def test_session_bundle_detects_a_missing_array(good_session, tmp_path):
    path = save_session(good_session, str(tmp_path / "s"))
    manifest, arrays = read_bundle(path)
    del arrays["point_ids"]
    broken = write_bundle(str(tmp_path / "broken"), manifest, arrays)
    with pytest.raises(SerializationError, match="missing"):
        load_session(broken)


def test_bundle_can_be_written_uncompressed(tmp_path):
    arrays = {"a": np.arange(100.0)}
    plain = write_bundle(str(tmp_path / "plain"), {"format": "x"}, arrays, compress=False)
    packed = write_bundle(str(tmp_path / "packed"), {"format": "x"}, arrays, compress=True)
    assert os.path.getsize(plain) >= os.path.getsize(packed)
    assert np.array_equal(read_bundle(plain)[1]["a"], arrays["a"])


def test_writing_to_an_unwritable_path_is_reported(tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    with pytest.raises(SerializationError, match="could not write"):
        write_bundle(str(blocker / "inner" / "b"), {"format": "x"}, {})


def test_fit_bundle_carries_the_per_view_information_blocks(good_session, tmp_path):
    """M4's leverage diagnostic needs them, so they have to survive a round trip."""
    fit = instrument(good_session)
    back = load_fit(save_fit(fit, str(tmp_path / "f")))
    assert np.allclose(back.equations.u_view, fit.equations.u_view)
    assert np.allclose(back.equations.u, fit.equations.u)
    assert np.allclose(back.equations.u, back.equations.u_view.sum(axis=0))


def test_fit_bundle_round_trips_cross_validation(good_session, tmp_path):
    import dataclasses

    from caltrust.validate import cross_validate

    validation = cross_validate(good_session)
    fit = dataclasses.replace(instrument(good_session), cross_validation=validation)
    back = load_fit(save_fit(fit, str(tmp_path / "f")))
    assert back.cross_validation is not None
    assert back.cross_validation.n_folds == validation.n_folds
    assert back.cross_validation.ratio == pytest.approx(validation.ratio)
    assert back.cross_validation.degenerate_folds == validation.degenerate_folds
    assert np.allclose(back.cross_validation.fold_spread, validation.fold_spread)
    assert [f.camera.fx for f in back.cross_validation.folds] == pytest.approx(
        [f.camera.fx for f in validation.folds]
    )


def test_a_fit_without_cross_validation_round_trips_as_none(good_session, tmp_path):
    fit = instrument(good_session)
    assert load_fit(save_fit(fit, str(tmp_path / "f"))).cross_validation is None
