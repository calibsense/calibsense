"""M3 — out-of-sample error by cross-validation."""

from __future__ import annotations

import numpy as np
import pytest

from caltrust.core.session import CalibrationSession
from caltrust.errors import RefitError, ValidationError
from caltrust.refit import RefitOptions, instrument
from caltrust.synthetic import diverse_poses, frontoparallel_poses, synthesise
from caltrust.validate import (
    DEFAULT_FOLDS,
    MIN_TRAIN_VIEWS,
    CrossValidation,
    choose_folds,
    cross_validate,
    evaluate_held_out,
    fold_assignments,
)

from . import rigs


def session_of(capture):
    return CalibrationSession(observations=capture.observations)


def test_folds_partition_every_view_exactly_once():
    parts = fold_assignments(18, 5)
    assert len(parts) == 5
    pooled = np.concatenate(parts)
    assert sorted(pooled.tolist()) == list(range(18))


def test_fold_sizes_differ_by_at_most_one():
    sizes = [len(part) for part in fold_assignments(18, 5)]
    assert max(sizes) - min(sizes) <= 1


def test_folds_are_deterministic_for_a_seed():
    first = fold_assignments(20, 4, seed=7)
    second = fold_assignments(20, 4, seed=7)
    assert all(np.array_equal(a, b) for a, b in zip(first, second))


def test_shuffling_changes_the_split():
    shuffled = fold_assignments(20, 4, shuffle=True, seed=1)
    ordered = fold_assignments(20, 4, shuffle=False)
    assert ordered[0].tolist() == [0, 1, 2, 3, 4]
    assert not all(np.array_equal(a, b) for a, b in zip(shuffled, ordered))


@pytest.mark.parametrize("n_folds", [0, 1])
def test_too_few_folds_is_rejected(n_folds):
    with pytest.raises(ValidationError, match="at least 2 folds"):
        fold_assignments(10, n_folds)


def test_more_folds_than_views_is_rejected():
    with pytest.raises(ValidationError, match="cannot make"):
        fold_assignments(4, 5)


def test_choose_folds_uses_the_default_when_it_fits():
    assert choose_folds(30) == DEFAULT_FOLDS


def test_choose_folds_grows_to_protect_the_training_set():
    """Fewer folds means a bigger test set, which starves the training set."""
    chosen = choose_folds(6, requested=2)
    largest_test = int(np.ceil(6 / chosen))
    assert 6 - largest_test >= MIN_TRAIN_VIEWS


def test_choose_folds_refuses_when_there_is_nothing_to_hold_back():
    with pytest.raises(ValidationError, match="to hold any back"):
        choose_folds(MIN_TRAIN_VIEWS)


def test_a_healthy_fit_generalises(pinhole, checkerboard):
    capture = synthesise(
        pinhole, checkerboard, diverse_poses(checkerboard, 18, seed=4),
        (1280, 720), noise_px=0.2, seed=17,
    )
    validation = cross_validate(session_of(capture))
    assert validation.trustworthy
    assert validation.ratio < 1.25
    assert validation.out_of_sample_rms > 0
    assert validation.held_out_points == capture.observations.total_points


def test_every_view_is_held_out_exactly_once(pinhole, checkerboard):
    capture = synthesise(
        pinhole, checkerboard, diverse_poses(checkerboard, 18, seed=4),
        (1280, 720), noise_px=0.2, seed=17,
    )
    validation = cross_validate(session_of(capture))
    held = [v.view_id for v in validation.held_out]
    assert sorted(held) == sorted(capture.observations.view_ids)


def test_out_of_sample_is_pooled_by_point_count_not_averaged(pinhole, checkerboard):
    """A fold holding back one thin view must not weigh the same as a full one."""
    capture = synthesise(
        pinhole, checkerboard, diverse_poses(checkerboard, 18, seed=4),
        (1280, 720), noise_px=0.2, seed=17,
    )
    validation = cross_validate(session_of(capture))
    squared = sum(v.rms ** 2 * v.n_points for v in validation.held_out)
    expected = np.sqrt(squared / validation.held_out_points)
    assert validation.out_of_sample_rms == pytest.approx(expected)
    naive = np.mean([f.held_out_rms for f in validation.folds])
    assert validation.out_of_sample_rms == pytest.approx(expected)
    # The two differ in general; the pooled one is the reported number.
    assert abs(naive - expected) < 0.05


def test_overfitting_shows_up_as_a_ratio_above_one(pinhole, checkerboard):
    """Fourteen distortion terms on a small capture fit the noise, not the lens."""
    capture = synthesise(
        pinhole, checkerboard, diverse_poses(checkerboard, 10, seed=4),
        (1280, 720), noise_px=0.3, seed=21,
    )
    session = session_of(capture)
    modest = cross_validate(session, RefitOptions(distortion_terms=5))
    greedy = cross_validate(session, RefitOptions(distortion_terms=14))
    assert greedy.ratio > modest.ratio


def test_the_ratio_is_blind_to_the_focal_depth_degeneracy(pinhole, checkerboard):
    """The finding that shaped this module, pinned so it cannot be forgotten.

    A held-out view solves its own pose, so a proportionally wrong depth cancels
    a proportionally wrong focal length and reprojection stays perfect.
    """
    capture = synthesise(
        pinhole, checkerboard, frontoparallel_poses(checkerboard, 18, 800.0, seed=4),
        (1280, 720), noise_px=0.2, seed=17,
    )
    validation = cross_validate(session_of(capture))
    assert not validation.trustworthy
    assert len(validation.degenerate_folds) == validation.n_folds
    # The ratio looks perfect even though the focal length is wildly wrong.
    assert validation.ratio < 1.1
    focal = [fold.camera.fx for fold in validation.folds]
    assert max(focal) / min(focal) > 1.3


def test_fold_spread_catches_what_the_ratio_misses(pinhole, checkerboard):
    """The model-free signal: resampling views, with no noise assumption at all."""
    healthy = synthesise(
        pinhole, checkerboard, diverse_poses(checkerboard, 18, seed=4),
        (1280, 720), noise_px=0.2, seed=17,
    )
    degenerate = synthesise(
        pinhole, checkerboard, frontoparallel_poses(checkerboard, 18, 800.0, seed=4),
        (1280, 720), noise_px=0.2, seed=17,
    )
    good = cross_validate(session_of(healthy))
    bad = cross_validate(session_of(degenerate))
    assert bad.spread_of("fx") > 50 * good.spread_of("fx")


def test_a_degenerate_verdict_never_calls_the_error_honest(pinhole, checkerboard):
    capture = synthesise(
        pinhole, checkerboard, frontoparallel_poses(checkerboard, 18, 800.0, seed=4),
        (1280, 720), noise_px=0.2, seed=17,
    )
    text = "\n".join(cross_validate(session_of(capture)).summary_lines())
    assert "honest" not in text
    assert "means nothing here" in text


def test_summary_calls_a_healthy_ratio_honest(pinhole, checkerboard):
    capture = synthesise(
        pinhole, checkerboard, diverse_poses(checkerboard, 18, seed=4),
        (1280, 720), noise_px=0.2, seed=17,
    )
    assert "honest" in "\n".join(cross_validate(session_of(capture)).summary_lines())


def test_predicted_spread_is_reported_alongside_the_fold_spread(pinhole, checkerboard):
    capture = synthesise(
        pinhole, checkerboard, diverse_poses(checkerboard, 18, seed=4),
        (1280, 720), noise_px=0.2, seed=17,
    )
    validation = cross_validate(session_of(capture))
    assert np.isfinite(validation.predicted_of("fx"))
    rows = validation.spread_table()
    assert len(rows) == len(validation.parameter_names)
    assert all(len(row) == 3 for row in rows)


def test_a_fixed_parameter_has_no_predicted_spread(good_capture, pinhole):
    from caltrust.core.session import CalibrationRecord

    session = CalibrationSession(
        observations=good_capture.observations,
        prior=CalibrationRecord(pinhole, (1280, 720), "a.yml"),
    )
    validation = cross_validate(session, RefitOptions(fixed=("cx", "cy")))
    assert np.isnan(validation.predicted_of("cx"))
    assert np.isfinite(validation.predicted_of("fx"))


def test_refit_false_is_overridden_because_a_fold_must_fit(session_with_prior):
    validation = cross_validate(session_with_prior, RefitOptions(refit=False))
    focal = {round(fold.camera.fx, 6) for fold in validation.folds}
    assert len(focal) > 1, "every fold returned the same camera, so none refitted"


def test_folds_never_train_on_their_own_held_out_views(pinhole, checkerboard):
    capture = synthesise(
        pinhole, checkerboard, diverse_poses(checkerboard, 18, seed=4),
        (1280, 720), noise_px=0.2, seed=17,
    )
    for fold in cross_validate(session_of(capture)).folds:
        assert not set(fold.train_view_ids) & set(fold.held_out_view_ids)
        assert len(fold.train_view_ids) >= MIN_TRAIN_VIEWS


def test_evaluate_held_out_is_exact_without_noise(pinhole, checkerboard):
    capture = synthesise(
        pinhole, checkerboard, diverse_poses(checkerboard, 8, seed=4),
        (1280, 720), noise_px=0.0,
    )
    results, squared, points = evaluate_held_out(
        pinhole, capture.observations, [0, 1], fold=0
    )
    assert len(results) == 2
    assert squared < 1e-12
    assert points == sum(v.n_points for v in capture.observations.views[:2])
    assert all(r.rms < 1e-7 for r in results)


def test_evaluate_held_out_refuses_a_view_too_thin_for_a_pose(
    pinhole, checkerboard
):
    from caltrust.core.observations import ObservationSet, ViewObservations
    from caltrust.refit.projection import projector_for

    pose = diverse_poses(checkerboard, 1, seed=0)[0]
    ids = np.arange(3)
    points = projector_for(pinhole).project(pinhole, pose, checkerboard.object_points(ids))
    observations = ObservationSet(
        checkerboard, (1280, 720), (ViewObservations("thin", ids, points),)
    )
    with pytest.raises(RefitError, match="below the"):
        evaluate_held_out(pinhole, observations, [0], fold=0)


def test_too_few_views_to_cross_validate(pinhole, checkerboard):
    capture = synthesise(
        pinhole, checkerboard, diverse_poses(checkerboard, 4, seed=4),
        (1280, 720), noise_px=0.2,
    )
    with pytest.raises(ValidationError, match="to hold any back"):
        cross_validate(session_of(capture))


def test_explicit_fold_count_is_honoured(pinhole, checkerboard):
    capture = synthesise(
        pinhole, checkerboard, diverse_poses(checkerboard, 20, seed=4),
        (1280, 720), noise_px=0.2, seed=17,
    )
    assert cross_validate(session_of(capture), folds=4).n_folds == 4


def test_worst_views_are_ordered(pinhole, checkerboard):
    capture = synthesise(
        pinhole, checkerboard, diverse_poses(checkerboard, 18, seed=4),
        (1280, 720), noise_px=0.2, seed=17,
    )
    worst = cross_validate(session_of(capture)).worst_views(4)
    assert len(worst) == 4
    assert [v.rms for v in worst] == sorted([v.rms for v in worst], reverse=True)


def test_spread_lookups_reject_an_unknown_parameter(pinhole, checkerboard):
    capture = synthesise(
        pinhole, checkerboard, diverse_poses(checkerboard, 18, seed=4),
        (1280, 720), noise_px=0.2, seed=17,
    )
    validation = cross_validate(session_of(capture))
    for call in (validation.spread_of, validation.predicted_of):
        with pytest.raises(ValidationError, match="not a parameter"):
            call("k9")


def test_dict_round_trip(pinhole, checkerboard):
    capture = synthesise(
        pinhole, checkerboard, diverse_poses(checkerboard, 14, seed=4),
        (1280, 720), noise_px=0.2, seed=17,
    )
    validation = cross_validate(session_of(capture))
    back = CrossValidation.from_dict(validation.to_dict())
    assert back.n_folds == validation.n_folds
    assert back.ratio == pytest.approx(validation.ratio)
    assert back.parameter_names == validation.parameter_names
    assert np.allclose(back.fold_spread, validation.fold_spread)
    assert np.allclose(back.predicted_spread, validation.predicted_spread, equal_nan=True)
    assert back.degenerate_folds == validation.degenerate_folds
    assert [v.view_id for v in back.held_out] == [v.view_id for v in validation.held_out]
    assert back.folds[0].camera.allclose(validation.folds[0].camera)


def test_zero_in_sample_rms_gives_an_infinite_ratio(pinhole, checkerboard):
    capture = synthesise(
        pinhole, checkerboard, diverse_poses(checkerboard, 14, seed=4),
        (1280, 720), noise_px=0.2, seed=17,
    )
    validation = cross_validate(session_of(capture))
    import dataclasses

    assert dataclasses.replace(validation, in_sample_rms=0.0).ratio == float("inf")
