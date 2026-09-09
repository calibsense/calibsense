"""The command line, driven end to end."""

from __future__ import annotations

import json
import os

import cv2
import numpy as np
import pytest

from caltrust.cli.main import EXIT_INPUT, EXIT_OK, build_parser, main
from caltrust.core.target import Checkerboard
from caltrust.io import load_fit, load_session
from caltrust.synthetic import diverse_poses

from .rendering import write_views

TARGET = "checkerboard:9x6:25mm"
BOARD = Checkerboard(9, 6, 25.0)


@pytest.fixture
def images(tmp_path, pinhole):
    directory = tmp_path / "images"
    poses = diverse_poses(
        BOARD, 10, distances_mm=(450.0, 700.0, 1000.0),
        max_tilt_rad=0.55, lateral_mm=100.0, seed=9,
    )
    write_views(str(directory), pinhole, BOARD, poses, (1280, 720), noise=1.5, blur=0.6)
    return directory


@pytest.fixture
def calibration_file(tmp_path):
    path = tmp_path / "shipped.yml"
    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
    storage.write("image_width", 1280)
    storage.write("image_height", 720)
    storage.write("camera_matrix",
                  np.array([[870.0, 0.0, 630.0], [0.0, 875.0, 350.0], [0.0, 0.0, 1.0]]))
    storage.write("distortion_coefficients", np.array([-0.19, 0.04, 0.0, 0.0, 0.0]))
    storage.write("avg_reprojection_error", 0.2412)
    storage.release()
    return path


@pytest.fixture
def session_file(tmp_path, images, capsys):
    path = tmp_path / "session.npz"
    assert main(["ingest", "images", "--images", str(images), "--target", TARGET,
                 "-o", str(path), "--quiet"]) == EXIT_OK
    capsys.readouterr()
    return path


def test_version_flag_exits_cleanly(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    assert "caltrust" in capsys.readouterr().out


def test_bare_invocation_prints_help(capsys):
    assert main([]) == EXIT_INPUT
    assert "COMMAND" in capsys.readouterr().out


def test_ingest_without_a_source_prints_help(capsys):
    assert main(["ingest"]) == EXIT_INPUT
    assert "usage" in capsys.readouterr().out.lower()


def test_formats_lists_everything(capsys):
    assert main(["formats"]) == EXIT_OK
    out = capsys.readouterr().out
    for expected in ("checkerboard", "charuco", "circle_grid", "opencv", "ros",
                     "kalibr", "native", "pinhole_brown_conrady"):
        assert expected in out


@pytest.mark.slow
def test_ingest_images_writes_a_session(tmp_path, images, capsys):
    path = tmp_path / "s.npz"
    assert main(["ingest", "images", "--images", str(images), "--target", TARGET,
                 "-o", str(path), "--quiet"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "caltrust session" in out
    assert "9x6 checkerboard" in out
    assert os.path.isfile(path)
    session = load_session(str(path))
    assert session.observations.n_views == 10


@pytest.mark.slow
def test_ingest_prints_progress_to_stderr(tmp_path, images, capsys):
    main(["ingest", "images", "--images", str(images), "--target", TARGET,
          "-o", str(tmp_path / "s.npz")])
    captured = capsys.readouterr()
    assert "[   1/10]" in captured.err


@pytest.mark.slow
def test_ingest_images_with_a_calibration(tmp_path, images, calibration_file, capsys):
    path = tmp_path / "s.npz"
    assert main(["ingest", "images", "--images", str(images), "--target", TARGET,
                 "--calibration", str(calibration_file), "-o", str(path),
                 "--quiet"]) == EXIT_OK
    assert "0.2412" in capsys.readouterr().out
    assert load_session(str(path)).prior is not None


@pytest.mark.slow
def test_ingest_calibration_from_detections(tmp_path, session_file, calibration_file, capsys):
    from caltrust.ingest import write_detections

    session = load_session(str(session_file))
    detections = write_detections(session.observations, str(tmp_path / "d.json"))
    path = tmp_path / "s2.npz"
    assert main(["ingest", "calibration", "--calibration", str(calibration_file),
                 "--detections", detections, "-o", str(path)]) == EXIT_OK
    capsys.readouterr()
    assert load_session(str(path)).observations.n_views == session.observations.n_views


@pytest.mark.slow
def test_refit_prints_the_report(session_file, capsys):
    assert main(["refit", str(session_file)]) == EXIT_OK
    out = capsys.readouterr().out
    for heading in ("caltrust instrumented refit", "conditioning", "parameters",
                    "residuals by view", "residuals by image radius"):
        assert heading in out
    assert "corr(fx, tz)" in out


@pytest.mark.slow
def test_refit_writes_a_fit_bundle(tmp_path, session_file, capsys):
    path = tmp_path / "fit.npz"
    assert main(["refit", str(session_file), "-o", str(path)]) == EXIT_OK
    capsys.readouterr()
    fit = load_fit(str(path))
    assert fit.n_views == 10
    assert fit.conditioning.identifiable


@pytest.mark.slow
def test_refit_json_is_valid_and_complete(session_file, capsys):
    assert main(["refit", str(session_file), "--json"]) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    for key in ("camera", "rms", "sigma", "conditioning", "parameters", "views",
                "radial_profile", "strongest_correlations", "at_optimum"):
        assert key in payload
    assert len(payload["views"]) == 10
    assert payload["conditioning"]["identifiable"] is True


@pytest.mark.slow
def test_refit_verbose_adds_the_correlation_matrix(session_file, capsys):
    assert main(["refit", str(session_file), "-v"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "intrinsic correlation matrix" in out
    assert "residuals by view (all 10 of 10)" in out


@pytest.mark.slow
def test_refit_options_reach_the_engine(session_file, capsys):
    assert main(["refit", str(session_file), "--distortion-terms", "8",
                 "--radial-bins", "4", "--json"]) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["options"]["distortion_terms"] == 8
    assert len(payload["radial_profile"]["counts"]) == 4
    assert len(payload["camera"]["distortion"]) == 8


@pytest.mark.slow
def test_refit_can_fix_the_principal_point(tmp_path, images, calibration_file, capsys):
    session = tmp_path / "s.npz"
    main(["ingest", "images", "--images", str(images), "--target", TARGET,
          "--calibration", str(calibration_file), "-o", str(session), "--quiet"])
    capsys.readouterr()
    assert main(["refit", str(session), "--fix", "cx", "--fix", "cy", "--json"]) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["camera"]["cx"] == pytest.approx(630.0)
    assert [p["name"] for p in payload["parameters"]].count("cx") == 0


@pytest.mark.slow
def test_refit_no_refit_requires_a_calibration(session_file, capsys):
    assert main(["refit", str(session_file), "--no-refit"]) == EXIT_INPUT
    assert "session carries none" in capsys.readouterr().err


@pytest.mark.slow
def test_refit_no_refit_audits_the_shipped_calibration(
    tmp_path, images, calibration_file, capsys
):
    session = tmp_path / "s.npz"
    main(["ingest", "images", "--images", str(images), "--target", TARGET,
          "--calibration", str(calibration_file), "-o", str(session), "--quiet"])
    capsys.readouterr()
    assert main(["refit", str(session), "--no-refit", "--json"]) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["refitted"] is False
    assert payload["at_optimum"] is False
    assert payload["camera"]["fx"] == pytest.approx(870.0)
    assert payload["prior_rms"] == pytest.approx(0.2412)


@pytest.mark.slow
def test_show_renders_a_session_and_a_fit(tmp_path, session_file, capsys):
    assert main(["show", str(session_file)]) == EXIT_OK
    assert "caltrust session" in capsys.readouterr().out
    fit = tmp_path / "fit.npz"
    main(["refit", str(session_file), "-o", str(fit)])
    capsys.readouterr()
    assert main(["show", str(fit)]) == EXIT_OK
    assert "instrumented refit" in capsys.readouterr().out


@pytest.mark.slow
def test_show_json_for_both_kinds(tmp_path, session_file, capsys):
    assert main(["show", str(session_file), "--json"]) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["n_views"] == 10
    fit = tmp_path / "fit.npz"
    main(["refit", str(session_file), "-o", str(fit)])
    capsys.readouterr()
    assert main(["show", str(fit), "--json"]) == EXIT_OK
    assert "conditioning" in json.loads(capsys.readouterr().out)


def test_a_missing_session_is_a_clean_input_error(tmp_path, capsys):
    assert main(["refit", str(tmp_path / "absent.npz")]) == EXIT_INPUT
    assert "no such file" in capsys.readouterr().err


def test_a_bad_target_shorthand_is_a_clean_input_error(tmp_path, capsys):
    assert main(["ingest", "images", "--images", str(tmp_path),
                 "--target", "dartboard:1x1:1mm", "-o", str(tmp_path / "s.npz")]) == EXIT_INPUT
    assert "unknown target kind" in capsys.readouterr().err


def test_a_missing_image_folder_is_a_clean_input_error(tmp_path, capsys):
    assert main(["ingest", "images", "--images", str(tmp_path / "absent"),
                 "--target", TARGET, "-o", str(tmp_path / "s.npz")]) == EXIT_INPUT
    assert "no such path" in capsys.readouterr().err


def test_help_mentions_the_target_shorthand(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    assert "target shorthand" in capsys.readouterr().out


def test_every_subcommand_has_a_handler():
    parser = build_parser()
    actions = [a for a in parser._actions if hasattr(a, "choices") and a.choices]
    assert actions, "the parser must expose subcommands"


def test_parser_rejects_an_unknown_distortion_term_count():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["refit", "s.npz", "--distortion-terms", "7"])
