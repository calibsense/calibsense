# caltrust - metric trust for camera calibration.
# Copyright (C) 2026 Abhishek Gola
#
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License, version 3, as published by
# the Free Software Foundation. This program is distributed WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the LICENSE file, or <https://www.gnu.org/licenses/>.

"""The `caltrust` command line.

Built on `argparse` rather than a third-party framework, and with a static
subcommand table, so that a PyInstaller build resolves every command without
dynamic discovery and the binary carries no dependency it does not need.

Exit codes: 0 success, 2 a problem with the input, 1 an unexpected failure, and
3 from `diagnose` when the calibration itself has a critical problem.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from typing import List, Optional, Sequence

from .._version import __version__
from ..core.camera import registered_models
from ..core.target import registered_targets
from ..errors import CalTrustError
from ..ingest import (
    DetectorOptions,
    detect_in_images,
    find_images,
    reader_names,
    registered_detectors,
    session_from_calibration,
    session_from_images,
)
from ..diagnose import diagnose
from ..io import load_fit, load_session, save_fit, save_session
from ..noisefloor import MIN_PRESENCE, measure_noise_floor
from ..refit import RefitOptions, instrument, set_single_threaded
from ..report import ReportMetadata, render_text, run_audit, write_json, write_pdf
from ..validate import cross_validate
from . import render
from .targets import SHORTHAND_HELP, resolve_target
from .tasks import SHORTHAND_HELP as TASK_SHORTHAND_HELP
from .tasks import parse_tasks

EXIT_OK = 0
EXIT_UNEXPECTED = 1
EXIT_INPUT = 2
#: `caltrust diagnose` returns this when it finds a critical problem, so a CI
#: step can gate on calibration quality without parsing the report.
EXIT_FINDINGS = 3


def _progress(index: int, total: int, view_id: str, error: Optional[str]) -> None:
    mark = "ok " if error is None else "..."
    detail = "" if error is None else f"  {error.split(':', 1)[-1].strip()}"
    print(f"  [{index:4d}/{total}] {mark} {view_id}{detail}", file=sys.stderr)


def _add_robot_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("hand-eye (optional)")
    group.add_argument(
        "--robot-poses",
        metavar="FILE",
        help="robot poses per view, as JSON or CSV, for a later hand-eye calibration",
    )
    group.add_argument(
        "--robot-convention",
        choices=("gripper2base", "base2gripper"),
        default="gripper2base",
        help="direction the robot poses are stored in (default: %(default)s)",
    )
    group.add_argument(
        "--robot-units",
        default="mm",
        help="length unit of the robot translations (default: %(default)s)",
    )


def _add_calibration_arguments(parser: argparse.ArgumentParser, required: bool) -> None:
    parser.add_argument(
        "--calibration",
        metavar="FILE",
        required=required,
        help="the calibration already in use, to audit against",
    )
    parser.add_argument(
        "--calibration-format",
        choices=[name for name, _ in reader_names()],
        help="force a reader instead of detecting the format",
    )


def _add_cross_validation_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("out-of-sample error")
    group.add_argument(
        "--cross-validate",
        action="store_true",
        help="measure held-out reprojection error by K-fold over views",
    )
    group.add_argument(
        "--folds",
        type=int,
        metavar="N",
        help="number of folds (default: 5, reduced if there are too few views)",
    )
    group.add_argument(
        "--no-shuffle",
        action="store_true",
        help="split views in capture order instead of shuffling first",
    )
    group.add_argument(
        "--fold-seed",
        type=int,
        default=0,
        metavar="N",
        help="seed for the fold shuffle (default: %(default)s)",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the full argument parser.

    Returns:
        The configured parser, subcommands included.
    """
    parser = argparse.ArgumentParser(
        prog="caltrust",
        description=(
            "Metric trust for camera calibration. Reports uncertainty, "
            "identifiability and residual structure, not just a reprojection number."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            f"target shorthand:\n  {SHORTHAND_HELP}\n\n"
            f"task shorthand:\n  {TASK_SHORTHAND_HELP}"
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=(
            f"caltrust {__version__}\n"
            "Copyright (C) 2026 Abhishek Gola\n"
            "Licence AGPL-3.0-only: GNU Affero GPL version 3 "
            "<https://www.gnu.org/licenses/agpl-3.0.html>\n"
            "This is free software with NO WARRANTY, to the extent permitted by law.\n"
            "Modifying it and offering network access to the result obliges you to "
            "publish your source."
        ),
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="pin OpenCV to one thread so results are bit-reproducible; "
             "without it, parallel reduction order makes them differ in the "
             "last few bits between runs",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

    ingest = subparsers.add_parser(
        "ingest", help="build a session from images, or from a calibration plus detections"
    )
    sources = ingest.add_subparsers(dest="source", metavar="SOURCE")

    images = sources.add_parser("images", help="detect a target across a folder of images")
    images.add_argument("--images", required=True, metavar="PATH", help="image folder or single image")
    images.add_argument("--target", required=True, help="target file, or shorthand")
    images.add_argument("-o", "--output", required=True, metavar="FILE", help="session bundle to write")
    images.add_argument("--no-recursive", action="store_true", help="do not descend into subfolders")
    images.add_argument("--min-points", type=int, default=8, metavar="N",
                        help="fewest points a view must yield (default: %(default)s)")
    images.add_argument("--no-refine", action="store_true", help="skip sub-pixel refinement")
    images.add_argument("--fast", action="store_true", help="favour speed over detection rate")
    images.add_argument("--quiet", action="store_true", help="do not print per-image progress")
    _add_calibration_arguments(images, required=False)
    _add_robot_arguments(images)
    images.set_defaults(handler=_ingest_images)

    calib = sources.add_parser(
        "calibration", help="audit an existing calibration using the detections behind it"
    )
    calib.add_argument("--detections", required=True, metavar="FILE",
                       help="corner detections, as caltrust JSON or an npz")
    calib.add_argument("--target", help="target file or shorthand, if the detections do not name one")
    calib.add_argument("-o", "--output", required=True, metavar="FILE", help="session bundle to write")
    _add_calibration_arguments(calib, required=True)
    _add_robot_arguments(calib)
    calib.set_defaults(handler=_ingest_calibration)

    refit = subparsers.add_parser("refit", help="refit with full instrumentation")
    refit.add_argument("session", metavar="SESSION", help="session bundle from ingest")
    refit.add_argument("-o", "--output", metavar="FILE", help="fit bundle to write")
    refit.add_argument("--model", choices=("pinhole", "fisheye"),
                       help="camera model (default: from the session's calibration, else pinhole)")
    refit.add_argument("--distortion-terms", type=int, default=5, choices=(4, 5, 8, 12, 14),
                       help="Brown-Conrady coefficients to estimate (default: %(default)s)")
    refit.add_argument("--fix", action="append", default=[], metavar="PARAM",
                       help="hold a parameter fixed; repeatable, e.g. --fix cx --fix cy")
    refit.add_argument("--tie-aspect", action="store_true", help="hold fy/fx constant")
    refit.add_argument("--no-refit", action="store_true",
                       help="instrument the session's existing calibration in place")
    refit.add_argument("--ignore-prior", action="store_true",
                       help="do not start the optimiser from the session's calibration")
    refit.add_argument("--radial-bins", type=int, default=10, metavar="N",
                       help="bins in the radial residual profile (default: %(default)s)")
    refit.add_argument("--check-condition", action="store_true",
                       help="make OpenCV reject ill-conditioned fisheye views")
    refit.add_argument("--rcond", type=float, default=1e-12, metavar="X",
                       help="relative eigenvalue cut (default: %(default)s)")
    _add_cross_validation_arguments(refit)
    refit.add_argument("-v", "--verbose", action="store_true",
                       help="every view and the full correlation matrix")
    refit.add_argument("--json", action="store_true", help="emit JSON instead of text")
    refit.set_defaults(handler=_refit)

    diagnose_parser = subparsers.add_parser(
        "diagnose",
        help="refit, cross-validate and name every degeneracy and coverage problem",
    )
    diagnose_parser.add_argument("session", metavar="SESSION", help="session bundle")
    diagnose_parser.add_argument("-o", "--output", metavar="FILE",
                                 help="fit bundle to write")
    diagnose_parser.add_argument("--model", choices=("pinhole", "fisheye"),
                                 help="camera model (default: from the session)")
    diagnose_parser.add_argument("--distortion-terms", type=int, default=5,
                                 choices=(4, 5, 8, 12, 14),
                                 help="Brown-Conrady coefficients (default: %(default)s)")
    diagnose_parser.add_argument("--fix", action="append", default=[], metavar="PARAM",
                                 help="hold a parameter fixed; repeatable")
    diagnose_parser.add_argument("--tie-aspect", action="store_true",
                                 help="hold fy/fx constant")
    diagnose_parser.add_argument("--all", action="store_true",
                                 help="list the diagnostics that found nothing too")
    diagnose_parser.add_argument("--json", action="store_true",
                                 help="emit JSON instead of text")
    diagnose_parser.add_argument("-v", "--verbose", action="store_true",
                                 help="print the full refit report as well")
    _add_cross_validation_arguments(diagnose_parser)
    diagnose_parser.set_defaults(handler=_diagnose)

    report = subparsers.add_parser(
        "report",
        help="one run: refit, cross-validate, diagnose, propagate to millimetres, "
             "and write a JSON and a PDF",
    )
    report.add_argument("session", metavar="SESSION", help="session bundle")
    report.add_argument("--pdf", metavar="FILE", help="write the PDF here")
    report.add_argument("--json", dest="json_path", metavar="FILE",
                        help="write the machine-readable JSON here")
    report.add_argument("--task", action="append", default=[], metavar="SPEC",
                        help="measurement to propagate; repeatable. "
                             f"{TASK_SHORTHAND_HELP}")
    report.add_argument("--model", choices=("pinhole", "fisheye"),
                        help="camera model (default: from the session)")
    report.add_argument("--distortion-terms", type=int, default=5,
                        choices=(4, 5, 8, 12, 14),
                        help="Brown-Conrady coefficients (default: %(default)s)")
    report.add_argument("--fix", action="append", default=[], metavar="PARAM",
                        help="hold a parameter fixed; repeatable")
    report.add_argument("--tie-aspect", action="store_true", help="hold fy/fx constant")
    report.add_argument("--mounting", choices=("eye_in_hand", "eye_to_hand"),
                        help="solve hand-eye with this mounting; needs robot poses")
    report.add_argument("--samples", type=int, default=2000, metavar="N",
                        help="Monte Carlo samples per task (default: %(default)s)")
    report.add_argument("--folds", type=int, metavar="N",
                        help="cross-validation folds (default: 5)")
    report.add_argument("--seed", type=int, default=0, metavar="N",
                        help="seed, so the report is reproducible (default: %(default)s)")
    report.add_argument("--camera-name", default="unnamed camera", metavar="NAME",
                        help="how this camera is known in the plant")
    report.add_argument("--title", default="Camera calibration measurement audit",
                        metavar="TEXT", help="report title")
    report.add_argument("--prepared-by", default="", metavar="NAME",
                        help="who ran the audit")
    report.add_argument("--contact", metavar="TEXT",
                        help="where to send questions; printed as the last line")
    report.add_argument("--open-question", metavar="TEXT",
                        help="the question the report puts back to the reader")
    report.add_argument("--noise-px", type=float, metavar="SIGMA",
                        help="pixel noise to propagate, instead of the one inferred "
                             "from this fit's residuals; take it from the `use` line "
                             "of `caltrust noise-floor`")
    report.set_defaults(handler=_report)

    noise = subparsers.add_parser(
        "noise-floor",
        help="measure detector noise directly, from frames of a static scene",
        description="Point a fixed camera at a fixed target, capture thirty or "
                    "more frames without touching either, and this reports what "
                    "the detector's noise actually is. Everything else in "
                    "caltrust infers sigma from fit residuals, which assumes the "
                    "model is right; this does not.",
    )
    noise.add_argument("--images", required=True, metavar="PATH",
                       help="folder of static frames, or a single image")
    noise.add_argument("--target", required=True, help="target file, or shorthand")
    noise.add_argument("-o", "--output", metavar="FILE", help="write the measurement as JSON")
    noise.add_argument("--json", action="store_true", help="emit JSON on stdout instead of text")
    noise.add_argument("--no-recursive", action="store_true", help="do not descend into subfolders")
    noise.add_argument("--min-points", type=int, default=8, metavar="N",
                       help="fewest points a frame must yield (default: %(default)s)")
    noise.add_argument("--no-refine", action="store_true",
                       help="skip sub-pixel refinement, to measure the raw detector")
    noise.add_argument("--fast", action="store_true", help="favour speed over detection rate")
    noise.add_argument("--min-presence", type=float, default=MIN_PRESENCE, metavar="F",
                       help="fraction of frames a corner must appear in (default: %(default)s)")
    noise.add_argument("--quiet", action="store_true", help="do not print per-image progress")
    noise.set_defaults(handler=_noise_floor)

    show = subparsers.add_parser("show", help="print a session or a fit bundle")
    show.add_argument("path", metavar="FILE", help="session or fit bundle")
    show.add_argument("-v", "--verbose", action="store_true", help="every view and the full matrix")
    show.add_argument("--json", action="store_true", help="emit JSON instead of text")
    show.set_defaults(handler=_show)

    formats = subparsers.add_parser("formats", help="list supported targets, models and file formats")
    formats.set_defaults(handler=_formats)
    return parser


def _detector_options(args: argparse.Namespace) -> DetectorOptions:
    return DetectorOptions(
        refine=not args.no_refine, min_points=args.min_points, fast=args.fast
    )


def _ingest_images(args: argparse.Namespace) -> int:
    session = session_from_images(
        images=args.images,
        target=resolve_target(args.target),
        options=_detector_options(args),
        calibration=args.calibration,
        calibration_format=args.calibration_format,
        robot_poses=args.robot_poses,
        robot_convention=args.robot_convention,
        robot_units=args.robot_units,
        recursive=not args.no_recursive,
        progress=None if args.quiet else _progress,
    )
    path = save_session(session, args.output)
    print(render.render_session(session), end="")
    print(f"\nwrote {path}")
    return EXIT_OK


def _ingest_calibration(args: argparse.Namespace) -> int:
    session = session_from_calibration(
        calibration=args.calibration,
        detections=args.detections,
        target=resolve_target(args.target) if args.target else None,
        calibration_format=args.calibration_format,
        robot_poses=args.robot_poses,
        robot_convention=args.robot_convention,
        robot_units=args.robot_units,
    )
    path = save_session(session, args.output)
    print(render.render_session(session), end="")
    print(f"\nwrote {path}")
    return EXIT_OK


def _refit(args: argparse.Namespace) -> int:
    session = load_session(args.session)
    options = RefitOptions(
        model=args.model,
        distortion_terms=args.distortion_terms,
        fixed=tuple(args.fix),
        tie_aspect=args.tie_aspect,
        refit=not args.no_refit,
        use_prior_as_guess=not args.ignore_prior,
        rcond=args.rcond,
        radial_bins=args.radial_bins,
        check_condition=args.check_condition,
    )
    fit = _fit_with_validation(session, options, args)
    if args.json:
        print(json.dumps(render.fit_to_json(fit), indent=2))
    else:
        print(render.render_fit(fit, args.verbose), end="")
    if args.output:
        path = save_fit(fit, args.output)
        print(f"\nwrote {path}", file=sys.stderr if args.json else sys.stdout)
    return EXIT_OK


def _fit_with_validation(session, options: RefitOptions, args) -> "InstrumentedFit":
    """Instrument a session, attaching cross-validation when it was requested."""
    fit = instrument(session, options)
    if not getattr(args, "cross_validate", False) and getattr(args, "folds", None) is None:
        return fit
    validation = cross_validate(
        session,
        options,
        folds=args.folds,
        shuffle=not args.no_shuffle,
        seed=args.fold_seed,
    )
    return dataclasses.replace(fit, cross_validation=validation)


def _diagnose(args: argparse.Namespace) -> int:
    session = load_session(args.session)
    options = RefitOptions(
        model=args.model,
        distortion_terms=args.distortion_terms,
        fixed=tuple(args.fix),
        tie_aspect=args.tie_aspect,
    )
    # Cross-validation is the point of the command, so it is on unless the fold
    # count makes it impossible.
    args.cross_validate = True
    try:
        fit = _fit_with_validation(session, options, args)
    except CalTrustError as exc:
        print(f"caltrust: out-of-sample error not measured: {exc}", file=sys.stderr)
        fit = instrument(session, options)
    diagnosis = diagnose(fit, session.observations, fit.cross_validation)
    if args.json:
        print(json.dumps({
            "fit": render.fit_to_json(fit),
            "diagnosis": render.diagnosis_to_json(diagnosis),
        }, indent=2))
    else:
        if args.verbose:
            print(render.render_fit(fit, verbose=True), end="")
            print()
        print(render.render_diagnosis(diagnosis, include_ok=args.all), end="")
    if args.output:
        path = save_fit(fit, args.output)
        print(f"\nwrote {path}", file=sys.stderr if args.json else sys.stdout)
    return EXIT_OK if diagnosis.severity.name != "CRITICAL" else EXIT_FINDINGS


def _report(args: argparse.Namespace) -> int:
    session = load_session(args.session)
    options = RefitOptions(
        model=args.model,
        distortion_terms=args.distortion_terms,
        fixed=tuple(args.fix),
        tie_aspect=args.tie_aspect,
    )
    metadata_fields = {
        "title": args.title,
        "camera_name": args.camera_name,
        "prepared_by": args.prepared_by,
    }
    if args.contact:
        metadata_fields["contact"] = args.contact
    if args.open_question:
        metadata_fields["open_question"] = args.open_question

    # A base-frame task needs the hand-eye solve to exist first, so the audit
    # runs once with the default task to obtain it, then again with the real
    # tasks. Every other task needs only the fit.
    needs_hand_eye = any(
        spec.split(":")[0].strip().lower() in ("base", "robot") for spec in args.task
    )
    tasks = None
    if args.task and not needs_hand_eye:
        tasks = parse_tasks(args.task)

    audit = run_audit(
        session, options, tasks, ReportMetadata(**metadata_fields),
        folds=args.folds, n_samples=args.samples, mounting=args.mounting,
        observation_noise_px=args.noise_px,
        seed=args.seed,
    )
    if needs_hand_eye:
        if audit.hand_eye is None:
            print(
                "caltrust: a base-frame task needs a hand-eye solve; pass "
                "--mounting and use a session that carries robot poses",
                file=sys.stderr,
            )
            return EXIT_INPUT
        flange = session.robot.aligned_with(session.observations)[0]
        audit = run_audit(
            session, options, parse_tasks(args.task, audit.hand_eye, flange),
            ReportMetadata(**metadata_fields), folds=args.folds,
            n_samples=args.samples, mounting=args.mounting, seed=args.seed,
        )

    print(render_text(audit), end="")
    written = []
    if args.pdf:
        written.append(write_pdf(audit, args.pdf))
    if args.json_path:
        written.append(write_json(audit, args.json_path))
    for path in written:
        print(f"wrote {path}")
    if not written:
        print(
            "\nnothing written; pass --pdf and/or --json to save the report",
            file=sys.stderr,
        )
    return EXIT_OK if audit.severity.name != "CRITICAL" else EXIT_FINDINGS


def _show(args: argparse.Namespace) -> int:
    from ..io.bundle import read_bundle
    from ..io.fit_io import FORMAT as FIT_FORMAT

    manifest, _ = read_bundle(args.path)
    if manifest.get("format") == FIT_FORMAT:
        fit = load_fit(args.path)
        payload = render.fit_to_json(fit) if args.json else None
        print(json.dumps(payload, indent=2) if args.json
              else render.render_fit(fit, args.verbose), end="" if not args.json else "\n")
    else:
        session = load_session(args.path)
        print(json.dumps(render.session_to_json(session), indent=2) if args.json
              else render.render_session(session), end="" if not args.json else "\n")
    return EXIT_OK


def _noise_floor(args: argparse.Namespace) -> int:
    observations = detect_in_images(
        find_images(args.images, recursive=not args.no_recursive),
        resolve_target(args.target),
        options=_detector_options(args),
        progress=None if args.quiet else _progress,
    )
    floor = measure_noise_floor(observations, min_presence=args.min_presence)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(floor.to_dict(), handle, indent=2)
    if args.json:
        print(json.dumps(floor.to_dict(), indent=2))
    else:
        if not args.quiet:
            print()
        print("\n".join(floor.summary_lines()))
        if args.output:
            print(f"\nwrote {args.output}")
    return EXIT_OK if floor.trustworthy_covariance else EXIT_FINDINGS


def _formats(args: argparse.Namespace) -> int:
    print("targets      " + ", ".join(registered_targets()))
    print("detectors    " + ", ".join(registered_detectors()))
    print("models       " + ", ".join(registered_models()))
    print("calibrations")
    for name, description in reader_names():
        print(f"  {name:10s} {description}")
    print("\ntarget shorthand")
    print(f"  {SHORTHAND_HELP}")
    return EXIT_OK


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the command line.

    Args:
        argv: Arguments to parse; `sys.argv[1:]` when omitted.

    Returns:
        The process exit code.
    """
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if getattr(args, "deterministic", False):
        set_single_threaded(True)
    if not getattr(args, "handler", None):
        # A bare `caltrust`, or `caltrust ingest` with no source.
        (parser if args.command is None else parser).print_help()
        return EXIT_INPUT
    try:
        return args.handler(args)
    except CalTrustError as exc:
        print(f"caltrust: {exc}", file=sys.stderr)
        return EXIT_INPUT
    except KeyboardInterrupt:
        print("caltrust: interrupted", file=sys.stderr)
        return EXIT_UNEXPECTED


if __name__ == "__main__":
    sys.exit(main())
