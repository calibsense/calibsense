"""M2 — refit with instrumentation.

One function, `instrument`, does the whole milestone: it re-runs the calibration
(or evaluates the one the engineer already has) and keeps every quantity the
standard OpenCV call computes internally and then throws away.

The OpenCV flag mapping is deliberately strict. `cv2.calibrateCamera` can hold
the principal point fixed but not `cx` alone, and can tie the aspect ratio but
not tie an arbitrary pair. Asking for something it cannot express raises rather
than silently fitting a different model than the one requested.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..core.camera import (
    CameraModel,
    FisheyeKannalaBrandt,
    PinholeBrownConrady,
)
from ..core.observations import ObservationSet
from ..core.parameters import ParameterBlock
from ..core.poses import Pose
from ..core.session import CalibrationSession
from ..errors import RefitError, ValidationError
from . import residuals as residuals_mod
from .cv_compat import fisheye_flag, pinhole_flag
from .covariance import covariance_from_normal_equations
from .normal import assemble
from .projection import MIN_POINTS_FOR_POSE, projector_for
from .result import (
    Conditioning,
    InstrumentedFit,
    RefitOptions,
    conditioning_from_covariance,
)

#: Optimiser stopping criteria. Tighter than OpenCV's default, because a
#: covariance is a statement about curvature at the optimum and a fit stopped
#: early is not at one.
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-10)

def _term_flags(terms: int) -> int:
    """OpenCV flags that switch on the higher distortion terms."""
    flags = 0
    if terms >= 8:
        flags |= pinhole_flag("CALIB_RATIONAL_MODEL")
    if terms >= 12:
        flags |= pinhole_flag("CALIB_THIN_PRISM_MODEL")
    if terms >= 14:
        flags |= pinhole_flag("CALIB_TILTED_MODEL")
    return flags


_PINHOLE_FIX_NAMES = ("k1", "k2", "k3", "k4", "k5", "k6")
_FISHEYE_FIX_NAMES = ("k1", "k2", "k3", "k4")


def instrument(
    session: CalibrationSession, options: Optional[RefitOptions] = None
) -> InstrumentedFit:
    """Refit a calibration and record everything the standard call discards.

    Args:
        session: The ingested session, from `caltrust.ingest`.
        options: Refit settings. Defaults re-estimate a five-coefficient
            pinhole model with every parameter free.

    Returns:
        The fit together with its covariance, residual distributions and
        conditioning report.

    Raises:
        RefitError: The calibration could not be estimated, or `refit=False` was
            asked for without a calibration to instrument.
        ValidationError: The requested parameter constraints cannot be expressed.
        DegenerateSystemError: There are not enough observations to estimate any
            variance at all.
    """
    options = options or RefitOptions()
    observations = session.observations
    model = _resolve_model(session, options)
    _check_capacity(observations, model, options)

    if options.refit:
        camera, poses, guess = _run_calibration(observations, model, options, session)
    else:
        guess = None
        if session.prior is None:
            raise RefitError(
                "refit=False instruments an existing calibration, but this "
                "session carries none; ingest one with --calibration"
            )
        camera = session.prior.camera
        poses = _solve_poses(camera, observations)

    block = _parameter_block(camera, options)
    equations, per_view = assemble(camera, poses, observations, block)
    covariance = covariance_from_normal_equations(equations, options.rcond)
    statistics = residuals_mod.summarise(
        observations, per_view, (camera.cx, camera.cy), options.radial_bins
    )
    decrement = covariance.newton_decrement(
        equations.gradient_intrinsic, equations.gradient_pose
    )
    return InstrumentedFit(
        camera=camera,
        poses=tuple(poses),
        view_ids=observations.view_ids,
        image_size=observations.image_size,
        covariance=covariance,
        residuals=statistics,
        conditioning=conditioning_from_covariance(covariance),
        equations=equations,
        refitted=options.refit,
        options=options,
        prior_rms=session.prior.reported_rms if session.prior else None,
        relative_decrement=decrement / max(equations.cost, 1e-30),
        initial_guess=guess,
    )


def _resolve_model(session: CalibrationSession, options: RefitOptions) -> str:
    if options.model is not None:
        return options.model
    if session.prior is not None:
        return (
            "fisheye"
            if isinstance(session.prior.camera, FisheyeKannalaBrandt)
            else "pinhole"
        )
    return "pinhole"


def _n_intrinsic(model: str, options: RefitOptions) -> int:
    if model == "fisheye":
        return 9
    terms = 5 if options.distortion_terms == 4 else options.distortion_terms
    return 4 + terms


def _check_capacity(
    observations: ObservationSet, model: str, options: RefitOptions
) -> None:
    thin = [
        v.view_id for v in observations.views if v.n_points < MIN_POINTS_FOR_POSE
    ]
    if thin:
        raise RefitError(
            f"{len(thin)} view(s) have fewer than {MIN_POINTS_FOR_POSE} points and "
            f"cannot constrain a pose, first is {thin[0]!r}; drop them with "
            "ObservationSet.filter_min_points"
        )
    residual_count = 2 * observations.total_points
    parameter_count = _n_intrinsic(model, options) + 6 * observations.n_views
    if residual_count <= parameter_count:
        raise RefitError(
            f"{observations.n_views} views with {observations.total_points} points "
            f"give {residual_count} residuals against up to {parameter_count} "
            "parameters; there is nothing left to estimate uncertainty from"
        )


def _parameter_block(camera: CameraModel, options: RefitOptions) -> ParameterBlock:
    fixed = set(options.fixed)
    if isinstance(camera, FisheyeKannalaBrandt):
        # OpenCV fixes skew unless explicitly told otherwise, and a skew
        # estimated from a normal capture is noise. It stays fixed here too, so
        # the covariance is not inflated by a parameter nobody estimated.
        fixed.add("alpha")
    elif options.distortion_terms == 4:
        fixed.add("k3")
    return camera.free_parameters(tuple(sorted(fixed)), options.tie_aspect)


def _initial_guess(
    observations: ObservationSet, session: CalibrationSession, options: RefitOptions
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if not options.use_prior_as_guess or session.prior is None:
        return None, None
    camera = session.prior.camera
    return camera.camera_matrix.copy(), np.asarray(camera.distortion, dtype=float).copy()


def _guess_matrix(observations: ObservationSet, focal: float) -> np.ndarray:
    width, height = observations.image_size
    return np.array(
        [[focal, 0.0, (width - 1) / 2.0],
         [0.0, focal, (height - 1) / 2.0],
         [0.0, 0.0, 1.0]]
    )


def _default_guess(observations: ObservationSet) -> np.ndarray:
    """A pinhole starting point: focal equal to the long image side."""
    return _guess_matrix(observations, float(max(observations.image_size)))


def _fisheye_guess_ladder(
    observations: ObservationSet, prior: Optional[np.ndarray]
) -> List[Tuple[str, np.ndarray]]:
    """Focal-length starting points to try for a fisheye fit, best first.

    `cv2.fisheye.calibrate` estimates its own starting intrinsics when not given
    any, and that estimate fails outright on ordinary captures — it raises
    `fabs(norm_u1) > 0` from `InitExtrinsics` rather than returning a poor
    answer. A guess near the truth converges, and a guess several times too
    large fails the same way, so caltrust supplies its own ladder instead of
    trusting the internal one.

    For an equidistant projection the image radius is `f * theta`, so a lens
    covering a full hemisphere puts `f` near `width / pi`. The ladder spans that
    down to a much longer lens.

    Args:
        observations: The capture, for its image size.
        prior: The calibration matrix already in hand, tried first when present.

    Returns:
        Named starting matrices, in the order they should be attempted.
    """
    width = float(observations.image_size[0])
    ladder: List[Tuple[str, np.ndarray]] = []
    if prior is not None:
        ladder.append(("prior", prior.copy()))
    for label, focal in (
        ("width/pi", width / np.pi),
        ("width/2.5", width / 2.5),
        ("width/4", width / 4.0),
        ("width/6", width / 6.0),
        ("width/1.5", width / 1.5),
    ):
        ladder.append((label, _guess_matrix(observations, focal)))
    return ladder


def _pinhole_flags(options: RefitOptions, has_guess: bool) -> int:
    fixed = set(options.fixed)
    terms = 5 if options.distortion_terms == 4 else options.distortion_terms
    flags = _term_flags(terms)
    if options.distortion_terms == 4:
        fixed.add("k3")

    if options.tie_aspect:
        flags |= pinhole_flag("CALIB_FIX_ASPECT_RATIO")
    if {"fx", "fy"} & fixed:
        if not {"fx", "fy"} <= fixed:
            raise ValidationError(
                "cv2.calibrateCamera can fix both focal lengths or neither; "
                "fixing only one is not expressible"
            )
        flags |= pinhole_flag("CALIB_FIX_FOCAL_LENGTH")
    if {"cx", "cy"} & fixed:
        if not {"cx", "cy"} <= fixed:
            raise ValidationError(
                "cv2.calibrateCamera can fix both principal point coordinates or "
                "neither; fixing only one is not expressible"
            )
        flags |= pinhole_flag("CALIB_FIX_PRINCIPAL_POINT")
    if {"p1", "p2"} & fixed:
        if not {"p1", "p2"} <= fixed:
            raise ValidationError(
                "cv2.calibrateCamera zeroes both tangential terms or neither; "
                "fixing only one is not expressible"
            )
        flags |= pinhole_flag("CALIB_ZERO_TANGENT_DIST")
    for name in _PINHOLE_FIX_NAMES:
        if name in fixed:
            flags |= pinhole_flag(f"CALIB_FIX_{name.upper()}")
    unsupported = fixed - {"fx", "fy", "cx", "cy", "p1", "p2"} - set(_PINHOLE_FIX_NAMES)
    if unsupported:
        raise ValidationError(
            f"cv2.calibrateCamera cannot hold {sorted(unsupported)} fixed during a "
            "refit; either drop them from --fix or use --no-refit to instrument "
            "an existing calibration in place"
        )
    # Fixing a parameter only means anything if OpenCV is told what value to
    # fix it at, which it reads from the intrinsic guess.
    pins = (
        pinhole_flag("CALIB_FIX_FOCAL_LENGTH")
        | pinhole_flag("CALIB_FIX_PRINCIPAL_POINT")
        | pinhole_flag("CALIB_FIX_ASPECT_RATIO")
    )
    if flags & pins or has_guess:
        flags |= pinhole_flag("CALIB_USE_INTRINSIC_GUESS")
    return flags


def _fisheye_flags(options: RefitOptions, has_guess: bool) -> int:
    fixed = set(options.fixed)
    # Skew is always fixed, matching _parameter_block, so the covariance is not
    # inflated by a parameter nobody estimated.
    flags = fisheye_flag("CALIB_RECOMPUTE_EXTRINSIC") | fisheye_flag("CALIB_FIX_SKEW")
    if options.check_condition:
        flags |= fisheye_flag("CALIB_CHECK_COND")
    if {"fx", "fy"} & fixed:
        if not {"fx", "fy"} <= fixed:
            raise ValidationError(
                "cv2.fisheye.calibrate can fix both focal lengths or neither"
            )
        flags |= fisheye_flag("CALIB_FIX_FOCAL_LENGTH")
    if {"cx", "cy"} & fixed:
        if not {"cx", "cy"} <= fixed:
            raise ValidationError(
                "cv2.fisheye.calibrate can fix both principal point coordinates "
                "or neither"
            )
        flags |= fisheye_flag("CALIB_FIX_PRINCIPAL_POINT")
    for name in _FISHEYE_FIX_NAMES:
        if name in fixed:
            flags |= fisheye_flag(f"CALIB_FIX_{name.upper()}")
    unsupported = fixed - {"fx", "fy", "cx", "cy", "alpha"} - set(_FISHEYE_FIX_NAMES)
    if unsupported:
        raise ValidationError(
            f"cv2.fisheye.calibrate cannot hold {sorted(unsupported)} fixed; "
            f"it estimates only fx, fy, cx, cy and k1..k4"
        )
    if has_guess:
        flags |= fisheye_flag("CALIB_USE_INTRINSIC_GUESS")
    return flags


def _run_calibration(
    observations: ObservationSet,
    model: str,
    options: RefitOptions,
    session: CalibrationSession,
) -> Tuple[CameraModel, List[Pose], str]:
    target = observations.target
    object_points = [
        np.ascontiguousarray(v.object_points(target), dtype=np.float64)
        for v in observations.views
    ]
    image_points = [
        np.ascontiguousarray(v.image_points, dtype=np.float64)
        for v in observations.views
    ]
    matrix, distortion = _initial_guess(observations, session, options)
    if model == "fisheye":
        return _run_fisheye(
            observations, object_points, image_points, options, matrix, distortion
        )
    camera, poses = _run_pinhole(
        observations, object_points, image_points, options, matrix, distortion
    )
    return camera, poses, "prior" if matrix is not None else "image size"


def _run_pinhole(
    observations, object_points, image_points, options, matrix, distortion
) -> Tuple[PinholeBrownConrady, List[Pose]]:
    flags = _pinhole_flags(options, matrix is not None)
    if flags & pinhole_flag("CALIB_USE_INTRINSIC_GUESS") and matrix is None:
        matrix = _default_guess(observations)
    terms = 5 if options.distortion_terms == 4 else options.distortion_terms
    guess = np.zeros(terms)
    if distortion is not None:
        keep = min(terms, distortion.size)
        guess[:keep] = distortion[:keep]
    try:
        # cv2.calibrateCamera takes float32 point arrays and rejects float64.
        # Residuals and the covariance are recomputed afterwards in float64
        # against the original detections, so the downcast affects only the
        # optimiser's own arithmetic.
        _, camera_matrix, coefficients, rvecs, tvecs = cv2.calibrateCamera(
            [p.reshape(-1, 1, 3).astype(np.float32) for p in object_points],
            [p.reshape(-1, 1, 2).astype(np.float32) for p in image_points],
            observations.image_size,
            None if matrix is None else matrix.copy(),
            guess,
            flags=flags,
            criteria=CRITERIA,
        )
    except cv2.error as exc:
        raise RefitError(f"cv2.calibrateCamera failed: {exc}") from exc
    camera = PinholeBrownConrady(
        camera_matrix[0, 0], camera_matrix[1, 1],
        camera_matrix[0, 2], camera_matrix[1, 2],
        np.asarray(coefficients, dtype=float).reshape(-1)[:terms],
    )
    poses = [
        Pose.from_rvec_tvec(np.asarray(r).reshape(3), np.asarray(t).reshape(3))
        for r, t in zip(rvecs, tvecs)
    ]
    return camera, poses


def _run_fisheye(
    observations, object_points, image_points, options, matrix, distortion
) -> Tuple[FisheyeKannalaBrandt, List[Pose], str]:
    # The guess always comes from the ladder, so USE_INTRINSIC_GUESS is always
    # set and OpenCV's own fragile initialiser never runs.
    flags = _fisheye_flags(options, has_guess=True)
    objects = [p.reshape(1, -1, 3) for p in object_points]
    images = [p.reshape(1, -1, 2) for p in image_points]
    # Starting point for the *free* coefficients. OpenCV's fisheye path zeroes
    # any coefficient covered by a CALIB_FIX_Kn flag and ignores the value passed
    # in for it, which is the opposite of what cv2.calibrateCamera does; see the
    # note on RefitOptions.fixed.
    coefficient_guess = np.zeros((4, 1))
    if distortion is not None:
        keep = min(4, distortion.size)
        coefficient_guess[:keep, 0] = distortion[:keep]
    attempts: List[str] = []
    for label, guess in _fisheye_guess_ladder(observations, matrix):
        try:
            _, camera_matrix, coefficients, rvecs, tvecs = cv2.fisheye.calibrate(
                objects, images, observations.image_size,
                guess.copy(), coefficient_guess.copy(),
                flags=flags, criteria=CRITERIA,
            )
        except cv2.error as exc:
            attempts.append(f"{label} ({str(exc).strip().splitlines()[-1][:70]})")
            continue
        camera = FisheyeKannalaBrandt(
            camera_matrix[0, 0], camera_matrix[1, 1],
            camera_matrix[0, 2], camera_matrix[1, 2],
            np.asarray(coefficients, dtype=float).reshape(-1)[:4],
            alpha=0.0,
        )
        poses = [
            Pose.from_rvec_tvec(np.asarray(r).reshape(3), np.asarray(t).reshape(3))
            for r, t in zip(rvecs, tvecs)
        ]
        return camera, poses, label
    raise RefitError(
        "cv2.fisheye.calibrate failed from every starting point tried "
        f"({len(attempts)}): {'; '.join(attempts)}. Fisheye calibration is "
        "sensitive to views where the board is small or near the frame edge; "
        "pass an existing calibration with --calibration to start from it, or "
        "--check-condition to make OpenCV name the view it rejects"
    )


def _solve_poses(camera: CameraModel, observations: ObservationSet) -> List[Pose]:
    projector = projector_for(camera)
    poses = []
    for view in observations.views:
        try:
            poses.append(
                projector.solve_pose(
                    camera,
                    view.object_points(observations.target),
                    view.image_points,
                )
            )
        except (RefitError, cv2.error) as exc:
            raise RefitError(
                f"view {view.view_id!r}: could not solve a pose at the given "
                f"intrinsics: {exc}"
            ) from exc
    return poses
