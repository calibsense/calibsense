"""M3 — out-of-sample error by cross-validation over views.

Reprojection RMS is a training error. It measures how well the estimated model
explains the very images used to estimate it, and with enough free parameters
and a poorly conditioned pose set it can be driven arbitrarily low while the
model itself is metrically wrong. Nobody reports the out-of-sample number, it
costs a handful of extra calibrations, and it is a far better estimator.

The held-out evaluation solves each held-out view's pose by PnP at the fold's
intrinsics, then measures reprojection. That is deliberate and it is the
deployment-relevant question: in use you have intrinsics, you observe a target,
and you solve a pose. Using the pose from the full fit instead would leak the
held-out view into its own prediction.

One consequence worth knowing: because six pose parameters are fitted to the
held-out view's own points, the held-out RMS is not a pure prediction error. It
is the best error achievable on a new view given these intrinsics, which is
exactly the quantity a measurement task cares about.

**The ratio has a blind spot, and it is a large one.** Any degeneracy a held-out
view's own free pose can absorb is invisible to reprojection-based
cross-validation. The focal-length/depth ambiguity is exactly such a degeneracy:
on a frontoparallel capture where the estimated focal length is wrong by
hundreds of pixels, the held-out view simply solves a proportionally wrong depth
and reprojects perfectly. Measured on a rig whose `fx` was wrong by 3333 px, the
ratio came out at 1.005.

Two things are therefore reported alongside it. Per-fold identifiability, which
gates the verdict — a ratio pooled from folds that did not determine their own
intrinsics is a lower bound, not a result. And the spread of each parameter
*across* folds, which on that same rig ranged from 782 to 1495 px and is the
signal the ratio cannot see. That spread is model-free: it resamples the actual
views and assumes nothing about the noise. It is not an unbiased estimate of the
sampling standard deviation, because folds share most of their training data, so
read it as a relative indicator rather than a substitute for the covariance.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..core.observations import ObservationSet
from ..core.session import CalibrationSession
from ..errors import RefitError, ValidationError
from ..refit.engine import instrument
from ..refit.projection import MIN_POINTS_FOR_POSE, projector_for
from ..refit.result import RefitOptions
from .result import CrossValidation, Fold, HeldOutView

#: Fewest views a fold may train on. Three is the algebraic floor for a planar
#: target; anything at the floor produces a fit that is technically a fit.
MIN_TRAIN_VIEWS = 4

#: Folds used when the caller does not say.
DEFAULT_FOLDS = 5


def fold_assignments(
    n_views: int, n_folds: int, shuffle: bool = True, seed: Optional[int] = 0
) -> List[np.ndarray]:
    """Split view indices into folds.

    Views are shuffled before splitting by default. Capture order is usually
    correlated — consecutive frames look alike — so contiguous blocks of the
    original order make the held-out set unrepresentatively hard, while
    shuffling makes each fold a fair sample of the capture.

    Args:
        n_views: Number of views to split.
        n_folds: Number of folds.
        shuffle: Shuffle before splitting.
        seed: Seed for the shuffle, so a report is reproducible.

    Returns:
        One array of view indices per fold, sizes differing by at most one.

    Raises:
        ValidationError: The fold count is not usable for this many views.
    """
    if n_folds < 2:
        raise ValidationError(f"cross-validation needs at least 2 folds, got {n_folds}")
    if n_folds > n_views:
        raise ValidationError(
            f"cannot make {n_folds} folds from {n_views} views; "
            f"pass folds<={n_views} or capture more views"
        )
    order = np.arange(n_views)
    if shuffle:
        np.random.default_rng(seed).shuffle(order)
    return [np.sort(part) for part in np.array_split(order, n_folds)]


def choose_folds(n_views: int, requested: Optional[int] = None) -> int:
    """Pick a fold count that leaves enough views to fit on.

    Args:
        n_views: Number of views available.
        requested: The caller's preference, or `None` for the default.

    Returns:
        A fold count between 2 and `n_views` that leaves at least
        `MIN_TRAIN_VIEWS` views in every training set.

    Raises:
        ValidationError: There are too few views to hold any back.
    """
    if n_views <= MIN_TRAIN_VIEWS:
        raise ValidationError(
            f"cross-validation needs more than {MIN_TRAIN_VIEWS} views to hold any "
            f"back, got {n_views}"
        )
    n_folds = requested if requested is not None else DEFAULT_FOLDS
    n_folds = int(np.clip(n_folds, 2, n_views))
    # Each fold trains on n_views - ceil(n_views / n_folds) views at worst.
    while n_folds < n_views:
        largest_test = int(np.ceil(n_views / n_folds))
        if n_views - largest_test >= MIN_TRAIN_VIEWS:
            break
        n_folds += 1
    largest_test = int(np.ceil(n_views / n_folds))
    if n_views - largest_test < MIN_TRAIN_VIEWS:
        raise ValidationError(
            f"{n_views} views cannot leave {MIN_TRAIN_VIEWS} in a training set "
            "under any fold count"
        )
    return n_folds


def evaluate_held_out(
    camera, observations: ObservationSet, indices: Sequence[int], fold: int
) -> Tuple[List[HeldOutView], float, int]:
    """Reproject held-out views at fixed intrinsics, solving each pose by PnP.

    Args:
        camera: The intrinsics estimated without these views.
        observations: The full observation set.
        indices: Positions of the held-out views.
        fold: Fold number, recorded on each result.

    Returns:
        The per-view results, the total squared residual, and the point count.

    Raises:
        RefitError: A held-out view's pose could not be solved.
    """
    projector = projector_for(camera)
    target = observations.target
    results: List[HeldOutView] = []
    squared_total = 0.0
    points_total = 0
    for index in indices:
        view = observations.views[index]
        object_points = view.object_points(target)
        if view.n_points < MIN_POINTS_FOR_POSE:
            raise RefitError(
                f"held-out view {view.view_id!r} has {view.n_points} points, "
                f"below the {MIN_POINTS_FOR_POSE} a pose needs"
            )
        pose = projector.solve_pose(camera, object_points, view.image_points)
        residual = projector.project(camera, pose, object_points) - view.image_points
        squared = float(np.sum(residual ** 2))
        results.append(
            HeldOutView(
                view_id=view.view_id,
                n_points=view.n_points,
                rms=float(np.sqrt(squared / view.n_points)),
                distance_mm=float(np.linalg.norm(pose.translation)),
                fold=fold,
            )
        )
        squared_total += squared
        points_total += view.n_points
    return results, squared_total, points_total


def cross_validate(
    session: CalibrationSession,
    options: Optional[RefitOptions] = None,
    folds: Optional[int] = None,
    shuffle: bool = True,
    seed: Optional[int] = 0,
) -> CrossValidation:
    """Measure out-of-sample reprojection error by K-fold over views.

    Args:
        session: The ingested session.
        options: Refit settings, applied identically to every fold. `refit` is
            forced on, because a fold has to estimate its own intrinsics.
        folds: Number of folds, or `None` for a default that leaves enough views
            in each training set.
        shuffle: Shuffle views before splitting.
        seed: Seed for the shuffle.

    Returns:
        In-sample RMS, out-of-sample RMS, their ratio, and the per-fold detail.

    Raises:
        ValidationError: There are too few views to hold any back.
        RefitError: A fold could not be fitted, or a held-out pose could not be
            solved.
    """
    observations = session.observations
    base = options or RefitOptions()
    if not base.refit:
        # A fold that reuses the session's own calibration is not a fold; it has
        # to estimate its own intrinsics from its own training views.
        base = RefitOptions.from_dict({**base.to_dict(), "refit": True})

    n_folds = choose_folds(observations.n_views, folds)
    full = instrument(session, base)
    assignments = fold_assignments(observations.n_views, n_folds, shuffle, seed)

    fold_results: List[Fold] = []
    held_out: List[HeldOutView] = []
    degenerate: List[int] = []
    squared_total = 0.0
    points_total = 0
    parameter_samples: List[np.ndarray] = []

    for index, test_indices in enumerate(assignments):
        train_indices = np.setdiff1d(np.arange(observations.n_views), test_indices)
        if train_indices.size < MIN_TRAIN_VIEWS:
            raise ValidationError(
                f"fold {index} would train on {train_indices.size} views, "
                f"below the minimum of {MIN_TRAIN_VIEWS}"
            )
        train_session = _subset(session, train_indices)
        fold_fit = instrument(train_session, base)
        if not fold_fit.conditioning.identifiable:
            degenerate.append(index)

        results, squared, points = evaluate_held_out(
            fold_fit.camera, observations, test_indices, index
        )
        held_out.extend(results)
        squared_total += squared
        points_total += points
        parameter_samples.append(fold_fit.camera.to_vector())

        fold_results.append(
            Fold(
                index=index,
                train_view_ids=tuple(observations.views[i].view_id for i in train_indices),
                held_out_view_ids=tuple(
                    observations.views[i].view_id for i in test_indices
                ),
                camera=fold_fit.camera,
                train_rms=fold_fit.rms,
                held_out_rms=float(np.sqrt(squared / points)) if points else 0.0,
                held_out_points=points,
                identifiable=fold_fit.conditioning.identifiable,
            )
        )

    # Pool by point count. Averaging per-fold RMS values would weight a fold
    # holding back one thin view the same as one holding back four full ones.
    out_of_sample = float(np.sqrt(squared_total / points_total)) if points_total else 0.0
    samples = np.stack(parameter_samples)
    names = full.camera.parameter_names()
    predicted = np.full(len(names), np.nan)
    estimated = dict(
        zip(full.covariance.intrinsic_names, full.covariance.intrinsic_std())
    )
    for position, name in enumerate(names):
        if name in estimated:
            predicted[position] = estimated[name]
    return CrossValidation(
        folds=tuple(fold_results),
        held_out=tuple(held_out),
        in_sample_rms=full.rms,
        out_of_sample_rms=out_of_sample,
        n_views=observations.n_views,
        fold_spread=(
            samples.std(axis=0, ddof=1) if len(samples) > 1
            else np.zeros(samples.shape[1])
        ),
        predicted_spread=predicted,
        parameter_names=names,
        degenerate_folds=tuple(degenerate),
    )


def _subset(session: CalibrationSession, indices: np.ndarray) -> CalibrationSession:
    """A session holding only the given views, everything else carried across."""
    return CalibrationSession(
        observations=session.observations.select(indices.tolist()),
        prior=session.prior,
        robot=None,
        created=session.created,
        caltrust_version=session.caltrust_version,
        metadata=session.metadata,
    )
