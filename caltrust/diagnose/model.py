"""Is the distortion model the right shape for this lens?

The distinction this diagnostic exists to make is between residuals that are
noise and residuals that are structure. An RMS cannot tell them apart: 0.25 px
of independent noise and 0.25 px of systematic radial error look identical in
that one number, and only one of them means the model is wrong.

Two tests run. A chi-square over the radial bands asks whether the profile is
consistent with zero *at all*, which catches structure of any shape; a truncated
radial polynomial leaves a residual that oscillates in sign, and a slope test
alone would walk straight past it. A weighted regression then describes the
shape, because a monotone trend and an oscillation call for different fixes.

Getting the standard errors right matters more than either test does,
and the naive choice is wrong: corners within one view share that view's pose,
so a small pose error moves all of them together and they are not independent
samples. Treating them as independent inflates every z-score, and on a capture
with a few hundred corners per bin it will report structure that is not there.

So the independent unit here is the **view**, not the corner. Each view
contributes one mean per radial bin, and the bin's standard error comes from the
scatter of those view means. That is a cluster-robust estimate: it needs no
assumption about the noise being independent, isotropic or Gaussian, only that
different views are independent of each other.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, List, Optional, Tuple

import numpy as np

from .base import Diagnostic, DiagnosticContext, Finding, Severity

#: Omnibus z-score on the whole radial profile above which the model is wrong.
#: This is the primary criterion; the slope only describes the shape.
FLATNESS_CRITICAL_Z = 5.0
FLATNESS_WARNING_Z = 3.0

#: Absolute t-statistic on the radial slope above which the model is wrong.
SLOPE_CRITICAL_T = 5.0
SLOPE_WARNING_T = 3.0

#: Absolute z-score of a single bin above which that bin is implausible.
BIN_CRITICAL_Z = 5.0
BIN_WARNING_Z = 3.5

#: A view must put at least this many corners in a bin to contribute a mean.
MIN_POINTS_PER_VIEW_BIN = 4

#: A bin needs means from at least this many views to have a standard error.
MIN_VIEWS_PER_BIN = 3

#: Radial bins used for the test, independent of the display profile.
N_BINS = 8

#: Points closer to the principal point than this many residual standard
#: deviations are dropped. The outward direction there is set by the point's own
#: measured position, so at small radius it is dominated by that point's noise
#: and correlates with the residual being projected onto it, which biases the
#: radial mean upwards.
MIN_RADIUS_SIGMAS = 30.0


@dataclass(frozen=True)
class ClusteredProfile:
    """Radial residual statistics with views as the independent unit.

    Attributes:
        centres: Bin centre radii in pixels, only for usable bins.
        means: Mean radial residual per bin, averaged over view means.
        errors: Cluster-robust standard error of each bin mean.
        view_counts: Views contributing a mean to each bin.
    """

    centres: np.ndarray
    means: np.ndarray
    errors: np.ndarray
    view_counts: np.ndarray

    @property
    def n_bins(self) -> int:
        """Number of usable bins."""
        return int(self.centres.size)

    def z_scores(self) -> np.ndarray:
        """How many standard errors each bin mean sits from zero."""
        safe = np.where(self.errors > 0, self.errors, np.inf)
        return self.means / safe


def clustered_radial_profile(
    context: DiagnosticContext, n_bins: int = N_BINS
) -> ClusteredProfile:
    """Bin radial residuals by image radius, clustering by view.

    Args:
        context: The fit and its detections.
        n_bins: Number of equal-width radial bins.

    Returns:
        Only the bins with enough views to carry a standard error.
    """
    camera = context.fit.camera
    centre = np.array([camera.cx, camera.cy])
    statistics = context.fit.residuals

    radii_all = np.linalg.norm(context.all_image_points - centre, axis=1)
    floor = MIN_RADIUS_SIGMAS * max(context.fit.covariance.sigma, 1e-6)
    edges = np.linspace(floor, max(float(radii_all.max()), floor + 1e-9), n_bins + 1)

    per_bin: List[List[float]] = [[] for _ in range(n_bins)]
    for index, view in enumerate(context.observations.views):
        residual = statistics.view_residuals(index)
        offset = view.image_points - centre
        radius = np.linalg.norm(offset, axis=1)
        outward = offset / np.where(radius[:, None] > 0, radius[:, None], 1.0)
        radial = np.sum(residual * outward, axis=1)
        usable = radius >= floor
        radial, radius = radial[usable], radius[usable]
        if radial.size == 0:
            continue
        assignment = np.clip(np.digitize(radius, edges[1:-1]), 0, n_bins - 1)
        for bin_index in range(n_bins):
            selected = radial[assignment == bin_index]
            if selected.size >= MIN_POINTS_PER_VIEW_BIN:
                per_bin[bin_index].append(float(selected.mean()))

    centres, means, errors, counts = [], [], [], []
    for bin_index, view_means in enumerate(per_bin):
        if len(view_means) < MIN_VIEWS_PER_BIN:
            continue
        values = np.asarray(view_means)
        centres.append(0.5 * (edges[bin_index] + edges[bin_index + 1]))
        means.append(float(values.mean()))
        errors.append(float(values.std(ddof=1) / np.sqrt(values.size)))
        counts.append(values.size)
    return ClusteredProfile(
        centres=np.asarray(centres),
        means=np.asarray(means),
        errors=np.asarray(errors),
        view_counts=np.asarray(counts, dtype=int),
    )


def flatness_z(profile: ClusteredProfile) -> float:
    """How implausible a flat radial profile is, as a z-score.

    An omnibus chi-square over the bands, converted to a standard normal by the
    Wilson-Hilferty transform so that it is comparable with the other thresholds
    in this package and needs no special-function library.

    Args:
        profile: The clustered radial profile.

    Returns:
        A z-score. Zero means the profile is exactly as flat as noise predicts;
        negative means flatter; large positive means there is structure.
    """
    bands = profile.n_bins
    if bands < 1:
        return 0.0
    chi_square = float(np.sum(profile.z_scores() ** 2))
    reduced = chi_square / bands
    shift = 1.0 - 2.0 / (9.0 * bands)
    scale = np.sqrt(2.0 / (9.0 * bands))
    return float((np.cbrt(reduced) - shift) / scale)


def radial_trend(profile: ClusteredProfile) -> Optional[Tuple[float, float, float]]:
    """Fit the mean radial residual against radius, weighted by its own error.

    Args:
        profile: The clustered radial profile.

    Returns:
        A triple `(slope_px_per_100px, standard_error, t_statistic)`, or `None`
        when fewer than three bins are usable.
    """
    if profile.n_bins < 3 or not np.all(profile.errors > 0):
        return None
    weights = 1.0 / profile.errors ** 2
    design = np.column_stack([np.ones(profile.n_bins), profile.centres])
    weighted = design * weights[:, None]
    try:
        covariance = np.linalg.inv(design.T @ weighted)
    except np.linalg.LinAlgError:
        return None
    coefficients = covariance @ (weighted.T @ profile.means)
    slope = float(coefficients[1])
    error = float(np.sqrt(max(covariance[1, 1], 0.0)))
    statistic = slope / error if error > 0 else 0.0
    # Reported per 100 px of radius, a readable magnitude for a lens.
    return slope * 100.0, error * 100.0, float(statistic)


class DistortionModelAdequacy(Diagnostic):
    """Does the residual field carry radial structure the model failed to absorb?

    A systematic radial trend means the model is wrong, not that the data is
    noisy. The usual causes are a Brown-Conrady model fitted to a lens wide
    enough to need Kannala-Brandt, and too few radial terms for a strongly
    distorting lens.
    """

    cause: ClassVar[str] = "distortion_model"
    title: ClassVar[str] = "Distortion model adequacy"

    def run(self, context: DiagnosticContext) -> Finding:
        """Regress the cluster-robust mean radial residual on radius."""
        profile = clustered_radial_profile(context)
        terms = context.fit.camera.distortion.size
        model = context.fit.camera.kind
        metrics = dict(
            usable_bins=profile.n_bins,
            distortion_terms=terms,
            model=model,
            min_views_per_bin=(
                int(profile.view_counts.min()) if profile.n_bins else 0
            ),
        )
        if profile.n_bins < 3:
            return self._finding(
                Severity.NOTE,
                f"only {profile.n_bins} radial band(s) are seen by at least "
                f"{MIN_VIEWS_PER_BIN} views, so the residuals cannot say whether "
                "the distortion model fits",
                "Spread the detections over more of the image radius, and across "
                "more views at each radius. The test needs several views per band "
                "so that a single view's pose error is not mistaken for structure.",
                **metrics,
            )
        scores = profile.z_scores()
        worst_index = int(np.argmax(np.abs(scores)))
        worst_z = float(scores[worst_index])
        flatness = flatness_z(profile)
        trend = radial_trend(profile)
        metrics.update(
            worst_bin_z=worst_z,
            worst_bin_radius_px=float(profile.centres[worst_index]),
            flatness_z=flatness,
            chi_square=float(np.sum(scores ** 2)),
            bin_z_scores=[float(z) for z in scores],
            bin_radii_px=[float(r) for r in profile.centres],
        )
        if trend is None:
            return self._finding(
                Severity.NOTE,
                f"the worst radial band sits {worst_z:+.1f} standard errors from "
                "zero, but the trend across bands could not be fitted",
                "",
                **metrics,
            )
        slope, error, statistic = trend
        metrics.update(
            radial_slope_px_per_100px=slope,
            radial_slope_stderr=error,
            radial_slope_t=statistic,
        )
        shared = (
            f"the radial residual profile departs from flat at z = {flatness:+.1f} "
            f"across {profile.n_bins} bands, trending {slope:+.4f} px per 100 px of "
            f"radius (t = {statistic:+.1f}), with the worst band {worst_z:+.1f} "
            "standard errors from zero; views are the independent unit throughout"
        )
        action = (
            "This is structure, not noise. "
            + (
                f"Try the fisheye Kannala-Brandt model, or raise --distortion-terms "
                f"above {terms}. "
                if model.startswith("pinhole")
                else "Check whether the lens really is equidistant. "
            )
            + "A model that cannot represent the lens will not improve with more views."
        )
        if (
            flatness > FLATNESS_CRITICAL_Z
            or abs(statistic) > SLOPE_CRITICAL_T
            or abs(worst_z) > BIN_CRITICAL_Z
        ):
            return self._finding(Severity.CRITICAL, shared, action, **metrics)
        if (
            flatness > FLATNESS_WARNING_Z
            or abs(statistic) > SLOPE_WARNING_T
            or abs(worst_z) > BIN_WARNING_Z
        ):
            return self._finding(Severity.WARNING, shared, action, **metrics)
        return self._finding(
            Severity.OK,
            shared + ", both consistent with independent noise",
            **metrics,
        )
