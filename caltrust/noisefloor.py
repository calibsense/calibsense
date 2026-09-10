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

"""Measuring corner noise directly, from a capture where nothing moved.

Everywhere else in caltrust the noise scale is inferred from residuals, which
means it is entangled with the model: a fit that absorbs part of the noise
reports a smaller sigma than the sensor actually has. This module measures the
noise instead. Bolt the camera and the target down, take a hundred frames,
change nothing, and every corner is then measured a hundred times at the same
physical position. The scatter *is* the noise, with no model in the way.

That matters because the covariance in `caltrust.refit` estimates the noise
*scale* from the data but asserts its *shape* — one variance per coordinate,
equal in x and y, uncorrelated between points. Measured against known truth,
most departures from that shape turn out to be harmless; anisotropy and
heteroscedasticity are averaged away by a few hundred corners at varied
orientations. **Spatial correlation of the noise across the frame is not
harmless.** It breaks the covariance by around a factor of eight, and it breaks
it in the worst direction: a correlated field is partly absorbable by the pose
and distortion parameters, so it lowers the reported residual while raising the
estimator's true spread. A better RMS and an interval far too tight.

So this module reports three things, in order of how much they matter:

* the **correlation length** of the noise field, which is the number that decides
  whether the classical covariance can be believed at all,
* the **scale**, whose irreducible figure is the one to compare against
  `fit.covariance.sigma`, since a fit's sigma is also measured after the pose
  has taken what it can,
* the **shape** — anisotropy, and whether the long axis lies along the board's
  own edge direction. Anisotropy is quoted against `anisotropy_floor`, because
  a covariance from few frames is elongated by chance alone and ten frames put
  that floor at 1.53; the raw ratio on its own says nothing.

Two quantities are reported for the scale, and the difference between them is
the point. *Total* scatter is everything the detector produced. *Irreducible*
scatter is what survives removing a per-frame affine transform, which stands in
for what a calibration's free per-view pose can absorb from a planar target. A
rig that is quietly drifting or vibrating shows a large total and a small
irreducible; a lens with field-varying defocus shows correlation that survives
the affine removal, and that is the case which damages the covariance.

The scene really does have to be static. If it moved, the scatter is motion
rather than noise and every number here is meaningless, so a capture whose
corners wander further than `MAX_STATIC_SCATTER_PX` is refused rather than
summarised.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

from .core.observations import ObservationSet
from .errors import ValidationError

#: Fewest frames that give a usable 2x2 covariance per corner. Ten is the floor;
#: the estimate is still poor there, which the report says out loud.
MIN_FRAMES = 10

#: Frames below which the covariance estimate is too noisy to characterise a
#: shape, as opposed to merely a scale.
ADVISED_FRAMES = 30

#: Corner scatter above this is motion, not noise, and the capture is refused.
MAX_STATIC_SCATTER_PX = 5.0

#: A corner must appear in at least this fraction of frames to be used.
MIN_PRESENCE = 0.9

#: Separation bins for the spatial correlation profile.
CORRELATION_BINS = 12

#: Correlation at which the length scale is read off, the usual 1/e convention.
CORRELATION_THRESHOLD = float(np.exp(-1.0))

#: Correlation length, in units of corner spacing, above which the classical
#: covariance should not be trusted. Measured in pixels this threshold cannot be
#: fixed: injecting one field into a 9x6 board at 700 mm gave a 51 px length and
#: into a 15x10 board at 600 mm gave 87 px, for the same fault. What decides
#: whether the independence assumption fails is whether *neighbouring corners*
#: share noise, so one corner spacing is the natural unit and the threshold
#: transfers between rigs.
CORRELATION_SPACINGS_WARNING = 1.0

#: Where the failure stops being present and becomes severe. The Monte Carlo
#: that measured the eightfold understatement injected a field spanning most of
#: the board; at around one corner spacing it was closer to twofold. Above this,
#: the advice changes from "use the robust covariance" to "find the cause before
#: quoting any interval".
CORRELATION_SPACINGS_CRITICAL = 3.0

#: How far the measured anisotropy must exceed what the frame count alone
#: produces before it is reported as a real effect. Isotropic noise never
#: exceeded 1.29 over 2000 trials at the hardest case this tool accepts, ten
#: frames and twenty-four corners, so 1.3 clears estimator noise.
ANISOTROPY_EXCESS_WARNING = 1.3

#: Share of corners whose long axis falls within 30 degrees of the board edge
#: purely by chance, since the angle is folded into `[0, 90]`. The measured
#: fraction only means something against this.
EDGE_ALIGNED_BY_CHANCE = 30.0 / 90.0


def anisotropy_floor(n_frames: int, quantile: float = 0.5) -> float:
    """Anisotropy that perfectly isotropic noise produces at this frame count.

    A sample covariance from few frames is elongated by chance alone, so a
    measured ellipse has to be compared against this rather than against one.
    Ten frames put the median at 1.53, which is why a fixed threshold would
    call half of all clean short captures anisotropic.

    For two dimensions the law is exact and needs no special functions: with
    `t = (l1 - l2) / (l1 + l2)` over the sample eigenvalues, `t**2` follows
    `Beta(1, (n - 2) / 2)`, so every quantile inverts in closed form and the
    anisotropy is `sqrt((1 + t) / (1 - t))`. Verified against simulation to
    three decimals from ten frames to a thousand.

    Args:
        n_frames: Frames the corner was detected in.
        quantile: Which quantile of the floor to return. The default median is
            what a cross-corner median should be compared against; a high
            quantile is the right comparison for a single corner.

    Returns:
        The anisotropy ratio, at or above one. Infinite below four frames,
        where the ratio is unbounded.
    """
    if n_frames < 4:
        return float("inf")
    if not 0.0 < quantile < 1.0:
        raise ValidationError(f"quantile must lie in (0, 1), got {quantile}")
    t = np.sqrt(1.0 - (1.0 - quantile) ** (2.0 / (n_frames - 2)))
    return float(np.sqrt((1.0 + t) / (1.0 - t)))


@dataclass(frozen=True)
class CornerNoise:
    """Measured noise at one target point.

    Attributes:
        point_id: The target point this describes.
        n_frames: Frames the point was detected in.
        mean_px: Mean image position over those frames.
        covariance: The `(2, 2)` sample covariance of the raw position, in
            px^2. This is the total scatter, rig movement included.
        irreducible_covariance: The same after a per-frame affine is removed,
            so whole-board movement no longer contributes. Scale is read from
            `covariance` and shape from this one: a rig that drifts sideways
            makes the raw ellipse long in x for reasons that have nothing to do
            with the detector, and `NoiseFloor.drift_px` already reports that
            separately.
        neighbour_direction_deg: Direction to the nearest other corner, in
            degrees from the image x axis. The board's own edge direction runs
            along this, which is what makes the comparison below meaningful.
    """

    point_id: int
    n_frames: int
    mean_px: np.ndarray
    covariance: np.ndarray
    irreducible_covariance: np.ndarray
    neighbour_direction_deg: float

    @property
    def sigma_px(self) -> float:
        """Equivalent isotropic deviation, per coordinate.

        Half the trace, square-rooted. Compare a fit's own `sigma` against
        `irreducible_sigma_px` instead: a fit reports the scatter left after
        each view's pose has absorbed what it can, which is the same thing the
        affine removal here stands in for.
        """
        return float(np.sqrt(max(np.trace(self.covariance), 0.0) / 2.0))

    @property
    def irreducible_sigma_px(self) -> float:
        """Equivalent isotropic deviation once a per-frame affine is removed."""
        return float(
            np.sqrt(max(np.trace(self.irreducible_covariance), 0.0) / 2.0)
        )

    @property
    def axes_px(self) -> Tuple[float, float]:
        """Major and minor standard deviations of the scatter ellipse.

        Taken from `irreducible_covariance`, so the shape describes the
        detector rather than the rig.
        """
        values = np.linalg.eigvalsh(self.irreducible_covariance)
        return float(np.sqrt(max(values[1], 0.0))), float(np.sqrt(max(values[0], 0.0)))

    @property
    def anisotropy(self) -> float:
        """Major over minor deviation; one is a circle."""
        major, minor = self.axes_px
        return major / minor if minor > 0 else float("inf")

    @property
    def orientation_deg(self) -> float:
        """Direction of the long axis, in degrees from the image x axis."""
        values, vectors = np.linalg.eigh(self.irreducible_covariance)
        major = vectors[:, int(np.argmax(values))]
        return float(np.degrees(np.arctan2(major[1], major[0])) % 180.0)

    @property
    def axis_vs_neighbour_deg(self) -> float:
        """Angle between the long axis and the direction to the nearest corner.

        Folded into `[0, 90]`. Near zero means the scatter is elongated along
        the board's edge, which is the classic signature of a corner detector
        being better at locating an edge across than along it.
        """
        difference = abs(self.orientation_deg - self.neighbour_direction_deg) % 180.0
        return float(min(difference, 180.0 - difference))


@dataclass(frozen=True)
class CorrelationProfile:
    """How strongly the noise at two corners moves together, against separation.

    Attributes:
        separation_px: Bin centre separations.
        correlation: Mean vector correlation of the two corners' deviations.
        pair_counts: Corner pairs contributing to each bin.
        length_px: Separation at which the correlation falls to `1/e`, linearly
            interpolated. `0.0` when it is already below that in the first bin,
            which is what independent noise looks like.
    """

    separation_px: np.ndarray
    correlation: np.ndarray
    pair_counts: np.ndarray
    length_px: float

    @property
    def n_bins(self) -> int:
        """Number of populated bins."""
        return int(self.separation_px.size)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain JSON-compatible dictionary."""
        return {
            "separation_px": self.separation_px.tolist(),
            "correlation": self.correlation.tolist(),
            "pair_counts": self.pair_counts.tolist(),
            "length_px": self.length_px,
        }


@dataclass(frozen=True)
class NoiseFloor:
    """Directly measured corner noise for one static capture.

    Attributes:
        corners: Per-corner measurements, in point-id order.
        n_frames: Frames used.
        n_detected: Distinct target points seen in at least one frame. Larger
            than `n_corners` when points were dropped for appearing too rarely,
            which is worth reporting rather than discarding quietly.
        image_size: Frame size as `(width, height)`.
        total_sigma_px: Pooled per-coordinate deviation of the raw positions.
        irreducible_sigma_px: The same after removing a per-frame affine
            transform, which stands in for what a free per-view pose absorbs.
        drift_px: Largest per-frame mean displacement, which is rig movement
            rather than detector noise.
        spacing_px: Median distance from a corner to its nearest neighbour. The
            unit the correlation length is judged in, since what matters is
            whether adjacent corners share noise rather than any absolute
            distance.
        correlation: The spatial correlation profile of the irreducible part.
        total_correlation: The same profile before the affine removal, kept so
            that rig motion and a genuinely correlated field can be told apart.
    """

    corners: Tuple[CornerNoise, ...]
    n_frames: int
    n_detected: int
    image_size: Tuple[int, int]
    total_sigma_px: float
    irreducible_sigma_px: float
    drift_px: float
    spacing_px: float
    correlation: CorrelationProfile
    total_correlation: CorrelationProfile

    @property
    def n_corners(self) -> int:
        """Corners measured."""
        return len(self.corners)

    @property
    def anisotropy_median(self) -> float:
        """Median major-over-minor deviation ratio across corners."""
        return float(np.median([c.anisotropy for c in self.corners]))

    @property
    def anisotropy_floor(self) -> float:
        """Anisotropy this frame count produces from isotropic noise alone."""
        return anisotropy_floor(self.n_frames)

    @property
    def anisotropy_excess(self) -> float:
        """Measured anisotropy over the floor; one means nothing to report.

        The ratio, not the raw anisotropy, is what says whether the detector is
        actually directional, because the raw figure is inflated by short
        captures no matter how clean the noise is.
        """
        floor = self.anisotropy_floor
        return self.anisotropy_median / floor if np.isfinite(floor) else 1.0

    @property
    def edge_aligned_fraction(self) -> float:
        """Fraction of corners whose long axis lies within 30 degrees of the edge.

        Compare against `EDGE_ALIGNED_BY_CHANCE`: the angle is folded into
        `[0, 90]`, so a third of corners land inside 30 degrees for no reason at
        all. Well above that means the detector really is oriented to the
        board, which is the expected behaviour of anything that finds a corner
        by locating two edges.
        """
        if not self.corners:
            return 0.0
        angles = np.array([c.axis_vs_neighbour_deg for c in self.corners])
        return float(np.mean(angles < 30.0))

    @property
    def absorbed_fraction(self) -> float:
        """Share of the total variance a per-view pose could absorb.

        A high value means most of the scatter is whole-frame movement — a
        drifting mount, vibration, a slowly warming lens — which a calibration
        will quietly soak up into its poses.
        """
        total = self.total_sigma_px ** 2
        if total <= 0:
            return 0.0
        return float(np.clip(1.0 - (self.irreducible_sigma_px ** 2) / total, 0.0, 1.0))

    @property
    def correlation_spacings(self) -> float:
        """Correlation length in units of corner spacing.

        This, not the pixel figure, is what the thresholds judge. Zero means
        each corner's noise is its own.
        """
        if self.spacing_px <= 0:
            return 0.0
        return self.correlation.length_px / self.spacing_px

    @property
    def trustworthy_covariance(self) -> bool:
        """Whether the classical covariance's independence assumption holds here.

        `False` means the noise field is correlated over distances comparable to
        the corner spacing, which is the regime where `sigma^2 (J'J)^-1` was
        measured to understate the parameter spread by around eightfold. Read
        `RobustCovariance` instead, and expect the `noise_model` finding to say
        the same thing from the other direction.
        """
        return self.correlation_spacings < CORRELATION_SPACINGS_WARNING

    def sigma_for_propagation(self) -> float:
        """The deviation to hand to `caltrust.task.propagate`.

        The irreducible part, because that is the noise a measurement actually
        suffers once the pose has taken what it can.
        """
        return self.irreducible_sigma_px

    def summary_lines(self) -> Tuple[str, ...]:
        """A human summary, leading with the number that decides the rest."""
        length = self.correlation.length_px
        lines = [
            f"frames       {self.n_frames} static frames, {self.n_corners} corners"
            + ("" if self.n_corners == self.n_detected else
               f" of {self.n_detected} detected ({self.n_detected - self.n_corners} "
               "seen too rarely to measure)"),
            f"scale        {self.total_sigma_px:.4f} px total, "
            f"{self.irreducible_sigma_px:.4f} px irreducible "
            f"({self.absorbed_fraction:.0%} absorbable by pose)",
            f"correlation  length {length:.1f} px, "
            f"{self.correlation_spacings:.2f} of the {self.spacing_px:.0f} px corner "
            f"spacing ({'independent' if self.trustworthy_covariance else 'CORRELATED'})",
            f"shape        anisotropy {self.anisotropy_median:.2f} against a "
            f"{self.anisotropy_floor:.2f} floor from {self.n_frames} frames, "
            f"{self.edge_aligned_fraction:.0%} of long axes within 30 deg of the "
            f"board edge ({EDGE_ALIGNED_BY_CHANCE:.0%} by chance)",
            f"drift        {self.drift_px:.4f} px largest whole-frame shift",
            f"use          sigma = {self.sigma_for_propagation():.4f} px when "
            "propagating, in place of one inferred from residuals"
            + ("" if self.trustworthy_covariance else
               " -- but read the COVARIANCE line first: a correct scale does not "
               "rescue a covariance whose independence assumption has failed"),
        ]
        if self.n_frames < ADVISED_FRAMES:
            lines.append(
                f"CAUTION      {self.n_frames} frames is enough for a scale but thin "
                f"for a shape; {ADVISED_FRAMES} or more makes the anisotropy and "
                "correlation figures worth quoting"
            )
        if self.correlation_spacings >= CORRELATION_SPACINGS_CRITICAL:
            lines.append(
                f"COVARIANCE   the noise is correlated over {length:.0f} px, which "
                f"spans {self.correlation_spacings:.1f} corner spacings. This is "
                "the regime where the classical covariance was measured to "
                "understate the parameter spread around "
                "eightfold while lowering the RMS, so no interval from this capture "
                "is quotable until the cause is found. Look for field-varying "
                "defocus, an illumination or vignetting gradient, a board that is "
                "not flat, motion blur or a rolling shutter"
            )
        elif not self.trustworthy_covariance:
            lines.append(
                f"COVARIANCE   the noise is correlated over {length:.0f} px, "
                f"{self.correlation_spacings:.1f} corner spacings, so adjacent "
                "corners share it. The classical covariance "
                "understates the parameter spread in this regime; use "
                "RobustCovariance and read the noise_model finding"
            )
        if self.absorbed_fraction > 0.5:
            lines.append(
                f"RIG          {self.absorbed_fraction:.0%} of the scatter is "
                "whole-frame movement, so the mount or the scene was not as static "
                "as intended; the irreducible figure is the one to use"
            )
        if self.anisotropy_excess > ANISOTROPY_EXCESS_WARNING:
            lines.append(
                f"SHAPE        corners are {self.anisotropy_median:.2f}x more "
                "uncertain along one axis than the other, against "
                f"{self.anisotropy_floor:.2f}x from {self.n_frames} frames of "
                "isotropic noise alone; harmless for the covariance by "
                "measurement, but it tells you the detector is working harder "
                "in one direction"
            )
        return tuple(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain JSON-compatible dictionary."""
        return {
            "n_frames": self.n_frames,
            "n_corners": self.n_corners,
            "n_detected": self.n_detected,
            "image_size": list(self.image_size),
            "total_sigma_px": self.total_sigma_px,
            "irreducible_sigma_px": self.irreducible_sigma_px,
            "sigma_for_propagation": self.sigma_for_propagation(),
            "absorbed_fraction": self.absorbed_fraction,
            "drift_px": self.drift_px,
            "spacing_px": self.spacing_px,
            "correlation_spacings": self.correlation_spacings,
            "anisotropy_median": self.anisotropy_median,
            "anisotropy_floor": self.anisotropy_floor,
            "anisotropy_excess": self.anisotropy_excess,
            "edge_aligned_fraction": self.edge_aligned_fraction,
            "trustworthy_covariance": self.trustworthy_covariance,
            "correlation": self.correlation.to_dict(),
            "total_correlation": self.total_correlation.to_dict(),
            "corners": [
                {
                    "point_id": c.point_id,
                    "n_frames": c.n_frames,
                    "mean_px": c.mean_px.tolist(),
                    "sigma_px": c.sigma_px,
                    "irreducible_sigma_px": c.irreducible_sigma_px,
                    "major_px": c.axes_px[0],
                    "minor_px": c.axes_px[1],
                    "anisotropy": c.anisotropy,
                    "orientation_deg": c.orientation_deg,
                    "axis_vs_neighbour_deg": c.axis_vs_neighbour_deg,
                }
                for c in self.corners
            ],
        }


def _stack_static(
    observations: ObservationSet, min_presence: float = MIN_PRESENCE
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Gather the frames into one array indexed by corner.

    Args:
        observations: Frames of a stationary scene, one view per frame.
        min_presence: Fraction of frames a corner must appear in to be kept.

    Returns:
        The point ids kept, an `(n_frames, n_corners, 2)` array of positions,
        and how many distinct points were detected at all.

    Raises:
        ValidationError: Too few frames, no corner seen often enough, or the
            scatter is large enough that the scene must have moved.
    """
    frames = observations.views
    if len(frames) < MIN_FRAMES:
        raise ValidationError(
            f"a noise floor needs at least {MIN_FRAMES} frames of a static scene, "
            f"got {len(frames)}"
        )
    counts: Dict[int, int] = {}
    for view in frames:
        for point_id in view.point_ids.tolist():
            counts[point_id] = counts.get(point_id, 0) + 1
    threshold = min_presence * len(frames)
    kept = sorted(pid for pid, seen in counts.items() if seen >= threshold)
    if not kept:
        raise ValidationError(
            f"no corner was detected in at least {min_presence:.0%} of the "
            f"{len(frames)} frames; the scene is not static, or detection is "
            "failing intermittently"
        )

    index = {pid: position for position, pid in enumerate(kept)}
    positions = np.full((len(frames), len(kept), 2), np.nan)
    for frame, view in enumerate(frames):
        for point_id, point in zip(view.point_ids.tolist(), view.image_points):
            if point_id in index:
                positions[frame, index[point_id]] = point

    scatter = np.nanmax(np.nanstd(positions, axis=0))
    if not np.isfinite(scatter):
        raise ValidationError("no corner had enough finite observations to measure")
    if scatter > MAX_STATIC_SCATTER_PX:
        raise ValidationError(
            f"corners move by up to {scatter:.1f} px across the capture, above the "
            f"{MAX_STATIC_SCATTER_PX:.0f} px this treats as static. That is motion, "
            "not detector noise: check the mount, the target and the lighting, and "
            "that no frame caught the board being adjusted"
        )
    return np.array(kept), positions, len(counts)


def _nearest_neighbours(mean_positions: np.ndarray) -> Tuple[np.ndarray, float]:
    """Each corner's nearest neighbour direction, and the median spacing.

    The direction stands in for the local board edge, which is what an
    anisotropic corner detector aligns with. The spacing is the unit the
    correlation length is judged in, because "do adjacent corners share noise"
    is the question that decides whether the independence assumption holds, and
    it cannot be asked in absolute pixels across different boards and
    distances.

    Args:
        mean_positions: An `(n, 2)` array of mean corner positions.

    Returns:
        One angle per corner in `[0, 180)`, and the median nearest-neighbour
        distance in pixels.
    """
    count = mean_positions.shape[0]
    if count < 2:
        return np.zeros(count), 0.0
    offsets = mean_positions[:, None, :] - mean_positions[None, :, :]
    distances = np.linalg.norm(offsets, axis=2)
    np.fill_diagonal(distances, np.inf)
    nearest = np.argmin(distances, axis=1)
    direction = mean_positions[nearest] - mean_positions
    angles = np.degrees(np.arctan2(direction[:, 1], direction[:, 0])) % 180.0
    return angles, float(np.median(distances[np.arange(count), nearest]))


def _remove_affine(positions: np.ndarray) -> np.ndarray:
    """Take out the part of each frame's deviation a rigid-ish motion explains.

    A calibration gives every view its own free pose, and for a planar target
    that pose can absorb close to a projective change of the corner field. An
    affine fit is the well-conditioned stand-in: on sub-pixel deviations the
    projective terms of a full homography are barely determined, while the six
    affine parameters capture the translation, rotation, scale and shear that a
    drifting or warming mount actually produces.

    What survives is the part of the noise a pose cannot take, which is the part
    that reaches the parameters.

    Args:
        positions: An `(n_frames, n_corners, 2)` array of corner positions.

    Returns:
        Deviations of the same shape, with the per-frame affine removed.
    """
    mean_positions = np.nanmean(positions, axis=0)
    design = np.column_stack(
        [mean_positions, np.ones(mean_positions.shape[0])]
    )
    residuals = np.full_like(positions, np.nan)
    for frame in range(positions.shape[0]):
        observed = positions[frame]
        usable = np.all(np.isfinite(observed), axis=1)
        if usable.sum() < 4:
            continue
        solution, *_ = np.linalg.lstsq(design[usable], observed[usable], rcond=None)
        residuals[frame] = observed - design @ solution
    return residuals


def _correlation_profile(
    deviations: np.ndarray, mean_positions: np.ndarray, n_bins: int = CORRELATION_BINS
) -> CorrelationProfile:
    """How strongly two corners' deviations move together, against separation.

    The statistic is a vector correlation, `<d_i . d_j>` normalised by each
    corner's own power, so both coordinates contribute at once and a field that
    translates a whole region together scores near one however it is oriented.

    Args:
        deviations: An `(n_frames, n_corners, 2)` array of deviations from the
            mean position.
        mean_positions: An `(n_corners, 2)` array of mean positions.
        n_bins: Separation bins.

    Returns:
        The binned profile and the interpolated `1/e` length.
    """
    with np.errstate(invalid="ignore"):
        with warnings.catch_warnings():
            # An all-NaN corner means "never co-observed", which the pair count
            # below discards; numpy warns about the empty mean regardless.
            warnings.simplefilter("ignore", RuntimeWarning)
            centred = deviations - np.nanmean(deviations, axis=0, keepdims=True)
    filled = np.nan_to_num(centred)
    valid = np.all(np.isfinite(centred), axis=2)

    # <d_i . d_j> over the frames where both corners were seen.
    joint = np.einsum("fik,fjk->ij", filled, filled)
    presence = valid.astype(float)
    pairs = presence.T @ presence

    # Each corner's power has to be taken over the same frames the pair shares,
    # not over all of its own. `shared[i, j]` is corner i's power across the
    # frames j was also seen in; normalising by anything wider biases every
    # correlation low whenever two corners were detected in different frames.
    shared = np.sum(filled ** 2, axis=2).T @ presence
    scale = np.sqrt(shared * shared.T)
    with np.errstate(invalid="ignore", divide="ignore"):
        correlation = np.where(scale > 0, joint / scale, 0.0)

    separation = np.linalg.norm(
        mean_positions[:, None, :] - mean_positions[None, :, :], axis=2
    )
    rows, columns = np.triu_indices(mean_positions.shape[0], k=1)
    usable = pairs[rows, columns] >= MIN_FRAMES
    rows, columns = rows[usable], columns[usable]
    if rows.size == 0:
        empty = np.zeros(0)
        return CorrelationProfile(empty, empty, empty.astype(int), 0.0)

    distances = separation[rows, columns]
    values = correlation[rows, columns]
    edges = np.linspace(0.0, float(distances.max()) + 1e-9, n_bins + 1)
    assignment = np.clip(np.digitize(distances, edges[1:-1]), 0, n_bins - 1)
    counts = np.bincount(assignment, minlength=n_bins)
    populated = counts > 0
    centres = (0.5 * (edges[:-1] + edges[1:]))[populated]
    means = (np.bincount(assignment, values, minlength=n_bins) / np.maximum(counts, 1))[
        populated
    ]
    return CorrelationProfile(
        separation_px=centres,
        correlation=means,
        pair_counts=counts[populated],
        length_px=_crossing(centres, means, CORRELATION_THRESHOLD),
    )


def _crossing(x: np.ndarray, y: np.ndarray, level: float) -> float:
    """Where a falling profile first crosses a level, linearly interpolated.

    Args:
        x: Bin centres, ascending.
        y: Values at those centres.
        level: The level to find.

    Returns:
        The interpolated crossing, `0.0` when the profile starts below the level
        and the last bin centre when it never reaches it. The last bin is a
        saturation rather than an extrapolation: a field smoother than the board
        has no measurable length, and inventing one past the widest separation
        observed would be a fabrication.
    """
    if x.size == 0:
        return 0.0
    if y[0] < level:
        return 0.0
    for index in range(1, x.size):
        if y[index] < level:
            # Everything before `index` is at or above the level, or the loop
            # would already have returned, so this span is strictly positive
            # and needs no guard.
            span = y[index - 1] - y[index]
            fraction = (y[index - 1] - level) / span
            return float(x[index - 1] + fraction * (x[index] - x[index - 1]))
    return float(x[-1])


def measure_noise_floor(
    observations: ObservationSet,
    min_presence: float = MIN_PRESENCE,
    n_bins: int = CORRELATION_BINS,
) -> NoiseFloor:
    """Measure corner noise from frames of a stationary scene.

    Args:
        observations: One view per frame, all of the same static scene. Build it
            with `caltrust.ingest.detect_in_images` over the captured frames.
        min_presence: Fraction of frames a corner must appear in to be used.
        n_bins: Separation bins in the correlation profile.

    Returns:
        The measured noise: its scale, its shape, and the correlation length
        that decides whether the classical covariance can be believed.

    Raises:
        ValidationError: Too few frames, too few consistently detected corners,
            scatter large enough that the scene cannot have been static, or an
            argument outside its range.
    """
    if not 0.0 < min_presence <= 1.0:
        raise ValidationError(
            f"min_presence is a fraction of the frames and must lie in (0, 1], "
            f"got {min_presence}"
        )
    if n_bins < 2:
        raise ValidationError(f"a correlation profile needs at least 2 bins, got {n_bins}")
    point_ids, positions, n_detected = _stack_static(observations, min_presence)
    mean_positions = np.nanmean(positions, axis=0)
    raw_deviations = positions - mean_positions
    irreducible = _remove_affine(positions)
    directions, spacing = _nearest_neighbours(mean_positions)

    corners: List[CornerNoise] = []
    for index, point_id in enumerate(point_ids.tolist()):
        samples = positions[:, index, :]
        usable = np.all(np.isfinite(samples), axis=1)
        if usable.sum() < MIN_FRAMES:
            continue
        corners.append(
            CornerNoise(
                point_id=int(point_id),
                n_frames=int(usable.sum()),
                mean_px=mean_positions[index],
                covariance=np.cov(samples[usable].T, ddof=1),
                irreducible_covariance=np.cov(
                    irreducible[usable, index, :].T, ddof=1
                ),
                neighbour_direction_deg=float(directions[index]),
            )
        )
    if not corners:
        raise ValidationError(
            f"no corner was seen in {MIN_FRAMES} frames after filtering"
        )

    def pooled(values: np.ndarray) -> float:
        finite = values[np.isfinite(values)]
        return float(np.sqrt(np.mean(finite ** 2))) if finite.size else 0.0

    # Per-frame mean displacement is the whole-frame component, which is rig
    # movement rather than anything the detector did.
    frame_shift = np.nanmean(raw_deviations, axis=1)
    return NoiseFloor(
        corners=tuple(corners),
        n_frames=observations.n_views,
        n_detected=n_detected,
        image_size=observations.image_size,
        total_sigma_px=pooled(raw_deviations),
        irreducible_sigma_px=pooled(irreducible),
        drift_px=float(np.nanmax(np.linalg.norm(frame_shift, axis=1))),
        spacing_px=spacing,
        correlation=_correlation_profile(irreducible, mean_positions, n_bins),
        total_correlation=_correlation_profile(raw_deviations, mean_positions, n_bins),
    )
