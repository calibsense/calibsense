# calibsense - measurement uncertainty for camera calibration.
# Copyright (C) 2026 Abhishek Gola
#
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License, version 3, as published by
# the Free Software Foundation. This program is distributed WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the LICENSE file, or <https://www.gnu.org/licenses/>.

"""Parameter covariance, in the factored form the block structure gives for free.

This is failure 2 from the problem statement. OpenCV's `calibrateCameraExtended`
returns marginal standard deviations: a column of numbers that says how uncertain
each parameter is on its own, and nothing about how they trade off against each
other. The trade-offs are the whole story. Focal length and working distance are
the classic pair, and only the off-diagonal entries can tell you they are
confounded.

The full covariance is never materialised unless asked for. It is kept as
`sigma^2 * S^-1` for the intrinsics plus per-view factors, from which any block
is reconstructed on demand:

* `cov(intrinsics)        = s2 * S^-1`
* `cov(intrinsics, pose_i)= -s2 * S^-1 @ Y_i.T`
* `cov(pose_i, pose_j)    = s2 * (V_i^-1 [i == j] + Y_i @ S^-1 @ Y_j.T)`

with `Y_i = V_i^-1 @ W_i.T`.

Two estimates come out of this, not one. `sigma^2 * S^-1` is the classical
result and assumes corner noise is independent, isotropic and homoscedastic.
`RobustCovariance` is the same quantity computed from the scatter of the views
themselves, and assumes only that different views are independent. Reporting
both is what turns the noise model from an assertion into a measurement: when
they agree the assumption held for this capture, and when they do not the
classical interval is understating the uncertainty by the ratio between them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..core.parameters import extrinsic_names
from ..errors import DegenerateSystemError, ValidationError
from .linalg import (
    DEFAULT_RCOND,
    SymmetricInverse,
    correlation_from_covariance,
    dominant_terms,
    invert_symmetric,
    jacobi_scale,
    top_correlations,
)
from .normal import POSE_DIMENSION, NormalEquations

#: Refuse to materialise a dense covariance larger than this, in entries.
DENSE_LIMIT = 64_000_000

#: Fewest views a view-clustered covariance needs before it is worth quoting.
#: An estimate built from `G` clusters has roughly `G - 1` degrees of freedom, and
#: below about ten it is itself so noisy that quoting it would trade one
#: unchecked number for another. Across repeated realisations of the same rig its
#: own relative spread, re-measured for the leverage-corrected form:
#:
#:      G = 8     32-48%
#:      G = 10    29-42%
#:      G = 14    26-36%
#:      G = 20    18-29%
#:      G = 30    16-27%
#:
#: Ten is where the spread stops growing sharply rather than where it becomes
#: small; it is still a third at that point, which is why `inflation_floor`
#: carries a `G` term instead of the finding comparing against a constant.
MIN_CLUSTERS_FOR_ROBUST = 10

#: Scaled eigenvalue ratio below which a direction counts as poorly determined.
#: Chosen so that a direction contributing less than about one part in a
#: thousand of the best-determined direction's curvature is called weak.
WEAK_DIRECTION_THRESHOLD = 1e-6


@dataclass(frozen=True)
class WeakDirection:
    """A parameter combination the capture barely constrains.

    Attributes:
        eigenvalue_ratio: Curvature along this direction relative to the
            best-constrained direction, after Jacobi scaling. Smaller is worse;
            zero means the direction is not constrained at all.
        terms: The parameters that dominate the direction, largest weight first,
            as `(name, weight)` pairs over a unit vector.
        vector: The full direction in Jacobi-scaled parameter space.
    """

    eigenvalue_ratio: float
    terms: List[Tuple[str, float]]
    vector: np.ndarray

    def describe(self) -> str:
        """A one-line description naming the parameters involved."""
        parts = " ".join(
            f"{weight:+.2f}*{name}" for name, weight in self.terms
        )
        return f"{parts} (curvature ratio {self.eigenvalue_ratio:.2e})"


@dataclass(frozen=True)
class RobustCovariance:
    """Intrinsic covariance that assumes only that views are independent.

    `sigma^2 * S^-1` estimates the noise *scale* from the data but asserts its
    *shape*: one variance, the same in x and y, uncorrelated between points.
    Measured against Monte Carlo, two of the ways real corner noise departs from
    that turn out not to matter — anisotropy along the edge direction and noise
    that is worse in some views than others both leave the classical covariance
    within about fifteen per cent, because several hundred corners at varied
    orientations average them out and `cost / dof` picks up the average power.

    Spatial correlation across the frame is the one that bites, and it bites
    hard. Noise that varies smoothly over the image — field-varying defocus,
    an illumination gradient pulling on the sub-pixel estimator, a bowed board,
    motion blur, rolling shutter — is partly absorbable by the pose and
    distortion parameters, so it *lowers* the residual while *raising* the
    estimator's true spread. On a synthetic rig with a 200 px correlation
    length, `sigma` fell from 0.25 px to 0.055 px while the actual spread of
    `fx` rose from 3.0 px to 5.2 px: a beautiful RMS and an interval nearly
    eight times too tight, with nothing in the report to say so.

    Weights cannot fix that, which is why this is not the per-point weighting
    route. The error lives in the off-diagonal of the noise covariance, and no
    amount of reweighting a diagonal repairs a correlation.

    So this estimate drops the assumption instead of refining it. Each view
    contributes one score `s_i` for the intrinsics, and the covariance comes
    from the scatter of the displacement each view would cause if it were
    dropped:

        d_i = S^-1/2 (I - M_i)^-1/2 S^-1/2 s_i,   M_i = S^-1/2 S_i S^-1/2

        Cov = (G - 1) / G * sum_i (d_i - d_bar)(d_i - d_bar)'

    `M_i` is view `i`'s leverage, and the correction matters. Without it — the
    plain sandwich with a `G / (G - 1)` scaling, which is what this used to be —
    residuals evaluated at a fitted optimum are shrunk towards it, and the
    estimate came out at 0.73 of the true spread at fourteen views under
    correlated noise. The full leave-one-out correction, `(I - M_i)^-1`,
    overshoots to 1.37. The square root sits at 0.95, and the same ordering
    holds at thirty views and under noise that obeys the model. See
    `leave_one_out_steps`.

    A consequence worth stating plainly: the correction makes this estimate
    deliberately conservative, so it sits about a fifth above the classical one
    even when the noise model is perfectly satisfied. That offset is not a
    signal, and `diagnose.noise` compares against a floor measured from this
    null rather than against one.

    Nothing is assumed about the noise *within* a view, so anisotropy, blur,
    neighbour correlation and heavy tails are all absorbed rather than
    modelled. Only the independence of different views is relied on. That is
    the same argument `calibsense.diagnose.model` already makes for the radial
    residual profile, applied to the parameters instead of the residuals.

    Three things it cannot do. It is blind to any error that is *identical*
    across views — a board scale error moves every view coherently, so it
    contributes nothing to the between-view scatter and stays invisible here.
    And it covers the intrinsics only: each view's pose is estimated from that
    view's own residuals, so there is exactly one cluster per pose and no
    between-cluster scatter to build a pose block from. And it wants the views
    to outnumber the free intrinsics: leverages sum to `p` across `G` views, so
    as `p` approaches `G` every view becomes near-decisive and the correction is
    applied at the edge of where it is defined. `MAX_VIEW_LEVERAGE` keeps that
    finite rather than letting it run away, and it was enough to hold a
    fourteen-coefficient fit on twelve views to a ratio of 1.4 — but a model
    that over-parameterised has an identifiability problem the conditioning
    report names long before this one matters.

    Attributes:
        covariance: Sandwich covariance of the free intrinsics, shape `(p, p)`.
        n_clusters: Views that contributed a score.
        intrinsic_names: Names of the free intrinsics, matching the rows.
    """

    covariance: np.ndarray
    n_clusters: int
    intrinsic_names: Tuple[str, ...]

    @property
    def degrees_of_freedom(self) -> int:
        """Clusters minus one.

        The scores sum to zero at the optimum, so the between-view scatter
        carries one fewer degree of freedom than there are views. An interval
        built from `std` wants a `t` multiplier on this many degrees of freedom
        rather than the normal one the classical deviations are usually read
        with — at fourteen views that is about eight per cent wider.
        """
        return max(self.n_clusters - 1, 0)

    @property
    def usable(self) -> bool:
        """Whether there are enough views for the estimate to mean anything.

        `False` does not mean the number below is wrong, only that it is too
        noisy to quote against the classical one. Reports say so rather than
        showing a ratio nobody should act on.
        """
        return self.n_clusters >= MIN_CLUSTERS_FOR_ROBUST

    def std(self) -> np.ndarray:
        """Standard deviation of each free intrinsic under this estimate."""
        return np.sqrt(np.clip(np.diag(self.covariance), 0.0, None))


@dataclass(frozen=True)
class CalibrationCovariance:
    """The covariance of a calibration, held in factored form.

    Attributes:
        sigma2: Estimated residual variance in square pixels, `cost / dof`.
        degrees_of_freedom: Residuals minus free parameters.
        intrinsic: Covariance of the free intrinsics, shape `(p, p)`.
        intrinsic_names: Names of those free intrinsics.
        n_views: Number of views.
        spectrum: Eigen-report of the Schur complement, including its rank,
            condition numbers and unconstrained directions.
        singular_views: Views whose own pose block was rank deficient, usually
            because the view holds too few or too nearly collinear points.
        robust: The same intrinsic covariance estimated from the scatter of the
            views instead of from `sigma^2`, which is what checks the noise
            model rather than assuming it. `None` when the equations did not
            carry per-view scores, which happens only for a fit restored from an
            old bundle.
    """

    sigma2: float
    degrees_of_freedom: int
    intrinsic: np.ndarray
    intrinsic_names: Tuple[str, ...]
    n_views: int
    spectrum: SymmetricInverse
    singular_views: np.ndarray
    _schur_inverse: np.ndarray
    _pose_inverse: np.ndarray
    _y: np.ndarray
    _scaled_eigenvalues: np.ndarray
    _scaled_vectors: np.ndarray
    robust: Optional[RobustCovariance] = None

    @property
    def n_intrinsic(self) -> int:
        """Number of free intrinsic parameters."""
        return len(self.intrinsic_names)

    @property
    def n_parameters(self) -> int:
        """Total free parameters, intrinsics plus six per view."""
        return self.n_intrinsic + POSE_DIMENSION * self.n_views

    @property
    def sigma(self) -> float:
        """Residual standard deviation in pixels, per residual component."""
        return float(np.sqrt(self.sigma2))

    def parameter_names(self) -> Tuple[str, ...]:
        """Names of every free parameter, intrinsics first then each view's pose."""
        names: List[str] = list(self.intrinsic_names)
        for index in range(self.n_views):
            names.extend(extrinsic_names(index))
        return tuple(names)

    def intrinsic_std(self) -> np.ndarray:
        """Standard deviation of each free intrinsic.

        This is the quantity `cv2.calibrateCameraExtended` returns, and the test
        suite checks the two agree.
        """
        return np.sqrt(np.clip(np.diag(self.intrinsic), 0.0, None))

    def robust_inflation(self) -> np.ndarray:
        """How much wider the view-clustered deviation is, per free intrinsic.

        This is the number that says whether the noise model can be believed
        *for this capture*. One means the classical deviation and the estimate
        that assumes nothing about the noise shape agree, so `sigma^2 * S^-1` is
        defensible here by measurement rather than by assertion. Three means the
        reported interval is a third of what the data supports.

        Returns:
            One ratio per free intrinsic, or an empty array when no robust
            estimate is available. Entries whose classical deviation is zero —
            the directions a pseudo-inverse dropped — come back as `nan`, since
            there is no interval there to widen.
        """
        if self.robust is None:
            return np.zeros(0)
        classical = self.intrinsic_std()
        determined = classical > 0
        safe = np.where(determined, classical, 1.0)
        return np.where(determined, self.robust.std() / safe, np.nan)

    def worst_robust_inflation(self) -> float:
        """The largest inflation across the free intrinsics.

        Returns:
            The headline ratio, or `nan` when there is no robust estimate or
            every parameter sits in a dropped direction.
        """
        ratios = self.robust_inflation()
        if ratios.size == 0 or not np.any(np.isfinite(ratios)):
            return float("nan")
        return float(np.nanmax(ratios))

    def widened_intrinsic(
        self, rcond: float = DEFAULT_RCOND
    ) -> Tuple[np.ndarray, float]:
        """The intrinsic covariance widened wherever the views say it is too tight.

        `worst_robust_inflation` names a factor and leaves the reader to apply
        it. This applies it, and does so without inventing a scalar to multiply
        the whole matrix by.

        The two estimates are compared in the basis that whitens the classical
        one: solving `A_r v = lambda A v` gives, per direction, the ratio of the
        two variances. Every `lambda` above one is a direction where the
        between-view scatter is wider than the noise model predicted, and those
        are raised; every `lambda` below one is left alone. Clamping rather than
        substituting matters, because the sandwich is mildly optimistic at low
        view counts and swapping it in wholesale would *narrow* the intervals on
        a clean short capture on the strength of a noisy estimate. Widening only
        where there is evidence to widen cannot make a report overconfident.

        Directions the fit did not determine at all keep their zero variance. A
        pseudo-inverse assigned them none, and there is no interval there to
        widen — `bounded` is what reports that, not this.

        Args:
            rcond: Numerical cut for the whitening. Eigenvalues of the classical
                covariance below this fraction of the largest are the ones a
                pseudo-inverse already zeroed, and they stay zero. This is a
                machine-precision cut and not `WEAK_DIRECTION_THRESHOLD`: the
                distortion coefficients have variances eight or nine orders
                below the focal length's on a perfectly healthy capture, so the
                weak-direction cut would discard them as unconstrained and
                silently drop them out of the sampling.

        Returns:
            `(covariance, inflation)` — the widened matrix, and the largest
            ratio of widened to classical standard deviation across the free
            intrinsics. An inflation of one means the matrix came back
            untouched.

            The ratio is reported per parameter rather than per direction on
            purpose. The largest ratio over *any* linear combination is a much
            noisier number — on a clean 24-view capture it reaches 1.6 while no
            parameter's own deviation moves by more than four per cent, because
            the direction it picks out carries almost no variance. Quoting that
            would raise an alarm the data does not support, and the per-
            parameter figure is the one a reader can check against the parameter
            table.
        """
        classical = np.asarray(self.intrinsic, dtype=float)
        if self.robust is None or not self.robust.usable:
            return classical, 1.0

        eigenvalues, vectors = np.linalg.eigh(0.5 * (classical + classical.T))
        largest = float(eigenvalues[-1]) if eigenvalues.size else 0.0
        keep = eigenvalues > max(largest, 0.0) * rcond
        if not np.any(keep):
            return classical, 1.0

        root = vectors[:, keep] * np.sqrt(eigenvalues[keep])
        whitener = vectors[:, keep] / np.sqrt(eigenvalues[keep])
        ratios, directions = np.linalg.eigh(
            whitener.T @ np.asarray(self.robust.covariance, dtype=float) @ whitener
        )
        scaled = (root @ directions) * np.maximum(ratios, 1.0) ** 0.5
        matrix = scaled @ scaled.T

        # At least one eigenvalue survived the cut above, so the trace is
        # positive and some diagonal entry is too; there is always something to
        # take a ratio against here.
        before = np.diag(classical)
        determined = before > 0
        inflation = np.sqrt((np.diag(matrix)[determined] / before[determined]).max())
        return matrix, float(max(inflation, 1.0))

    def extrinsic_block(self, view: int) -> np.ndarray:
        """Covariance of one view's pose parameters.

        Args:
            view: View index.

        Returns:
            A `(6, 6)` covariance over `[rx, ry, rz, tx, ty, tz]`.
        """
        self._check_view(view)
        y = self._y[view]
        return self.sigma2 * (
            self._pose_inverse[view] + y @ self._schur_inverse @ y.T
        )

    def cross_block(self, view: int) -> np.ndarray:
        """Covariance between the intrinsics and one view's pose.

        Args:
            view: View index.

        Returns:
            A `(p, 6)` block. Its correlations are where the focal-length and
            distance confounding becomes visible.
        """
        self._check_view(view)
        return -self.sigma2 * (self._schur_inverse @ self._y[view].T)

    def pose_cross_block(self, view_a: int, view_b: int) -> np.ndarray:
        """Covariance between two views' poses.

        Args:
            view_a: First view index.
            view_b: Second view index.

        Returns:
            A `(6, 6)` block. Off-diagonal view pairs are correlated only
            through the shared intrinsics, which is exactly what makes a
            calibration's views non-independent.
        """
        self._check_view(view_a)
        self._check_view(view_b)
        shared = self._y[view_a] @ self._schur_inverse @ self._y[view_b].T
        if view_a == view_b:
            return self.sigma2 * (self._pose_inverse[view_a] + shared)
        return self.sigma2 * shared

    def dense(self, limit: int = DENSE_LIMIT) -> np.ndarray:
        """Materialise the full covariance matrix.

        Args:
            limit: Refuse to build a matrix with more entries than this.

        Returns:
            A `(n_parameters, n_parameters)` covariance.

        Raises:
            ValidationError: The matrix would exceed `limit`.
        """
        size = self.n_parameters
        if size * size > limit:
            raise ValidationError(
                f"a dense covariance would be {size}x{size} ({size * size:,} entries), "
                f"above the limit of {limit:,}. Use the per-view blocks instead."
            )
        dense = np.zeros((size, size))
        p = self.n_intrinsic
        dense[:p, :p] = self.intrinsic
        for view in range(self.n_views):
            start = p + POSE_DIMENSION * view
            stop = start + POSE_DIMENSION
            cross = self.cross_block(view)
            dense[:p, start:stop] = cross
            dense[start:stop, :p] = cross.T
            for other in range(self.n_views):
                other_start = p + POSE_DIMENSION * other
                dense[start:stop, other_start : other_start + POSE_DIMENSION] = (
                    self.pose_cross_block(view, other)
                )
        return 0.5 * (dense + dense.T)

    def intrinsic_correlation(self) -> np.ndarray:
        """Correlation matrix of the free intrinsics."""
        return correlation_from_covariance(self.intrinsic)

    def correlation(self, limit: int = DENSE_LIMIT) -> np.ndarray:
        """Correlation matrix over every free parameter.

        Args:
            limit: Passed through to `dense`.

        Returns:
            A `(n_parameters, n_parameters)` correlation matrix.
        """
        return correlation_from_covariance(self.dense(limit))

    def correlation_with_poses(self, parameter: str) -> np.ndarray:
        """Correlation of one intrinsic with every view's six pose parameters.

        This is the cheap route to the headline diagnostic: passing `"fx"` and
        reading the `tz` column tells you directly whether focal length and
        working distance are confounded, without building a dense matrix.

        Args:
            parameter: Name of a free intrinsic, such as `"fx"`.

        Returns:
            A `(n_views, 6)` array of correlations, columns ordered
            `[rx, ry, rz, tx, ty, tz]`.

        Raises:
            ValidationError: The parameter is not a free intrinsic.
        """
        try:
            index = self.intrinsic_names.index(parameter)
        except ValueError:
            raise ValidationError(
                f"{parameter!r} is not a free intrinsic; have "
                f"{list(self.intrinsic_names)}"
            ) from None
        own_std = float(np.sqrt(max(self.intrinsic[index, index], 0.0)))
        result = np.zeros((self.n_views, POSE_DIMENSION))
        if own_std <= 0:
            return result
        for view in range(self.n_views):
            pose_std = np.sqrt(np.clip(np.diag(self.extrinsic_block(view)), 0.0, None))
            safe = np.where(pose_std > 0, pose_std, 1.0)
            result[view] = self.cross_block(view)[index] / (own_std * safe)
        return np.clip(result, -1.0, 1.0)

    def strongest_correlations(
        self, count: int = 10, threshold: float = 0.0
    ) -> List[Tuple[str, str, float]]:
        """Most strongly correlated intrinsic pairs, strongest first.

        Args:
            count: How many pairs to return.
            threshold: Ignore pairs below this absolute correlation.

        Returns:
            Triples of `(name_a, name_b, correlation)`.
        """
        return top_correlations(
            self.intrinsic_correlation(), self.intrinsic_names, count, threshold
        )

    def unconstrained_directions(self, count: int = 3) -> List[List[Tuple[str, float]]]:
        """Describe each unconstrained direction by its dominant parameters.

        An empty list means the intrinsics are fully determined by the data. A
        non-empty one names the parameter combinations the capture cannot
        separate at all.

        Args:
            count: How many parameters to name per direction.

        Returns:
            One list of `(name, weight)` pairs per null-space direction.
        """
        return [
            dominant_terms(self.spectrum.null_space[:, i], self.intrinsic_names, count)
            for i in range(self.spectrum.null_space.shape[1])
        ]

    def weak_directions(
        self, threshold: float = WEAK_DIRECTION_THRESHOLD
    ) -> List[WeakDirection]:
        """Parameter combinations this capture does not determine.

        Read this before reading `intrinsic_std`. When a direction is
        unconstrained, the covariance along it is not large — it is *absent*,
        because a pseudo-inverse assigns it zero variance. A standard deviation
        computed under those conditions looks confident and means nothing, and
        this method is what says so.

        The test is run on the Jacobi-scaled Schur complement, so a focal length
        in pixels and a distortion coefficient near zero are compared on equal
        terms rather than by the accident of their units.

        Args:
            threshold: Scaled eigenvalue ratio below which a direction counts as
                weak.

        Returns:
            One entry per weak direction, worst first. An empty list means every
            intrinsic combination is determined by the data.
        """
        largest = float(self._scaled_eigenvalues[-1])
        if largest <= 0:
            return []
        ratios = np.clip(self._scaled_eigenvalues / largest, 0.0, None)
        weak = np.flatnonzero(ratios < threshold)
        return [
            WeakDirection(
                eigenvalue_ratio=float(ratios[i]),
                terms=dominant_terms(
                    self._scaled_vectors[:, i], self.intrinsic_names, 3
                ),
                vector=np.ascontiguousarray(self._scaled_vectors[:, i]),
            )
            for i in weak
        ]

    def parameter_participation(
        self, threshold: float = WEAK_DIRECTION_THRESHOLD
    ) -> np.ndarray:
        """How much of each intrinsic lies in the poorly determined subspace.

        Args:
            threshold: Passed to `weak_directions`.

        Returns:
            One value per free intrinsic, in `[0, 1]`. A value near one means
            that parameter is essentially not determined by this capture,
            whatever its reported standard deviation says.
        """
        directions = self.weak_directions(threshold)
        if not directions:
            return np.zeros(self.n_intrinsic)
        stacked = np.stack([d.vector for d in directions], axis=1)
        return np.clip(np.sqrt(np.sum(stacked ** 2, axis=1)), 0.0, 1.0)

    def is_identifiable(self, threshold: float = WEAK_DIRECTION_THRESHOLD) -> bool:
        """Whether every intrinsic combination is determined by the data.

        Args:
            threshold: Passed to `weak_directions`.

        Returns:
            `True` when there is no weak direction.
        """
        return not self.weak_directions(threshold)

    def newton_decrement(
        self, gradient_intrinsic: np.ndarray, gradient_pose: np.ndarray
    ) -> float:
        """The cost reduction a full Newton step would predict.

        This is the scale-invariant way to ask whether a parameter set sits at
        the optimum of its own objective. A raw gradient norm cannot answer that,
        because it carries the units of the parameters and grows with the number
        of residuals; `g' N^-1 g` compared against the current cost does not.

        Args:
            gradient_intrinsic: `J_intrinsic.T @ residual`, length `p`.
            gradient_pose: `J_pose.T @ residual` per view, shape `(v, 6)`.

        Returns:
            `g' N^-1 g`, in the same units as the sum of squared residuals.
        """
        intrinsic = np.asarray(gradient_intrinsic, dtype=float).reshape(-1)
        pose = np.asarray(gradient_pose, dtype=float).reshape(self.n_views, -1)
        reduced = intrinsic - np.einsum("nij,ni->j", self._y, pose)
        pose_term = float(np.einsum("ni,nij,nj->", pose, self._pose_inverse, pose))
        return float(reduced @ self._schur_inverse @ reduced) + pose_term

    def _check_view(self, view: int) -> None:
        if not 0 <= view < self.n_views:
            raise ValidationError(
                f"view index {view} out of range for {self.n_views} views"
            )


def covariance_from_normal_equations(
    equations: NormalEquations, rcond: float = DEFAULT_RCOND
) -> CalibrationCovariance:
    """Turn assembled normal equations into a covariance.

    Args:
        equations: The normal equations, as built by `refit.normal.assemble`.
        rcond: Relative eigenvalue cut for the Schur complement and the per-view
            pose blocks.

    Returns:
        The covariance in factored form.

    Raises:
        DegenerateSystemError: There are no degrees of freedom left, so no
            variance can be estimated, or the intrinsic system carries no
            information at all.
    """
    dof = equations.degrees_of_freedom
    if dof <= 0:
        raise DegenerateSystemError(
            f"{equations.n_residuals} residuals against {equations.n_parameters} "
            "free parameters leaves no degrees of freedom; add views, add points, "
            "or fix parameters the data cannot support"
        )
    schur, pose_inverse, y, singular = equations.schur_complement(rcond)
    spectrum = invert_symmetric(schur, rcond)
    scaled_eigenvalues, scaled_vectors = np.linalg.eigh(jacobi_scale(schur))
    sigma2 = equations.cost / dof
    return CalibrationCovariance(
        sigma2=float(sigma2),
        degrees_of_freedom=int(dof),
        intrinsic=sigma2 * spectrum.inverse,
        intrinsic_names=equations.intrinsic_names,
        n_views=equations.n_views,
        spectrum=spectrum,
        singular_views=np.asarray(singular, dtype=bool),
        _schur_inverse=spectrum.inverse,
        _pose_inverse=pose_inverse,
        _y=y,
        _scaled_eigenvalues=scaled_eigenvalues,
        _scaled_vectors=scaled_vectors,
        robust=_robust_covariance(equations, spectrum.inverse, rcond),
    )


#: Largest per-view leverage the CR2 correction will act on. A view that alone
#: determined a direction would have leverage one there and an infinite
#: correction; clamping just below one keeps such a direction from swamping the
#: estimate, and a capture with a view that dominant has an identifiability
#: problem the conditioning report names first.
MAX_VIEW_LEVERAGE = 0.99


def _symmetric_root(matrix: np.ndarray, rcond: float) -> np.ndarray:
    """The symmetric square root of a positive semi-definite matrix.

    Only ever used on a *covariance*, never on an information matrix. Taking a
    negative power of the Schur complement directly would need a cut that says
    which of its eigenvalues are zero, and its eigenvalues carry the units of
    the parameters — a focal length in pixels beside a distortion coefficient
    that is dimensionless — so any relative cut on them means something
    different for each parameter and silently discards whichever ones happen to
    be small in their own units. Rooting the pseudo-inverse instead inherits the
    rank decision the module already made in the Jacobi-scaled basis, and a
    positive power cannot amplify anything.

    Args:
        matrix: The matrix to root.
        rcond: Eigenvalues below this fraction of the largest are treated as
            zero. Only exact null directions are at stake here, because the
            power is positive.

    Returns:
        The root, symmetric by construction.
    """
    eigenvalues, vectors = np.linalg.eigh(0.5 * (matrix + matrix.T))
    largest = max(float(eigenvalues[-1]) if eigenvalues.size else 0.0, 0.0)
    keep = eigenvalues > largest * rcond
    scaled = np.zeros_like(eigenvalues)
    scaled[keep] = np.sqrt(eigenvalues[keep])
    return (vectors * scaled) @ vectors.T


def leave_one_out_steps(
    information: np.ndarray,
    scores: np.ndarray,
    schur_inverse: np.ndarray,
    rcond: float = DEFAULT_RCOND,
) -> np.ndarray:
    """How far the intrinsics would move if each view were dropped, to first order.

    Removing view `i` leaves the remaining gradient at `-s_i`, because the total
    gradient vanishes at the optimum, and the remaining curvature at `S - S_i`.
    One Newton step is therefore `(S - S_i)^-1 s_i`, which is the *exact*
    leave-one-out displacement for a linear model and first-order correct here —
    the order at which everything else in this module is computed. Measured
    against actually refitting without each view, the two agree to within six per
    cent in scale and correlate above 0.99 per view, so the jackknife this
    enables costs a `p x p` solve rather than `G` refits.

    What is returned is the **CR2** form rather than that step. Writing the
    view's leverage as `M_i = S^-1/2 S_i S^-1/2`, the leave-one-out step is
    `S^-1/2 (I - M_i)^-1 S^-1/2 s_i` and this returns

        S^-1/2 (I - M_i)^-1/2 S^-1/2 s_i

    the square root of the same correction. The reason is measured rather than
    aesthetic. Against the empirical spread of repeated refits, the uncorrected
    scores (the plain sandwich) run 0.73 of the truth under correlated noise and
    the full leave-one-out correction runs 1.37, overshooting as badly as the
    sandwich undershot; the square root sits at 0.95. The same ordering holds at
    thirty views and under noise that obeys the model.

    Args:
        information: Each view's contribution to the intrinsic information,
            shape `(v, p, p)`, summing to the Schur complement.
        scores: Each view's score for the intrinsics, shape `(v, p)`.
        schur_inverse: `S^+`, already computed for the classical covariance. Its
            square root is what the leverage is measured in, so the correction
            inherits the module's existing rank decision rather than making a
            second one in the parameters' own units.
        rcond: Relative eigenvalue cut for the root, which only has to separate
            exact null directions.

    Returns:
        A `(v, p)` array of displacements, one per view.
    """
    root_inverse = _symmetric_root(schur_inverse, rcond)
    steps = np.empty_like(scores)
    for index, block in enumerate(information):
        leverage = root_inverse @ block @ root_inverse
        eigenvalues, vectors = np.linalg.eigh(0.5 * (leverage + leverage.T))
        remaining = 1.0 - np.clip(eigenvalues, 0.0, MAX_VIEW_LEVERAGE)
        adjust = (vectors * remaining ** -0.5) @ vectors.T
        steps[index] = root_inverse @ adjust @ root_inverse @ scores[index]
    return steps


def _robust_covariance(
    equations: NormalEquations, schur_inverse: np.ndarray, rcond: float
) -> Optional[RobustCovariance]:
    """The view-clustered sandwich, when the equations carry per-view scores.

    Args:
        equations: The assembled normal equations.
        schur_inverse: `S^-1`, already computed for the classical covariance.
        rcond: Relative eigenvalue cut, passed through to the pose inverses.

    Returns:
        The robust estimate, or `None` for equations restored from a bundle old
        enough not to have stored the per-view intrinsic gradient.
    """
    if equations.gradient_intrinsic_view is None:
        return None
    scores = equations.view_scores(rcond)
    clusters = int(scores.shape[0])
    if clusters < 2:
        return None

    information = equations.schur_contributions(rcond)
    steps = leave_one_out_steps(information, scores, schur_inverse, rcond)
    centred = steps - steps.mean(axis=0)
    covariance = centred.T @ centred * ((clusters - 1.0) / clusters)
    return RobustCovariance(
        covariance=0.5 * (covariance + covariance.T),
        n_clusters=clusters,
        intrinsic_names=equations.intrinsic_names,
    )
