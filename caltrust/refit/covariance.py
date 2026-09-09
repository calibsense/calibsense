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
    )
