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

"""Which parameters are free, and how fixed ones are removed from the estimate.

Calibration flags do not merely decorate a fit; they change the dimension of the
estimated parameter space. A covariance computed over parameters that were held
fixed is singular, and a covariance that ignores a tie such as a fixed aspect
ratio is wrong rather than merely singular. Both cases are handled here by one
object: a linear reduction from the full parameter vector to the free one.

The relationship is `dtheta = T @ dphi`, where `theta` is the full parameter
vector, `phi` the free parameters actually estimated, and `T` the reduction
matrix. Covariance maps back the same way, `cov_theta = T @ cov_phi @ T.T`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence, Tuple

import numpy as np

from ..errors import ValidationError

#: A tie states that `follower` moves with `leader` at a fixed ratio:
#: `dtheta[follower] = scale * dtheta[leader]`.
Tie = Tuple[int, int, float]


@dataclass(frozen=True)
class ParameterBlock:
    """A named parameter vector with fixed entries and linear ties.

    Attributes:
        names: Name of every parameter, in full-vector order.
        free: Boolean mask, `True` where the parameter is estimated.
        ties: Proportional constraints between free parameters, as
            `(follower, leader, scale)` index triples.
    """

    names: Tuple[str, ...]
    free: np.ndarray
    ties: Tuple[Tie, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        free = np.asarray(self.free, dtype=bool).reshape(-1)
        object.__setattr__(self, "free", free)
        object.__setattr__(self, "names", tuple(self.names))
        object.__setattr__(self, "ties", tuple(tuple(t) for t in self.ties))
        if free.size != len(self.names):
            raise ValidationError(
                f"free mask has {free.size} entries for {len(self.names)} names"
            )
        followers = set()
        leaders = set()
        for follower, leader, scale in self.ties:
            if not (0 <= follower < free.size and 0 <= leader < free.size):
                raise ValidationError(f"tie index out of range: {(follower, leader)}")
            if follower == leader:
                raise ValidationError("a parameter cannot be tied to itself")
            if follower in followers:
                raise ValidationError(f"{self.names[follower]!r} is tied twice")
            if not np.isfinite(scale) or scale == 0.0:
                raise ValidationError(f"tie scale must be finite and non-zero, got {scale}")
            followers.add(follower)
            leaders.add(leader)
        if followers & leaders:
            raise ValidationError("chained ties are not supported")

    @property
    def n_full(self) -> int:
        """Number of parameters in the full vector, fixed ones included."""
        return len(self.names)

    @property
    def n_free(self) -> int:
        """Number of independently estimated parameters."""
        return self.reduction().shape[1]

    def free_names(self) -> Tuple[str, ...]:
        """Names of the independently estimated parameters, in reduced order.

        A tied follower does not get its own name; it is absorbed into its
        leader's column.
        """
        followers = {f for f, _, _ in self.ties if self.free[f]}
        return tuple(
            name
            for i, name in enumerate(self.names)
            if self.free[i] and i not in followers
        )

    def reduction(self) -> np.ndarray:
        """Build the reduction matrix `T` of shape `(n_full, n_free)`.

        Returns:
            A matrix mapping a free-parameter step to a full-parameter step.
            With no ties it is a column selection; a tie adds the follower's
            scaled row to its leader's column.
        """
        followers = {f: (l, s) for f, l, s in self.ties if self.free[f]}
        columns = [
            i for i in range(self.n_full) if self.free[i] and i not in followers
        ]
        column_of = {idx: col for col, idx in enumerate(columns)}
        reduction = np.zeros((self.n_full, len(columns)))
        for idx, col in column_of.items():
            reduction[idx, col] = 1.0
        for follower, (leader, scale) in followers.items():
            if leader not in column_of:
                raise ValidationError(
                    f"{self.names[follower]!r} is tied to {self.names[leader]!r}, "
                    "which is not free"
                )
            reduction[follower, column_of[leader]] = scale
        return reduction

    def expand(self, values_free: Sequence[float]) -> np.ndarray:
        """Map a free-parameter step back into the full parameter space.

        Args:
            values_free: A step in the reduced space, length `n_free`.

        Returns:
            The corresponding full-length step, zero at fixed parameters.
        """
        values = np.asarray(values_free, dtype=float).reshape(-1)
        if values.size != self.n_free:
            raise ValidationError(
                f"expected {self.n_free} free values, got {values.size}"
            )
        return self.reduction() @ values

    def with_all_free(self) -> "ParameterBlock":
        """Return a copy with every parameter free and no ties."""
        return ParameterBlock(self.names, np.ones(self.n_full, dtype=bool), ())


def extrinsic_names(view_index: int) -> Tuple[str, ...]:
    """Parameter names for one view's pose.

    Args:
        view_index: Zero-based index of the view.

    Returns:
        Six names, three rotation-vector components then three translation
        components, each prefixed with the view index so names stay unique
        across a whole problem.
    """
    prefix = f"view{view_index}"
    return tuple(f"{prefix}.{n}" for n in ("rx", "ry", "rz", "tx", "ty", "tz"))
