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

"""Symmetric inversion, conditioning and correlation.

The normal equations of a badly posed calibration are near-singular by
construction, and that near-singularity is the finding, not an inconvenience.
Nothing here calls `numpy.linalg.inv`: every inverse goes through an
eigendecomposition that reports the rank, the condition number and the
directions in parameter space the data does not constrain.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..errors import DegenerateSystemError, ValidationError

#: Relative eigenvalue below which a direction counts as unconstrained.
DEFAULT_RCOND = 1e-12


@dataclass(frozen=True)
class SymmetricInverse:
    """The result of inverting a symmetric positive semi-definite matrix.

    Attributes:
        inverse: The inverse, or the Moore-Penrose pseudo-inverse when the
            matrix is rank deficient.
        eigenvalues: Eigenvalues in ascending order.
        rank: Number of eigenvalues above the cut.
        condition_number: Ratio of largest to smallest eigenvalue, `inf` when
            rank deficient. Units-dependent, so read `scaled_condition_number`
            when comparing across problems.
        scaled_condition_number: Condition number after Jacobi scaling, which
            removes the effect of parameters being measured in different units.
            This is the number that says whether a system is genuinely
            ill-posed.
        null_space: Columns spanning the unconstrained directions, shape
            `(n, n - rank)`.
    """

    inverse: np.ndarray
    eigenvalues: np.ndarray
    rank: int
    condition_number: float
    scaled_condition_number: float
    null_space: np.ndarray

    @property
    def rank_deficient(self) -> bool:
        """Whether any direction was cut."""
        return self.rank < self.eigenvalues.size


def _symmetrise(matrix: np.ndarray) -> np.ndarray:
    array = np.asarray(matrix, dtype=float)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValidationError(f"expected a square matrix, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise DegenerateSystemError(
            "the normal equations contain non-finite entries; the fit diverged"
        )
    return 0.5 * (array + array.T)


def jacobi_scale(matrix: np.ndarray) -> np.ndarray:
    """Scale a symmetric matrix to unit diagonal.

    Args:
        matrix: A symmetric matrix.

    Returns:
        `D @ matrix @ D` with `D = diag(1 / sqrt(diag(matrix)))`, leaving rows
        with a non-positive diagonal untouched.
    """
    symmetric = _symmetrise(matrix)
    diagonal = np.diag(symmetric).copy()
    scale = np.where(diagonal > 0, 1.0 / np.sqrt(np.maximum(diagonal, 1e-300)), 1.0)
    return symmetric * np.outer(scale, scale)


def condition_number(matrix: np.ndarray, scaled: bool = False) -> float:
    """Condition number of a symmetric matrix.

    Args:
        matrix: A symmetric matrix.
        scaled: Apply Jacobi scaling first, which makes the number comparable
            across problems whose parameters carry different units.

    Returns:
        Largest eigenvalue divided by smallest, or `inf` if the smallest is not
        positive.
    """
    target = jacobi_scale(matrix) if scaled else _symmetrise(matrix)
    eigenvalues = np.linalg.eigvalsh(target)
    smallest, largest = float(eigenvalues[0]), float(eigenvalues[-1])
    if smallest <= 0 or largest <= 0:
        return float("inf")
    return largest / smallest


def invert_symmetric(
    matrix: np.ndarray, rcond: float = DEFAULT_RCOND
) -> SymmetricInverse:
    """Invert a symmetric positive semi-definite matrix, reporting its spectrum.

    Args:
        matrix: The matrix to invert, symmetrised on the way in.
        rcond: Eigenvalues below `rcond * max(eigenvalue)` are treated as zero
            and their directions moved into the null space.

    Returns:
        The inverse together with everything needed to judge whether to trust it.

    Raises:
        DegenerateSystemError: Every eigenvalue was cut, so the matrix carries
            no information at all.
    """
    symmetric = _symmetrise(matrix)
    eigenvalues, vectors = np.linalg.eigh(symmetric)
    largest = float(eigenvalues[-1]) if eigenvalues.size else 0.0
    if largest <= 0:
        raise DegenerateSystemError(
            "the normal equations are not positive definite in any direction"
        )
    cut = largest * rcond
    keep = eigenvalues > cut
    if not np.any(keep):
        raise DegenerateSystemError(
            f"every eigenvalue is below the cut of {cut:.3e}; nothing is constrained"
        )
    inverse = (vectors[:, keep] / eigenvalues[keep]) @ vectors[:, keep].T
    smallest_kept = float(eigenvalues[keep][0])
    return SymmetricInverse(
        inverse=inverse,
        eigenvalues=eigenvalues,
        rank=int(np.count_nonzero(keep)),
        condition_number=(
            largest / smallest_kept if np.all(keep) else float("inf")
        ),
        scaled_condition_number=condition_number(symmetric, scaled=True),
        null_space=np.ascontiguousarray(vectors[:, ~keep]),
    )


def correlation_from_covariance(covariance: np.ndarray) -> np.ndarray:
    """Convert a covariance matrix to a correlation matrix.

    Args:
        covariance: A symmetric covariance matrix.

    Returns:
        The correlation matrix. Rows with zero variance — a parameter that was
        held fixed — come back as zero off the diagonal and one on it, rather
        than as `nan`.
    """
    symmetric = _symmetrise(covariance)
    deviations = np.sqrt(np.clip(np.diag(symmetric), 0.0, None))
    safe = np.where(deviations > 0, deviations, 1.0)
    correlation = symmetric / np.outer(safe, safe)
    zero = deviations <= 0
    if np.any(zero):
        correlation[zero, :] = 0.0
        correlation[:, zero] = 0.0
    np.fill_diagonal(correlation, 1.0)
    return np.clip(correlation, -1.0, 1.0)


def top_correlations(
    correlation: np.ndarray, names: Sequence[str], count: int = 10, threshold: float = 0.0
) -> List[Tuple[str, str, float]]:
    """The most strongly correlated parameter pairs.

    This is what turns a covariance matrix into the sentence an engineer can
    act on: "focal length and distance are correlated at 0.94".

    Args:
        correlation: A correlation matrix.
        names: Parameter names, one per row.
        count: How many pairs to return.
        threshold: Ignore pairs whose absolute correlation is below this.

    Returns:
        Up to `count` triples of `(name_a, name_b, correlation)`, strongest
        absolute correlation first.
    """
    matrix = np.asarray(correlation, dtype=float)
    if matrix.shape[0] != len(names):
        raise ValidationError(
            f"{len(names)} names for a {matrix.shape[0]}x{matrix.shape[1]} matrix"
        )
    rows, columns = np.triu_indices(matrix.shape[0], k=1)
    values = matrix[rows, columns]
    keep = np.abs(values) >= threshold
    rows, columns, values = rows[keep], columns[keep], values[keep]
    order = np.argsort(-np.abs(values))[:count]
    return [(names[rows[i]], names[columns[i]], float(values[i])) for i in order]


def dominant_terms(
    vector: np.ndarray, names: Sequence[str], count: int = 3
) -> List[Tuple[str, float]]:
    """The parameters that dominate a direction in parameter space.

    Used to describe a null-space direction in words rather than as a column of
    numbers.

    Args:
        vector: A direction, one entry per parameter.
        names: Parameter names.
        count: How many terms to return.

    Returns:
        Up to `count` pairs of `(name, weight)`, largest absolute weight first.
    """
    values = np.asarray(vector, dtype=float).reshape(-1)
    if values.size != len(names):
        raise ValidationError(f"{len(names)} names for a length-{values.size} vector")
    order = np.argsort(-np.abs(values))[:count]
    return [(names[i], float(values[i])) for i in order]


def block_inverse_stack(
    blocks: np.ndarray, rcond: float = DEFAULT_RCOND
) -> Tuple[np.ndarray, np.ndarray]:
    """Invert a stack of small symmetric matrices.

    Args:
        blocks: An `(n, k, k)` array of symmetric matrices.
        rcond: Relative eigenvalue cut, applied per block.

    Returns:
        The stacked inverses and a boolean array flagging rank-deficient blocks.
    """
    stack = np.asarray(blocks, dtype=float)
    if stack.ndim != 3 or stack.shape[1] != stack.shape[2]:
        raise ValidationError(f"expected an (n, k, k) stack, got shape {stack.shape}")
    symmetric = 0.5 * (stack + np.transpose(stack, (0, 2, 1)))
    eigenvalues, vectors = np.linalg.eigh(symmetric)
    largest = np.maximum(eigenvalues[:, -1:], 0.0)
    keep = eigenvalues > largest * rcond
    reciprocal = np.where(keep, 1.0 / np.where(keep, eigenvalues, 1.0), 0.0)
    inverses = np.einsum("nik,nk,njk->nij", vectors, reciprocal, vectors)
    return inverses, ~np.all(keep, axis=1)
