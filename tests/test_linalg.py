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

"""Symmetric inversion, conditioning and correlation."""

from __future__ import annotations

import numpy as np
import pytest

from caltrust.errors import DegenerateSystemError, ValidationError
from caltrust.refit.linalg import (
    block_inverse_stack,
    condition_number,
    correlation_from_covariance,
    dominant_terms,
    invert_symmetric,
    jacobi_scale,
    top_correlations,
)


def positive_definite(size, seed=0, ridge=1.0):
    rng = np.random.default_rng(seed)
    root = rng.normal(size=(size, size))
    return root @ root.T + np.eye(size) * ridge


def test_inverse_is_a_true_inverse_when_full_rank():
    matrix = positive_definite(6)
    result = invert_symmetric(matrix)
    assert result.rank == 6
    assert not result.rank_deficient
    assert np.allclose(result.inverse @ matrix, np.eye(6), atol=1e-10)


def test_inverse_symmetrises_a_slightly_asymmetric_input():
    matrix = positive_definite(4)
    skewed = matrix + np.triu(np.ones((4, 4)) * 1e-14, 1)
    assert np.allclose(invert_symmetric(skewed).inverse, invert_symmetric(matrix).inverse)


def test_a_rank_deficient_matrix_yields_a_pseudo_inverse_and_a_null_space():
    rng = np.random.default_rng(5)
    direction = rng.normal(size=6)
    direction /= np.linalg.norm(direction)
    projector = np.eye(6) - np.outer(direction, direction)
    matrix = projector @ positive_definite(6) @ projector
    result = invert_symmetric(matrix)
    assert result.rank == 5
    assert result.rank_deficient
    assert result.null_space.shape == (6, 1)
    assert abs(float(result.null_space[:, 0] @ direction)) == pytest.approx(1.0, abs=1e-6)
    assert result.condition_number == float("inf")
    # A pseudo-inverse annihilates the null direction rather than blowing up.
    assert np.allclose(result.inverse @ direction, 0.0, atol=1e-8)


def test_a_matrix_with_no_positive_eigenvalue_is_an_error():
    with pytest.raises(DegenerateSystemError, match="positive definite"):
        invert_symmetric(np.zeros((3, 3)))


def test_non_finite_entries_are_reported_as_a_diverged_fit():
    matrix = np.eye(3)
    matrix[0, 0] = np.nan
    with pytest.raises(DegenerateSystemError, match="non-finite"):
        invert_symmetric(matrix)


def test_non_square_input_is_rejected():
    with pytest.raises(ValidationError, match="square matrix"):
        invert_symmetric(np.zeros((2, 3)))


def test_jacobi_scaling_gives_a_unit_diagonal():
    scaled = jacobi_scale(positive_definite(5))
    assert np.allclose(np.diag(scaled), 1.0)


def test_jacobi_scaling_leaves_a_zero_diagonal_row_alone():
    matrix = np.diag([4.0, 0.0, 9.0])
    scaled = jacobi_scale(matrix)
    assert scaled[0, 0] == pytest.approx(1.0)
    assert scaled[1, 1] == 0.0


def test_scaled_condition_number_ignores_a_change_of_units():
    """The whole point: rescaling a parameter must not change the verdict."""
    matrix = positive_definite(5, seed=2)
    scale = np.diag([1.0, 1e4, 1e-3, 1.0, 1e2])
    rescaled = scale @ matrix @ scale
    assert condition_number(rescaled, scaled=True) == pytest.approx(
        condition_number(matrix, scaled=True), rel=1e-6
    )
    assert condition_number(rescaled) > condition_number(matrix) * 100


def test_condition_number_is_infinite_for_a_singular_matrix():
    assert condition_number(np.diag([1.0, 0.0])) == float("inf")


def test_correlation_has_a_unit_diagonal_and_matches_by_hand():
    covariance = np.array([[4.0, 1.9, 0.0], [1.9, 1.0, 0.0], [0.0, 0.0, 9.0]])
    correlation = correlation_from_covariance(covariance)
    assert np.allclose(np.diag(correlation), 1.0)
    assert correlation[0, 1] == pytest.approx(1.9 / (2.0 * 1.0))


def test_correlation_of_a_zero_variance_parameter_is_not_nan():
    covariance = np.diag([4.0, 0.0])
    correlation = correlation_from_covariance(covariance)
    assert np.all(np.isfinite(correlation))
    assert correlation[1, 1] == 1.0
    assert correlation[0, 1] == 0.0


def test_correlation_is_clipped_into_range():
    covariance = np.array([[1.0, 1.0 + 1e-12], [1.0 + 1e-12, 1.0]])
    assert correlation_from_covariance(covariance).max() <= 1.0


def test_top_correlations_are_ordered_by_absolute_value():
    correlation = np.array([
        [1.0, 0.2, -0.9],
        [0.2, 1.0, 0.5],
        [-0.9, 0.5, 1.0],
    ])
    pairs = top_correlations(correlation, ["a", "b", "c"], count=2)
    assert pairs[0] == ("a", "c", pytest.approx(-0.9))
    assert pairs[1] == ("b", "c", pytest.approx(0.5))


def test_top_correlations_honour_the_threshold():
    correlation = np.array([[1.0, 0.2], [0.2, 1.0]])
    assert top_correlations(correlation, ["a", "b"], threshold=0.5) == []


def test_top_correlations_never_report_a_diagonal_pair():
    correlation = np.eye(3)
    assert top_correlations(correlation, ["a", "b", "c"], count=5) == [
        ("a", "b", 0.0), ("a", "c", 0.0), ("b", "c", 0.0)
    ]


def test_top_correlations_checks_the_name_count():
    with pytest.raises(ValidationError, match="names for a"):
        top_correlations(np.eye(3), ["a", "b"])


def test_dominant_terms_ranks_by_absolute_weight():
    assert dominant_terms([0.1, -0.9, 0.3], ["a", "b", "c"], count=2) == [
        ("b", pytest.approx(-0.9)), ("c", pytest.approx(0.3))
    ]


def test_dominant_terms_checks_the_name_count():
    with pytest.raises(ValidationError, match="names for a"):
        dominant_terms([1.0, 2.0], ["a"])


def test_block_inverse_stack_inverts_each_block():
    blocks = np.stack([np.eye(3) * 2.0, positive_definite(3, seed=7)])
    inverses, singular = block_inverse_stack(blocks)
    assert not singular.any()
    assert np.allclose(inverses[0], np.eye(3) / 2.0)
    assert np.allclose(inverses[1] @ blocks[1], np.eye(3), atol=1e-10)


def test_block_inverse_stack_flags_a_singular_block():
    blocks = np.stack([np.eye(2), np.diag([1.0, 0.0])])
    inverses, singular = block_inverse_stack(blocks)
    assert singular.tolist() == [False, True]
    assert np.allclose(inverses[1], np.diag([1.0, 0.0]))


def test_block_inverse_stack_checks_the_shape():
    with pytest.raises(ValidationError, match=r"\(n, k, k\)"):
        block_inverse_stack(np.zeros((2, 3, 4)))
