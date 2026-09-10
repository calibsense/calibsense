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

"""M2 — refit with instrumentation.

`instrument` is the entry point. Everything else is exported because the pieces
are independently useful: the projector for its verified analytic Jacobians, the
normal-equation assembly for building a covariance at parameters calibsense did
not choose, and the linear algebra for its refusal to hide a rank deficiency.
"""

from __future__ import annotations

from .cv_compat import set_single_threaded
from .covariance import (
    CalibrationCovariance,
    RobustCovariance,
    WeakDirection,
    covariance_from_normal_equations,
)
from .engine import instrument, parameter_block
from .linalg import (
    SymmetricInverse,
    condition_number,
    correlation_from_covariance,
    invert_symmetric,
    top_correlations,
)
from .normal import NormalEquations, assemble
from .projection import (
    FisheyeProjector,
    PinholeProjector,
    Projector,
    numerical_jacobian,
    projector_for,
)
from .residuals import RadialProfile, ResidualStatistics, ViewResiduals
from .result import Conditioning, InstrumentedFit, RefitOptions

__all__ = [
    "CalibrationCovariance",
    "Conditioning",
    "FisheyeProjector",
    "InstrumentedFit",
    "NormalEquations",
    "PinholeProjector",
    "Projector",
    "RadialProfile",
    "RefitOptions",
    "ResidualStatistics",
    "RobustCovariance",
    "SymmetricInverse",
    "ViewResiduals",
    "WeakDirection",
    "assemble",
    "condition_number",
    "correlation_from_covariance",
    "covariance_from_normal_equations",
    "instrument",
    "parameter_block",
    "invert_symmetric",
    "numerical_jacobian",
    "projector_for",
    "set_single_threaded",
    "top_correlations",
]
