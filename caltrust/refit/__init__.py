"""M2 — refit with instrumentation.

`instrument` is the entry point. Everything else is exported because the pieces
are independently useful: the projector for its verified analytic Jacobians, the
normal-equation assembly for building a covariance at parameters caltrust did
not choose, and the linear algebra for its refusal to hide a rank deficiency.
"""

from __future__ import annotations

from .covariance import (
    CalibrationCovariance,
    WeakDirection,
    covariance_from_normal_equations,
)
from .engine import instrument
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
    "SymmetricInverse",
    "ViewResiduals",
    "WeakDirection",
    "assemble",
    "condition_number",
    "correlation_from_covariance",
    "covariance_from_normal_equations",
    "instrument",
    "invert_symmetric",
    "numerical_jacobian",
    "projector_for",
    "top_correlations",
]
