"""M5 — task-space error propagation.

`propagate` turns a covariance into millimetres. Everything else here is either
a task or the machinery that samples parameter sets from the fit.
"""

from __future__ import annotations

from .base import (
    DEFAULT_LEVEL,
    MeasurementDistribution,
    Quantity,
    Task,
    TaskResult,
)
from .propagate import DEFAULT_SAMPLES, propagate, propagate_all
from .sampling import (
    CovarianceSampler,
    ParameterSample,
    factorise,
    joint_covariance,
)
from .tasks import (
    CameraToBase,
    LengthAtDepth,
    PlaneLocation,
    StereoTriangulation,
)

__all__ = [
    "DEFAULT_LEVEL",
    "DEFAULT_SAMPLES",
    "CameraToBase",
    "CovarianceSampler",
    "LengthAtDepth",
    "MeasurementDistribution",
    "ParameterSample",
    "PlaneLocation",
    "Quantity",
    "StereoTriangulation",
    "Task",
    "TaskResult",
    "factorise",
    "joint_covariance",
    "propagate",
    "propagate_all",
]
