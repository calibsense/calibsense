"""caltrust — metric trust for camera calibration.

A reprojection RMS is an in-sample fit statistic. It says how well a model
explains the images used to fit it, and with a poorly conditioned set of poses
it can be driven arbitrarily low while the model itself is metrically wrong.
caltrust computes what that number cannot: the full parameter covariance, which
parameter combinations the capture does not determine at all, where the
residuals are structured rather than random, and how well posed the estimation
problem was.

Two steps, in order:

```python
from caltrust import session_from_images, instrument
from caltrust.cli.targets import resolve_target

session = session_from_images("captures/", resolve_target("checkerboard:9x6:25mm"))
fit = instrument(session)
print("\\n".join(fit.summary_lines()))
```

`caltrust.ingest` covers M1: images with a known target, or a calibration file
plus the detections behind it, for pinhole with Brown-Conrady and fisheye with
Kannala-Brandt, optionally with robot poses for hand-eye.

`caltrust.refit` covers M2: the same calibration re-run with everything the
standard call discards kept — full parameter covariance, per-view residual
distributions, per-corner residuals, the condition number of the normal
equations, and the parameter correlation matrix.
"""

from __future__ import annotations

from ._version import __version__
from .core.camera import (
    CameraModel,
    FisheyeKannalaBrandt,
    PinholeBrownConrady,
    camera_from_dict,
)
from .core.observations import ObservationSet, ViewObservations
from .core.poses import Pose
from .core.session import CalibrationRecord, CalibrationSession, RobotPoses
from .core.target import (
    CharucoBoard,
    Checkerboard,
    CircleGrid,
    TargetSpec,
    target_from_dict,
)
from .errors import (
    CalTrustError,
    DegenerateSystemError,
    DetectionError,
    IngestError,
    RefitError,
    SerializationError,
    UnsupportedFormatError,
    ValidationError,
)
from .ingest import (
    DetectorOptions,
    read_calibration,
    read_detections,
    read_robot_poses,
    session_from_calibration,
    session_from_images,
)
from .io import load_fit, load_session, save_fit, save_session
from .refit import (
    CalibrationCovariance,
    Conditioning,
    InstrumentedFit,
    RefitOptions,
    instrument,
)

__all__ = [
    "CalTrustError",
    "CalibrationCovariance",
    "CalibrationRecord",
    "CalibrationSession",
    "CameraModel",
    "CharucoBoard",
    "Checkerboard",
    "CircleGrid",
    "Conditioning",
    "DegenerateSystemError",
    "DetectionError",
    "DetectorOptions",
    "FisheyeKannalaBrandt",
    "IngestError",
    "InstrumentedFit",
    "ObservationSet",
    "PinholeBrownConrady",
    "Pose",
    "RefitError",
    "RefitOptions",
    "RobotPoses",
    "SerializationError",
    "TargetSpec",
    "UnsupportedFormatError",
    "ValidationError",
    "ViewObservations",
    "__version__",
    "camera_from_dict",
    "instrument",
    "load_fit",
    "load_session",
    "read_calibration",
    "read_detections",
    "read_robot_poses",
    "save_fit",
    "save_session",
    "session_from_calibration",
    "session_from_images",
    "target_from_dict",
]
