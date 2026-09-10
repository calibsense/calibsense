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

`caltrust.validate` covers M3: K-fold cross-validation over views, which turns
the in-sample reprojection RMS into an honest out-of-sample number and a ratio
between them.

`caltrust.diagnose` covers M4: one diagnostic per cause — pose diversity, depth
variation, frontoparallel dominance, image coverage, target scale, distortion
model adequacy, per-view leverage and the noise model — each returning a finding
that names the cause and says what to do about it.

`caltrust.task` covers M5: the parameter covariance propagated by Monte Carlo
into the millimetres of the measurement actually being made — a length, a plane
fit, a stereo depth, a distance to a base frame — separating what calibration
contributes from what pixel noise does.

`caltrust.handeye` covers M6: the camera-to-robot transform with its own
covariance, and whether the pose set determines it at all.

`caltrust.report` covers M7: the whole audit as one paragraph in millimetres, a
JSON record, and a PDF, ending with what the audit could not settle.

`caltrust.noisefloor` measures corner noise directly from frames of a static
scene, with no model in the way. Everything else infers sigma from residuals,
which assumes the model is right; this does not, and it is the only place the
spatial correlation that invalidates the covariance can actually be seen.
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
from .noisefloor import (
    CornerNoise,
    CorrelationProfile,
    NoiseFloor,
    anisotropy_floor,
    measure_noise_floor,
)
from .diagnose import (
    Diagnosis,
    Finding,
    Severity,
    diagnose,
)
from .refit import (
    CalibrationCovariance,
    Conditioning,
    InstrumentedFit,
    RefitOptions,
    RobustCovariance,
    instrument,
)
from .validate import CrossValidation, Fold, HeldOutView, cross_validate

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
    "CornerNoise",
    "CorrelationProfile",
    "CrossValidation",
    "DegenerateSystemError",
    "Diagnosis",
    "DetectionError",
    "DetectorOptions",
    "Finding",
    "FisheyeKannalaBrandt",
    "Fold",
    "HeldOutView",
    "IngestError",
    "InstrumentedFit",
    "NoiseFloor",
    "ObservationSet",
    "PinholeBrownConrady",
    "Pose",
    "RefitError",
    "RefitOptions",
    "RobustCovariance",
    "RobotPoses",
    "SerializationError",
    "Severity",
    "TargetSpec",
    "UnsupportedFormatError",
    "ValidationError",
    "ViewObservations",
    "__version__",
    "anisotropy_floor",
    "camera_from_dict",
    "cross_validate",
    "diagnose",
    "instrument",
    "load_fit",
    "load_session",
    "measure_noise_floor",
    "read_calibration",
    "read_detections",
    "read_robot_poses",
    "save_fit",
    "save_session",
    "session_from_calibration",
    "session_from_images",
    "target_from_dict",
]
