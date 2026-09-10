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

"""M6 — hand-eye calibration with a covariance.

`solve_hand_eye` estimates the camera's transform onto the arm, either
flange-to-camera for a camera that rides the wrist or base-to-camera for one
bolted to the cell, and reports its uncertainty. `diagnose_hand_eye` says
whether the robot motions were varied enough to determine it.
"""

from __future__ import annotations

from .diagnose import HAND_EYE_DIAGNOSTICS, HandEyeContext, diagnose_hand_eye
from .result import MOUNTINGS, HandEyeResult
from .solve import (
    MIN_VIEWS,
    closed_form,
    relative_motions,
    resample_covariance,
    solve_hand_eye,
)

__all__ = [
    "HAND_EYE_DIAGNOSTICS",
    "MIN_VIEWS",
    "MOUNTINGS",
    "HandEyeContext",
    "HandEyeResult",
    "closed_form",
    "diagnose_hand_eye",
    "relative_motions",
    "resample_covariance",
    "solve_hand_eye",
]
