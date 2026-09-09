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

"""M4 — degeneracy and coverage diagnostics.

`diagnose` runs every check and returns a ranked set of findings, each naming
one cause and what to do about it. The thresholds are calibrated against
synthetic rigs with known truth rather than chosen by taste; see
`examples/degeneracy_demo.py` for the measurement they come from.
"""

from __future__ import annotations

from .base import (
    Diagnostic,
    DiagnosticContext,
    Finding,
    Severity,
    worst,
)
from .coverage import ImageCoverage, TargetScale, nearest_neighbour_spacing, occupancy
from .generalisation import OutOfSampleError
from .geometry import (
    DepthVariation,
    FrontoparallelDominance,
    PoseDiversity,
    normal_spread_degrees,
    orientation_tensor,
)
from .model import (
    ClusteredProfile,
    DistortionModelAdequacy,
    clustered_radial_profile,
    flatness_z,
    radial_trend,
)
from .noise import NoiseModelValidity
from .report import DIAGNOSTICS, Diagnosis, diagnose
from .views import OutlierViews, ViewInfluence, view_influences

__all__ = [
    "DIAGNOSTICS",
    "DepthVariation",
    "Diagnosis",
    "Diagnostic",
    "DiagnosticContext",
    "DistortionModelAdequacy",
    "Finding",
    "FrontoparallelDominance",
    "ImageCoverage",
    "NoiseModelValidity",
    "OutOfSampleError",
    "OutlierViews",
    "PoseDiversity",
    "Severity",
    "TargetScale",
    "ViewInfluence",
    "ClusteredProfile",
    "clustered_radial_profile",
    "diagnose",
    "flatness_z",
    "nearest_neighbour_spacing",
    "normal_spread_degrees",
    "occupancy",
    "orientation_tensor",
    "radial_trend",
    "view_influences",
    "worst",
]
