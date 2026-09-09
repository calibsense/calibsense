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

"""M3 — out-of-sample error by cross-validation.

`cross_validate` is the entry point. It answers the one question a reprojection
RMS cannot: how much of that number is fit rather than accuracy.
"""

from __future__ import annotations

from .crossval import (
    DEFAULT_FOLDS,
    MIN_TRAIN_VIEWS,
    choose_folds,
    cross_validate,
    evaluate_held_out,
    fold_assignments,
)
from .result import CrossValidation, Fold, HeldOutView

__all__ = [
    "DEFAULT_FOLDS",
    "MIN_TRAIN_VIEWS",
    "CrossValidation",
    "Fold",
    "HeldOutView",
    "choose_folds",
    "cross_validate",
    "evaluate_held_out",
    "fold_assignments",
]
