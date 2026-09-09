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
