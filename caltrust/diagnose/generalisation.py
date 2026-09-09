"""Does the fit predict views it never saw?

This turns M3's ratio into a finding so it sits in the same report as the causes.
It also carries M3's caveat: the ratio is blind to any degeneracy a held-out
view's own free pose can absorb, and the focal-length ambiguity is one of those.
A ratio of 1.0 from folds that were not identifiable says nothing.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from .base import Diagnostic, DiagnosticContext, Finding, Severity

#: Out-of-sample over in-sample ratio above which the fit is not generalising.
RATIO_CRITICAL = 2.0
RATIO_WARNING = 1.25


class OutOfSampleError(Diagnostic):
    """Compare held-out reprojection error against the in-sample number."""

    cause: ClassVar[str] = "out_of_sample_error"
    title: ClassVar[str] = "Out-of-sample error"

    def run(self, context: DiagnosticContext) -> Finding:
        """Report the cross-validated ratio, gated on per-fold identifiability."""
        validation = context.cross_validation
        if validation is None:
            return self._finding(
                Severity.NOTE,
                "out-of-sample error was not measured, so the reported RMS is an "
                "in-sample fit statistic with nothing to compare it against",
                "Run with cross-validation enabled. It costs a handful of extra "
                "calibrations and it is the single most useful number in the report.",
                measured=False,
            )
        metrics = dict(
            measured=True,
            in_sample_rms=validation.in_sample_rms,
            out_of_sample_rms=validation.out_of_sample_rms,
            ratio=validation.ratio,
            n_folds=validation.n_folds,
            held_out_points=validation.held_out_points,
            degenerate_folds=list(validation.degenerate_folds),
        )
        shared = (
            f"held-out RMS is {validation.out_of_sample_rms:.4f} px against "
            f"{validation.in_sample_rms:.4f} px in sample, a ratio of "
            f"{validation.ratio:.2f}x over {validation.n_folds} folds"
        )
        if validation.degenerate_folds:
            worst = self._largest_relative_spread(validation)
            return self._finding(
                Severity.CRITICAL,
                f"{shared}, but the ratio is uninformative: "
                f"{len(validation.degenerate_folds)} of {validation.n_folds} folds "
                "trained on a set that does not determine every intrinsic, and a "
                "held-out view's own pose absorbs exactly those errors"
                + (
                    f"; across folds {worst[0]} varied by {worst[1]:.4g}"
                    if worst
                    else ""
                ),
                "Fix the identifiability problem named elsewhere in this report. "
                "Until then, treat the ratio as a lower bound and read the "
                "across-fold parameter spread instead.",
                **metrics,
            )
        if validation.ratio > RATIO_CRITICAL:
            return self._finding(
                Severity.CRITICAL,
                f"{shared}; the in-sample error understates out-of-sample error by "
                f"{validation.ratio:.1f}x",
                "The model is fitting these images rather than this camera. Reduce "
                "--distortion-terms, or add views, and check the conditioning "
                "findings above.",
                **metrics,
            )
        if validation.ratio > RATIO_WARNING:
            return self._finding(
                Severity.WARNING,
                shared,
                "Some overfitting. Fewer distortion terms or more views would "
                "close the gap.",
                **metrics,
            )
        return self._finding(
            Severity.OK,
            shared + ", so the in-sample number is an honest error estimate",
            **metrics,
        )

    @staticmethod
    def _largest_relative_spread(validation):
        """The parameter whose fold spread is largest against its prediction."""
        best = None
        for name, fold, predicted in validation.spread_table():
            if not np.isfinite(predicted) or predicted <= 0:
                continue
            score = fold / predicted
            if best is None or score > best[2]:
                best = (name, fold, score)
        return (best[0], best[1]) if best else None
