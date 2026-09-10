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

"""Can the noise model behind every interval in this report be believed?

Every standard deviation calibsense prints rests on `sigma^2 = cost / (m - p)`,
which estimates the noise *scale* from the data but asserts its *shape*: one
variance per coordinate, the same in x and y, uncorrelated between points. Real
corner noise is not shaped like that, and until this diagnostic existed nothing
in the tool checked whether the difference mattered.

It usually does not. Measured against Monte Carlo on synthetic rigs, anisotropic
noise elongated along the edge direction, noise that is several times worse in
some views than others, and a few per cent of badly mis-detected corners all
leave the classical covariance within about fifteen per cent of the truth. A few
hundred corners at varied orientations average those away, and `cost / dof`
picks up whatever average power is left.

The exception is spatial correlation across the frame, and it is severe. Noise
that varies smoothly over the image is partly absorbable by the pose and
distortion parameters, so it *lowers* the residual while *raising* the
estimator's real spread. With a 200 px correlation length the reported interval
came out nearly eight times too tight on a capture whose RMS looked better than
the honest one. Great RMS, wrong answer, no warning — which is the exact failure
this whole tool exists to catch.

So this diagnostic compares the classical deviation against
`refit.covariance.RobustCovariance`, which assumes only that different views are
independent. Agreement means the noise model held *for this capture*, and the
intervals elsewhere in the report are defensible by measurement. Disagreement
means they are understating the uncertainty, by roughly the ratio reported here.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from ..refit.covariance import MIN_CLUSTERS_FOR_ROBUST
from .base import Diagnostic, DiagnosticContext, Finding, Severity

#: Robust-over-classical deviation ratio above which the noise model has failed
#: and every interval in the report is understating the uncertainty.
#:
#: Calibrated against Monte Carlo rather than chosen: with independent noise the
#: sandwich lands between about 0.65 and 1.10 of the classical deviation, and its
#: own sampling spread is 21 to 45 per cent at fourteen views. A ratio of 1.4 is
#: therefore about where the signal leaves the estimator's own noise. Correlated
#: noise that mattered produced 2.3 and 5.8, so the critical cut sits at 2.0 with
#: room on both sides. Both cuts were measured at fourteen to thirty views and
#: are conservative above that, since the sandwich tightens as views are added.
INFLATION_CRITICAL = 2.0
INFLATION_WARNING = 1.4


class NoiseModelValidity(Diagnostic):
    """Check the assumed corner-noise shape against the scatter of the views."""

    cause: ClassVar[str] = "noise_model"
    title: ClassVar[str] = "Noise model"

    def run(self, context: DiagnosticContext) -> Finding:
        """Compare the classical and view-clustered intrinsic deviations."""
        covariance = context.fit.covariance
        robust = covariance.robust
        if robust is None:
            return self._finding(
                Severity.NOTE,
                "the noise model was not checked, because this fit carries no "
                "per-view scores; it was restored from a bundle written before "
                "calibsense stored them",
                "Re-run the refit from the detections to get the check.",
                measured=False,
            )
        if not robust.usable:
            return self._finding(
                Severity.NOTE,
                f"the noise model could not be checked: a view-clustered "
                f"covariance needs at least {MIN_CLUSTERS_FOR_ROBUST} views to "
                f"mean anything and this capture has {robust.n_clusters}",
                f"Capture at least {MIN_CLUSTERS_FOR_ROBUST} views. Until then "
                "every interval in this report is conditional on corner noise "
                "being independent between points, which nothing here has "
                "tested.",
                measured=False,
                n_clusters=robust.n_clusters,
                min_clusters=MIN_CLUSTERS_FOR_ROBUST,
            )
        if not context.fit.conditioning.identifiable:
            return self._finding(
                Severity.NOTE,
                "the noise model check is not meaningful on a capture that does "
                "not determine every parameter, because both estimates drop the "
                "same unconstrained directions and agree about a subspace "
                "rather than about the noise",
                "Fix the identifiability finding first; this check becomes "
                "informative once every parameter is constrained.",
                measured=False,
                n_clusters=robust.n_clusters,
            )
        if not context.fit.at_optimum:
            return self._finding(
                Severity.NOTE,
                "the noise model check is not meaningful away from an optimum: "
                "the per-view scores sum to zero only at a minimum, and here "
                "they carry a gradient that would inflate the comparison on its "
                "own",
                "Refit these detections rather than instrumenting the "
                "parameters in place, then read this check again.",
                measured=False,
                n_clusters=robust.n_clusters,
            )

        ratios = covariance.robust_inflation()
        worst = covariance.worst_robust_inflation()
        names = covariance.intrinsic_names
        finite = np.isfinite(ratios)
        index = int(np.nanargmax(np.where(finite, ratios, -np.inf)))
        metrics = dict(
            measured=True,
            worst_inflation=float(worst),
            worst_parameter=names[index],
            n_clusters=robust.n_clusters,
            degrees_of_freedom=robust.degrees_of_freedom,
            inflation={
                name: (float(r) if np.isfinite(r) else None)
                for name, r in zip(names, ratios)
            },
            # Named, because `inflation` is a dict and these are positional:
            # a JSON consumer that zipped them against the dict's key order
            # would silently pair the wrong parameter with the wrong deviation.
            names=list(names),
            classical_std=covariance.intrinsic_std().tolist(),
            robust_std=robust.std().tolist(),
        )
        shared = (
            f"the view-clustered deviation is {worst:.2f}x the classical one at "
            f"worst ({names[index]}), over {robust.n_clusters} independent views"
        )
        if worst > INFLATION_CRITICAL:
            return self._finding(
                Severity.CRITICAL,
                f"{shared}; the reported intervals assume corner noise is "
                f"uncorrelated between points and this capture's noise is not, "
                f"so every uncertainty in this report is roughly {worst:.1f}x "
                f"too tight — including the task-space millimetres, which are "
                f"propagated from the classical covariance",
                "Something is correlating the corner errors across each frame. "
                "The usual causes are field-varying defocus, an illumination or "
                "vignetting gradient pulling on the sub-pixel refinement, a "
                "board that is not flat, motion blur, or a rolling shutter. "
                "To tell them apart, measure the noise directly rather than "
                "inferring it: capture thirty frames without touching the "
                "camera or the target and run `calibsense noise-floor`, which "
                "reports the correlation length and separates a genuinely "
                "correlated field from a mount that simply drifted. Until the "
                "cause is found, multiply every interval in this report by "
                f"{worst:.1f}. Note that the RMS will look *better* under this "
                "fault, not worse, so it cannot be used to check the fix.",
                **metrics,
            )
        if worst > INFLATION_WARNING:
            return self._finding(
                Severity.WARNING,
                f"{shared}, which is more than the estimator's own scatter at "
                f"this view count explains",
                "Mild correlation in the corner errors, or simply few views. "
                "Add views: the check sharpens as they accumulate, and the "
                "ratio will either settle near one or grow, which tells you "
                "which it was.",
                **metrics,
            )
        return self._finding(
            Severity.OK,
            f"{shared}, so the assumed noise shape holds for this capture and "
            "the intervals elsewhere in this report are not resting on it "
            "untested",
            **metrics,
        )
