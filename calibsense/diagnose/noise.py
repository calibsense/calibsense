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

#: Coefficients of the null floor for the robust-over-classical deviation ratio,
#: as `intercept + slope / sqrt(G - 1)` over `G` views.
#:
#: A fixed cut cannot work for this statistic, for two reasons that pull in the
#: same direction. The view-clustered estimate carries a leverage correction
#: that is deliberately conservative, so it sits about a fifth above the
#: classical deviation even when the noise model is perfectly satisfied; and the
#: statistic is a maximum over the free intrinsics of a ratio of variance
#: estimates, whose spread shrinks as views accumulate. Together they put the
#: 95th percentile of the null at 1.73 over ten views and 1.41 over sixty, so a
#: single number is either trigger-happy at one end or blind at the other. The
#: cut this replaced was 1.4, which the same simulation shows firing on more
#: than a tenth of clean captures at every view count up to forty-five.
#:
#: Fitted to the 95th percentile of 80 clean captures per view count at
#: G = 10, 14, 20, 30, 45 and 60, residuals within 0.08.
INFLATION_FLOOR_INTERCEPT = 1.29
INFLATION_FLOOR_SLOPE = 1.47

#: Multiples of that floor at which the finding fires. One means "at the 95th
#: percentile of captures whose noise model holds", so a warning is already
#: saying the ratio is past where clean data reaches. Measured against held-out
#: clean captures on two rigs it fires on 3 to 13 per cent of them, against the
#: 5 per cent the floor nominally allows.
#:
#: The critical multiple is set from the other side, by asking how far the
#: classical interval was actually out at each correlation length. Injecting
#: fields into a 24-view rig and comparing against the spread of repeated
#: refits:
#:
#:     length   excess   classical / truth
#:       20 px    1.74        0.60
#:       30 px    2.20        0.44
#:       45 px    2.14        0.33
#:       60 px    2.28        0.28
#:      200 px    8.94        0.15
#:
#: Two is where the classical interval crosses from mildly optimistic to more
#: than twice too tight. A 20 px length sits at the `cornerSubPix` window scale,
#: which is the one place neighbouring corners genuinely share image gradients,
#: so it is close to unavoidable on real captures and warns rather than
#: condemning. Note the excess plateaus between 30 and 60 px while the interval
#: keeps degrading, so it ranks severity rather than measuring it.
INFLATION_CRITICAL = 2.0
INFLATION_WARNING = 1.0


def inflation_floor(n_clusters: int) -> float:
    """The ratio a capture whose noise model holds stays under, 19 times in 20.

    Args:
        n_clusters: Views contributing a score.

    Returns:
        The 95th percentile of the null distribution at that view count.
    """
    return INFLATION_FLOOR_INTERCEPT + INFLATION_FLOOR_SLOPE / np.sqrt(
        max(n_clusters - 1, 1)
    )


def _constraints(options) -> str:
    """The parameter constraints this fit was run under, named for the reader.

    Args:
        options: The refit settings.

    Returns:
        A phrase naming them, or an empty string when the fit was unconstrained.
    """
    held = []
    if getattr(options, "tie_aspect", False):
        held.append("fy tied to fx")
    for name in getattr(options, "fixed", ()) or ():
        held.append(f"{name} held fixed")
    if not held:
        return ""
    if len(held) == 1:
        return held[0]
    return ", ".join(held[:-1]) + " and " + held[-1]


def _critical_action(held: str) -> str:
    """What to do about a large disagreement between the two covariances.

    The disagreement is the classical information-matrix test: it says the model
    and the data do not match, and it does not say which part of the model is
    wrong. Correlated corner noise is the usual cause and the one this check was
    built for, but a constraint the capture does not actually satisfy produces
    the same signature and is far cheaper to rule out — a tied aspect ratio on a
    camera whose pixels are not square measured 3.2x on otherwise clean data,
    which is well into critical territory. So when the fit was constrained, that
    goes first.

    Args:
        held: Constraints this fit was run under, from `_constraints`.

    Returns:
        The action text.
    """
    constraint_first = (
        f"This fit was run with {held}. Rule that out first, because it is free "
        "to test and produces exactly this signature when the constraint is not "
        "true of the camera: refit without it and see whether the ratio falls. "
        "A tied aspect ratio on a camera whose pixels are only half a per cent "
        "from square is enough to do this on otherwise clean data. If the ratio "
        "survives an unconstrained refit, it is the corner noise. "
        if held else ""
    )
    return (
        constraint_first
        + "Something is making the errors depart from the assumed model. The "
        "usual cause is correlation across each frame: field-varying defocus, "
        "an illumination or vignetting gradient pulling on the sub-pixel "
        "refinement, a board that is not flat, motion blur, or a rolling "
        "shutter. To tell those apart, measure the noise directly rather than "
        "inferring it: capture thirty frames without touching the camera or the "
        "target and run `calibsense noise-floor`, which reports the correlation "
        "length and separates a genuinely correlated field from a mount that "
        "simply drifted. Widening the interval repairs the *interval*, not the "
        "calibration: the parameters are no better than the fault left them, "
        "and an error shared by every view stays invisible to a between-view "
        "estimate. Note also that the RMS will look *better* under a correlated "
        "field, not worse, so it cannot be used to check the fix."
    )


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
        floor = inflation_floor(robust.n_clusters)
        excess = worst / floor
        metrics = dict(
            measured=True,
            worst_inflation=float(worst),
            inflation_floor=float(floor),
            inflation_excess=float(excess),
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
            constraints=_constraints(context.fit.options),
        )
        shared = (
            f"the view-clustered deviation is {worst:.2f}x the classical one at "
            f"worst ({names[index]}), against a {floor:.2f} floor from "
            f"{robust.n_clusters} independent views"
        )
        held = _constraints(context.fit.options)
        if excess > INFLATION_CRITICAL:
            return self._finding(
                Severity.CRITICAL,
                f"{shared}; the two estimates disagree by more than sampling "
                f"explains, which means the model this fit assumes does not "
                f"describe the data, so the task-space millimetres are "
                f"propagated from the view-clustered estimate instead and are "
                f"widened by up to {worst:.1f}x",
                _critical_action(held),
                **metrics,
            )
        if excess > INFLATION_WARNING:
            return self._finding(
                Severity.WARNING,
                f"{shared}, which is past where captures whose model holds "
                f"reach nineteen times in twenty at this view count",
                (
                    f"Mild disagreement, or simply few views. The task-space "
                    f"intervals already carry the widening, so nothing here "
                    f"needs applying by hand."
                    + (
                        f" This fit was run with {held}; refitting without that "
                        "constraint is the cheapest thing to rule out."
                        if held else ""
                    )
                    + " Add views: the check sharpens as they accumulate, and "
                    "the ratio will either settle near one or grow, which tells "
                    "you which it was."
                ),
                **metrics,
            )
        return self._finding(
            Severity.OK,
            f"{shared}, so the assumed noise shape holds for this capture and "
            "the intervals elsewhere in this report are not resting on it "
            "untested",
            **metrics,
        )
