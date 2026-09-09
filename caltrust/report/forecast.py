"""Predicting what more views would buy.

The last sentence of the target report is a forecast: "adding six views at
400 mm and 1200 mm would reduce the interval to approximately ±0.35 mm". It is
answerable because uncertainty depends on the *geometry* of a capture and on the
corner noise, not on the true parameter values — so synthesising the extra views
from the calibration already in hand, refitting, and re-propagating gives a
sound prediction of the interval.

What it does not predict is bias. The synthetic views are generated from the
fitted camera, so they agree with it by construction; if the current fit is
metrically wrong, the forecast says how much *tighter* the interval would get,
not that the answer would be right. Every forecast carries that caveat, and the
recommendation is always to fix identifiability first.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.observations import ObservationSet
from ..core.session import CalibrationSession
from ..diagnose.base import Severity
from ..diagnose.report import Diagnosis
from ..errors import CalTrustError, ValidationError
from ..refit.engine import instrument
from ..refit.result import InstrumentedFit
from ..synthetic import pose_for_view, synthesise
from ..task.base import Task, TaskResult
from ..task.propagate import propagate

#: Tilt used for views added to fix a pose-diversity problem, in degrees. Chosen
#: from the measured relationship in `examples/degeneracy_demo.py`, where 40
#: degrees put the focal-length deviation at its floor.
RECOMMENDED_TILT_DEG = 40.0

#: Views a recommendation suggests adding. Six is enough to change the
#: conditioning materially without being a re-shoot.
RECOMMENDED_VIEWS = 6


@dataclass(frozen=True)
class Recommendation:
    """A concrete change to the capture, and what it is expected to buy.

    Attributes:
        cause: The finding this addresses.
        description: What to do, in a sentence.
        n_views: Views to add.
        distances_mm: Working distances for the added views.
        tilt_deg: Board tilt for the added views.
        lateral_mm: Lateral wander for the added views, which drives coverage.
    """

    cause: str
    description: str
    n_views: int
    distances_mm: Tuple[float, ...]
    tilt_deg: float
    lateral_mm: float

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dictionary."""
        return {
            "cause": self.cause,
            "description": self.description,
            "n_views": self.n_views,
            "distances_mm": list(self.distances_mm),
            "tilt_deg": self.tilt_deg,
            "lateral_mm": self.lateral_mm,
        }


@dataclass(frozen=True)
class Forecast:
    """The predicted effect of a recommendation.

    Attributes:
        recommendation: What was tried.
        quantity: The task quantity the interval refers to.
        current_half_width: Today's 95% interval half-width.
        forecast_half_width: The predicted half-width after the change.
        currently_identifiable: Whether today's fit determines every parameter.
        forecast_identifiable: Whether the extended capture would.
        n_views_before: Views in the current capture.
        n_views_after: Views after the addition.
    """

    recommendation: Recommendation
    quantity: str
    current_half_width: float
    forecast_half_width: float
    currently_identifiable: bool
    forecast_identifiable: bool
    n_views_before: int
    n_views_after: int

    @property
    def improvement(self) -> float:
        """How many times narrower the interval becomes; one means no change."""
        if self.forecast_half_width <= 0:
            return float("inf")
        return self.current_half_width / self.forecast_half_width

    @property
    def worthwhile(self) -> bool:
        """Whether the change buys either identifiability or a real narrowing."""
        gained = self.forecast_identifiable and not self.currently_identifiable
        return gained or self.improvement > 1.2

    def statement(self, unit: str = "mm") -> str:
        """The forecast sentence for the report."""
        if not self.currently_identifiable and self.forecast_identifiable:
            return (
                f"{self.recommendation.description} would make the calibration "
                f"identifiable, bringing the interval to approximately "
                f"+/-{self.forecast_half_width:.3g} {unit} from a figure that is "
                "currently only a lower bound"
            )
        return (
            f"{self.recommendation.description} would reduce the interval to "
            f"approximately +/-{self.forecast_half_width:.3g} {unit}, "
            f"{self.improvement:.1f}x narrower than today"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain JSON-compatible dictionary."""
        return {
            "recommendation": self.recommendation.to_dict(),
            "quantity": self.quantity,
            "current_half_width": self.current_half_width,
            "forecast_half_width": self.forecast_half_width,
            "improvement": self.improvement,
            "worthwhile": self.worthwhile,
            "currently_identifiable": self.currently_identifiable,
            "forecast_identifiable": self.forecast_identifiable,
            "n_views_before": self.n_views_before,
            "n_views_after": self.n_views_after,
            "predicts_uncertainty_only": True,
        }


def recommend(
    diagnosis: Diagnosis, distances_mm: Sequence[float]
) -> Optional[Recommendation]:
    """Turn the worst finding into a concrete change to the capture.

    The order is the order the causes actually matter in, which is not the order
    they were listed in: tilt before depth, because depth variation alone does
    not make a focal length identifiable, and coverage last because it affects
    the distortion rather than the scale.

    Args:
        diagnosis: The findings from M4.
        distances_mm: Working distances in the current capture.

    Returns:
        A recommendation, or `None` when nothing needs changing.
    """
    nearest, furthest = float(np.min(distances_mm)), float(np.max(distances_mm))
    median = float(np.median(distances_mm))

    def severity(cause: str) -> Severity:
        try:
            return diagnosis.by_cause(cause).severity
        except CalTrustError:
            return Severity.OK

    if max(severity("pose_diversity"), severity("frontoparallel_dominance")) >= Severity.WARNING:
        return Recommendation(
            cause="pose_diversity",
            description=(
                f"Adding {RECOMMENDED_VIEWS} views tilted about "
                f"{RECOMMENDED_TILT_DEG:.0f} degrees in varied directions"
            ),
            n_views=RECOMMENDED_VIEWS,
            distances_mm=(median,),
            tilt_deg=RECOMMENDED_TILT_DEG,
            lateral_mm=0.12 * median,
        )
    if severity("depth_variation") >= Severity.WARNING:
        near, far = 0.6 * nearest, 1.5 * furthest
        return Recommendation(
            cause="depth_variation",
            description=(
                f"Adding {RECOMMENDED_VIEWS} views at {near:.0f} mm and {far:.0f} mm"
            ),
            n_views=RECOMMENDED_VIEWS,
            distances_mm=(near, far),
            tilt_deg=RECOMMENDED_TILT_DEG,
            lateral_mm=0.12 * median,
        )
    if severity("image_coverage") >= Severity.WARNING:
        return Recommendation(
            cause="image_coverage",
            description=(
                f"Adding {RECOMMENDED_VIEWS} views with the board pushed into the "
                "frame edges and corners"
            ),
            n_views=RECOMMENDED_VIEWS,
            distances_mm=(nearest, median),
            tilt_deg=RECOMMENDED_TILT_DEG,
            lateral_mm=0.45 * median,
        )
    return None


def extend(
    session: CalibrationSession,
    fit: InstrumentedFit,
    recommendation: Recommendation,
    seed: Optional[int] = 0,
) -> ObservationSet:
    """Append synthetic views to a capture, following a recommendation.

    The synthetic views are projected through the *fitted* camera with noise at
    the fit's own residual sigma, which is what makes the forecast a statement
    about geometry rather than about the parameter values.

    Args:
        session: The current session.
        fit: The current fit, used as the truth for the synthetic views.
        recommendation: The geometry to add.
        seed: Seed for the synthetic noise.

    Returns:
        The current views followed by the synthetic ones.

    Raises:
        ValidationError: No synthetic view landed in frame.
    """
    target = session.observations.target
    rng = np.random.default_rng(seed)
    poses = [
        pose_for_view(
            target,
            recommendation.distances_mm[index % len(recommendation.distances_mm)],
            tilt_rad=np.radians(rng.uniform(0.8, 1.0) * recommendation.tilt_deg),
            tilt_axis_rad=rng.uniform(0.0, 2 * np.pi),
            roll_rad=rng.uniform(-np.pi, np.pi),
            offset_mm=rng.uniform(-recommendation.lateral_mm, recommendation.lateral_mm, 2),
        )
        for index in range(recommendation.n_views)
    ]
    capture = synthesise(
        fit.camera, target, poses, session.image_size,
        noise_px=max(fit.covariance.sigma, 1e-6), seed=seed,
    )
    extra = tuple(
        view.__class__(
            view_id=f"forecast{index:03d}",
            point_ids=view.point_ids,
            image_points=view.image_points,
            metadata={"forecast": True},
        )
        for index, view in enumerate(capture.observations.views)
    )
    return ObservationSet(
        target=target,
        image_size=session.image_size,
        views=session.observations.views + extra,
        summary=session.observations.summary,
    )


def forecast(
    session: CalibrationSession,
    fit: InstrumentedFit,
    task: Task,
    current: TaskResult,
    recommendation: Recommendation,
    quantity: Optional[str] = None,
    n_samples: int = 800,
    seed: Optional[int] = 0,
) -> Forecast:
    """Predict a task's interval after following a recommendation.

    Args:
        session: The current session.
        fit: The current fit.
        task: The task whose interval is being forecast.
        current: The task's propagated result today.
        recommendation: The change to evaluate.
        quantity: Which quantity's interval to report; defaults to the first.
        n_samples: Monte Carlo samples for the forecast propagation.
        seed: Seed, so the forecast is reproducible.

    Returns:
        The predicted interval and whether the change is worth making.

    Raises:
        ValidationError: The extended capture could not be fitted.
    """
    name = quantity or task.quantities()[0].name
    extended = extend(session, fit, recommendation, seed)
    extended_session = CalibrationSession(
        observations=extended,
        prior=session.prior,
        metadata=session.metadata,
    )
    extended_fit = instrument(extended_session, fit.options)
    predicted = propagate(extended_fit, task, n_samples, seed=seed)
    return Forecast(
        recommendation=recommendation,
        quantity=name,
        current_half_width=current.distribution(name).half_width(),
        forecast_half_width=predicted.distribution(name).half_width(),
        currently_identifiable=fit.conditioning.identifiable,
        forecast_identifiable=extended_fit.conditioning.identifiable,
        n_views_before=session.observations.n_views,
        n_views_after=extended.n_views,
    )
