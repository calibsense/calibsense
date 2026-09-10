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

"""M7 — one run, everything in it.

`run_audit` is the single command the whole package builds up to: ingest is
already done, this refits with instrumentation, cross-validates, diagnoses,
propagates the tasks into millimetres, forecasts what more views would buy, and
optionally solves hand-eye. The result is one object that both report writers
render, so the JSON and the PDF can never disagree about a number.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .._version import __version__
from ..core.session import CalibrationSession
from ..diagnose.base import DiagnosticContext, Severity
from ..diagnose.report import Diagnosis, diagnose
from ..errors import CalibSenseError, ValidationError
from ..refit.result import InstrumentedFit, RefitOptions
from ..refit.engine import instrument
from ..task.base import Task, TaskResult
from ..task.propagate import DEFAULT_SAMPLES, propagate
from ..task.tasks import LengthAtDepth
from ..validate.crossval import cross_validate
from ..validate.result import CrossValidation
from .forecast import Forecast, Recommendation, forecast, recommend

#: What the report ends with when the caller supplies nothing. Both are meant to
#: be replaced; the defaults exist so the field is never silently blank.
DEFAULT_OPEN_QUESTION = (
    "Which of these measurements does your acceptance test actually depend on, "
    "and what tolerance does it need? Every number above is conditional on the "
    "task, and the task is the one thing this report cannot infer."
)
DEFAULT_CONTACT = "set --contact to the person who should answer questions about this report"

#: Findings that describe a symptom rather than a cause, so the headline never
#: blames them. Out-of-sample error is the thing being explained, not the
#: explanation.
SYMPTOM_CAUSES = ("out_of_sample_error",)


@dataclass(frozen=True)
class ReportMetadata:
    """Who the report is for and what it should say at the end.

    Attributes:
        title: Document title.
        camera_name: How the camera is known in the plant.
        prepared_by: Who ran the audit.
        contact: Where to send questions. Printed as the last line.
        open_question: The question the report puts back to the reader.
        created: ISO-8601 UTC timestamp.
    """

    title: str = "Camera calibration measurement audit"
    camera_name: str = "unnamed camera"
    prepared_by: str = ""
    contact: str = DEFAULT_CONTACT
    open_question: str = DEFAULT_OPEN_QUESTION
    created: str = ""

    def __post_init__(self) -> None:
        if not self.created:
            object.__setattr__(
                self, "created", datetime.now(timezone.utc).isoformat(timespec="seconds")
            )

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dictionary."""
        return {
            "title": self.title,
            "camera_name": self.camera_name,
            "prepared_by": self.prepared_by,
            "contact": self.contact,
            "open_question": self.open_question,
            "created": self.created,
            "calibsense_version": __version__,
        }


@dataclass(frozen=True)
class Audit:
    """Everything one audit run produced.

    Attributes:
        metadata: Report framing.
        session: The ingested capture.
        fit: The instrumented refit.
        validation: Out-of-sample results, when measured.
        diagnosis: The named-cause findings.
        tasks: Propagated task-space error, one per task.
        prediction: What more views would buy, when there is something to fix.
        hand_eye: The solved hand-eye transform, when robot poses were supplied.
        hand_eye_diagnosis: Pose-set sufficiency for that solve.
    """

    metadata: ReportMetadata
    session: CalibrationSession
    fit: InstrumentedFit
    diagnosis: Diagnosis
    tasks: Tuple[TaskResult, ...] = ()
    validation: Optional[CrossValidation] = None
    prediction: Optional[Forecast] = None
    hand_eye: Optional[Any] = None
    hand_eye_diagnosis: Optional[Diagnosis] = None

    @property
    def headline_task(self) -> Optional[TaskResult]:
        """The task the report leads with, which is the first one given."""
        return self.tasks[0] if self.tasks else None

    @property
    def severity(self) -> Severity:
        """The worst severity across every diagnosis in the audit."""
        worst = self.diagnosis.severity
        if self.hand_eye_diagnosis is not None:
            worst = max(worst, self.hand_eye_diagnosis.severity)
        return worst

    @property
    def trustworthy(self) -> bool:
        """Whether any number in this report can be read at face value.

        `False` when something makes every figure below a bound rather than an
        answer. `caveats` says which, and this is just "is that list empty".

        This flag is the one thing a machine consumer is likely to branch on, so
        it has to cover every such condition rather than the first one that was
        implemented. It used to mean identifiability alone.
        """
        return not self.caveats()

    def caveats(self) -> Tuple[str, ...]:
        """Why every figure in this report is a bound rather than an answer.

        Two conditions qualify, and they fail in the same direction for
        different reasons. An unidentifiable calibration leaves a parameter
        direction with no variance at all, so every interval derived from it is
        narrower than the truth. Correlated corner noise leaves the covariance
        itself measurably too tight, which the noise model finding quantifies.
        Both mean the same thing to a reader — the figures understate — so both
        belong in the banner the report opens with.

        Returns:
            One sentence per condition, worst first, or an empty tuple when
            every figure can be read as it stands.
        """
        # Kept to one terse line each, because these are set as a full-width
        # banner in the PDF and a second line there would be a wrapped
        # afterthought rather than a warning.
        reasons: List[str] = []
        if not self.fit.conditioning.identifiable:
            reasons.append(
                "This calibration does not determine every parameter - "
                "every figure below is a lower bound"
            )
        for finding in self.diagnosis.findings:
            if finding.cause != "noise_model":
                continue
            if finding.severity < Severity.CRITICAL:
                continue
            factor = finding.metrics.get("worst_inflation")
            scale = f" by about {factor:.1f}x" if factor else ""
            reasons.append(
                "The corner noise is correlated - every figure below is too "
                f"tight{scale}"
            )
        return tuple(reasons)

    def dominant_cause(self) -> Optional[Any]:
        """The worst finding that names a cause rather than a symptom.

        The out-of-sample finding is excluded on purpose. A poor ratio is what
        the report is explaining, so blaming it would be circular.
        """
        for finding in self.diagnosis.ranked():
            if finding.severity < Severity.WARNING:
                break
            if finding.cause not in SYMPTOM_CAUSES:
                return finding
        return None

    def headline(self) -> Tuple[str, ...]:
        """The paragraph the report leads with.

        Four sentences, in the order an engineer needs them: what the error is
        in task units, how much the reprojection number was flattering itself,
        what is causing it, and what to do about it.
        """
        sentences: List[str] = []
        task = self.headline_task
        if task is not None:
            quantity = task.quantities[0]
            distribution = task.distribution(quantity.name)
            sentences.append(distribution.statement().capitalize() + ".")
        if self.validation is not None:
            if not self.validation.trustworthy:
                sentences.append(
                    f"The in-sample reprojection error of {self.fit.rms:.4g} px cannot "
                    "be checked against a held-out figure here, because the folds "
                    "trained on a capture that does not determine every parameter and "
                    "a held-out view's own pose absorbs exactly those errors."
                )
            elif self.validation.ratio < 1.1:
                sentences.append(
                    f"The reported {self.fit.rms:.4g} px reprojection error is an "
                    f"honest error estimate: held-out views reprojected to "
                    f"{self.validation.out_of_sample_rms:.4g} px, a ratio of "
                    f"{self.validation.ratio:.2f}x."
                )
            else:
                sentences.append(
                    f"The reported {self.fit.rms:.4g} px reprojection error understates "
                    f"out-of-sample error by {self.validation.ratio:.2g}x."
                )
        cause = self.dominant_cause()
        if cause is not None:
            sentences.append(f"The dominant cause is {cause.title.lower()}: {cause.summary}.")
        if self.prediction is not None and self.prediction.worthwhile:
            unit = task.quantities[0].unit if task is not None else "mm"
            sentences.append(self.prediction.statement(unit) + ".")
        return tuple(sentences)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the whole audit to JSON-compatible data."""
        from ..cli import render as text_render

        observations = self.session.observations
        return {
            "metadata": self.metadata.to_dict(),
            "headline": list(self.headline()),
            "severity": self.severity.label,
            "trustworthy": self.trustworthy,
            "caveats": list(self.caveats()),
            "session": text_render.session_to_json(self.session),
            "fit": text_render.fit_to_json(self.fit),
            "diagnosis": self.diagnosis.to_dict(),
            "tasks": [result.to_dict() for result in self.tasks],
            "forecast": self.prediction.to_dict() if self.prediction else None,
            "hand_eye": self.hand_eye.to_dict() if self.hand_eye else None,
            "hand_eye_diagnosis": (
                self.hand_eye_diagnosis.to_dict() if self.hand_eye_diagnosis else None
            ),
        }


def default_tasks(fit: InstrumentedFit, observations) -> Tuple[Task, ...]:
    """A sensible task when the caller names none.

    Uses the capture's own median working distance, because an error figure at a
    distance nobody works at is a number nobody needs.

    Args:
        fit: The instrumented refit.
        observations: The detections, for the target geometry.

    Returns:
        One task, measuring a length at the median working distance.
    """
    context = DiagnosticContext(fit, observations)
    depth = float(np.median(context.board_distances_mm))
    # Rounded, because a report that quotes an error "at 867.186 mm" invites the
    # reader to believe the distance was chosen rather than measured. Note that
    # on a fit that is not identifiable the depth is itself wrong by the same
    # factor the focal length is, which the report's own caveat covers.
    rounded = max(50.0, 50.0 * round(depth / 50.0))
    return (LengthAtDepth(depth_mm=rounded, length_mm=100.0),)


def run_audit(
    session: CalibrationSession,
    options: Optional[RefitOptions] = None,
    tasks: Optional[Sequence[Task]] = None,
    metadata: Optional[ReportMetadata] = None,
    folds: Optional[int] = None,
    n_samples: int = DEFAULT_SAMPLES,
    mounting: Optional[str] = None,
    seed: Optional[int] = 0,
    forecast_samples: int = 800,
    observation_noise_px: Optional[float] = None,
) -> Audit:
    """Run the whole audit: refit, cross-validate, diagnose, propagate, forecast.

    Args:
        session: The ingested capture.
        options: Refit settings.
        tasks: Measurements to propagate; a length at the median working
            distance when omitted.
        metadata: Report framing.
        folds: Cross-validation folds, or `None` for the default.
        n_samples: Monte Carlo samples per task.
        mounting: Solve hand-eye with this mounting when the session carries
            robot poses. `None` skips it.
        seed: Seed, so the whole report is reproducible.
        forecast_samples: Monte Carlo samples for the forecast, which is run at
            a lower count because it is a comparison rather than a claim.
        observation_noise_px: Pixel noise to propagate, overriding the sigma the
            fit infers from its own residuals. Pass
            `NoiseFloor.sigma_for_propagation()` from `calibsense noise-floor` to
            use a directly measured figure instead of one that assumes the
            model is right.

    Returns:
        The audit.

    Raises:
        RefitError: The calibration could not be estimated.
    """
    options = options or RefitOptions()
    fit = instrument(session, options)
    observations = session.observations

    validation: Optional[CrossValidation] = None
    try:
        validation = cross_validate(session, options, folds=folds, seed=seed)
        fit = replace(fit, cross_validation=validation)
    except CalibSenseError:
        # Too few views to hold any back. Not a failure of the audit; the
        # out-of-sample finding reports its own absence.
        validation = None

    diagnosis = diagnose(fit, observations, validation)
    chosen = tuple(tasks) if tasks else default_tasks(fit, observations)
    results = tuple(
        propagate(
            fit,
            task,
            n_samples,
            observation_noise_px=observation_noise_px,
            seed=None if seed is None else seed + 101 * index,
        )
        for index, task in enumerate(chosen)
    )

    prediction: Optional[Forecast] = None
    if results:
        context = DiagnosticContext(fit, observations)
        suggestion = recommend(diagnosis, context.board_distances_mm)
        if suggestion is not None:
            try:
                prediction = forecast(
                    session, fit, chosen[0], results[0], suggestion,
                    n_samples=forecast_samples, seed=seed,
                )
            except CalibSenseError:
                prediction = None

    hand_eye = None
    hand_eye_diagnosis = None
    if mounting is not None and session.robot is not None:
        from ..handeye import diagnose_hand_eye, solve_hand_eye

        try:
            hand_eye = solve_hand_eye(fit, session, mounting, seed=seed)
            hand_eye_diagnosis = diagnose_hand_eye(
                hand_eye,
                list(session.robot.aligned_with(observations)),
                list(fit.poses),
            )
        except CalibSenseError:
            hand_eye = None
            hand_eye_diagnosis = None

    return Audit(
        metadata=metadata or ReportMetadata(),
        session=session,
        fit=fit,
        diagnosis=diagnosis,
        tasks=results,
        validation=validation,
        prediction=prediction,
        hand_eye=hand_eye,
        hand_eye_diagnosis=hand_eye_diagnosis,
    )
