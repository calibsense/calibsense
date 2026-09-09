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

"""Rendering an audit as a PDF and as JSON.

The PDF is for a person who has to sign something. It leads with the task-space
statement, because that is the only sentence in the document a quality manager
can act on, and it ends with a question and a contact, because a report that
implies it has settled everything is worse than one that names what it has not.

The JSON carries every number the PDF shows and many it does not, so a CI step
or a dashboard never has to parse prose.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..diagnose.base import Severity
from ..errors import SerializationError
from .audit import Audit
from .pdf import Document, text_width

#: Severity colours, muted enough to print on a mono laser without smearing.
_COLOURS = {
    Severity.CRITICAL: (0.72, 0.18, 0.14),
    Severity.WARNING: (0.78, 0.50, 0.10),
    Severity.NOTE: (0.35, 0.35, 0.38),
    Severity.OK: (0.20, 0.45, 0.28),
}
_GREY = (0.42, 0.42, 0.45)
_BAR_CALIBRATION = (0.20, 0.35, 0.60)
_BAR_NOISE = (0.62, 0.70, 0.80)


def _heading(document: Document, text: str) -> None:
    document.space(8)
    document.ensure(34)
    document.text(text.upper(), 9.0, "bold", colour=_GREY)
    document.rule(0.6, (0.8, 0.8, 0.82))
    document.space(3)


def _key_values(document: Document, rows: Sequence[Tuple[str, str]]) -> None:
    for key, value in rows:
        document.text(key, 8.5, "bold", advance=False)
        document.text(value, 8.5, x=document.margin + 130.0)


def render_pdf(audit: Audit) -> bytes:
    """Render an audit as a PDF.

    Args:
        audit: The audit to render.

    Returns:
        The complete PDF file as bytes.
    """
    metadata = audit.metadata
    document = Document(title=metadata.title)

    document.text(metadata.title, 17.0, "bold")
    document.space(2)
    subtitle = f"{metadata.camera_name} - {metadata.created}"
    if metadata.prepared_by:
        subtitle += f" - prepared by {metadata.prepared_by}"
    document.text(subtitle, 8.5, colour=_GREY)
    document.rule(1.0, (0.3, 0.3, 0.32))
    document.space(8)

    if not audit.trustworthy:
        document.bar(document.margin, document.content_width, 16.0, (0.96, 0.90, 0.89))
        document.text(
            "  THIS CALIBRATION DOES NOT DETERMINE EVERY PARAMETER - "
            "EVERY FIGURE BELOW IS A LOWER BOUND",
            8.5, "bold", colour=_COLOURS[Severity.CRITICAL],
        )
        document.space(8)

    _heading(document, "What this camera can measure")
    for sentence in audit.headline():
        document.paragraph(sentence, 10.0)
        document.space(4)

    task = audit.headline_task
    if task is not None:
        _heading(document, "Task-space error")
        rows = []
        for quantity in task.quantities:
            distribution = task.distribution(quantity.name)
            low, high = distribution.interval()
            rows.append([
                quantity.name,
                f"{distribution.nominal:.4g}",
                f"{distribution.expected_error:.3g}",
                f"{low:+.3g}",
                f"{high:+.3g}",
                f"{distribution.bias:+.3g}",
            ])
        document.table(
            ["quantity", "nominal", "expected error", "95% low", "95% high", "bias"],
            rows, [110, 80, 90, 70, 70, 62],
        )
        document.space(6)
        document.text("where the error comes from", 8.5, "bold")
        document.space(2)
        for quantity in task.quantities:
            calibration, noise = task.variance_share(quantity.name)
            width = document.content_width - 190.0
            document.text(quantity.name, 8.0, advance=False)
            document.bar(document.margin + 110.0, width * calibration, 7.0, _BAR_CALIBRATION)
            document.bar(
                document.margin + 110.0 + width * calibration,
                width * noise, 7.0, _BAR_NOISE,
            )
            document.text(
                f"{calibration:.0%} / {noise:.0%}", 8.0, align="right", advance=False
            )
            document.space(11)
        document.paragraph(
            f"Dark is {task.parameter_source_label}; light is "
            f"{task.observation_noise_px:.3g} px of pixel noise at measurement "
            "time. Only the dark part can be improved by re-calibrating.",
            7.5, colour=_GREY,
        )

    _heading(document, "The calibration")
    fit = audit.fit
    rows = [
        ("Model", f"{fit.camera.kind}, {fit.camera.distortion.size} distortion terms"),
        ("Views / points", f"{fit.n_views} / {fit.residuals.total_points}"),
        ("In-sample RMS", f"{fit.rms:.4f} px"),
        ("Residual sigma", f"{fit.covariance.sigma:.4f} px per coordinate"),
        ("Identifiable", "yes" if fit.conditioning.identifiable else "NO"),
        (
            "Scaled condition number",
            f"{fit.conditioning.scaled_condition_number:.3e} "
            f"(rank {fit.conditioning.rank}/{fit.conditioning.n_intrinsic})",
        ),
    ]
    if audit.validation is not None:
        rows.append(
            (
                "Out-of-sample RMS",
                f"{audit.validation.out_of_sample_rms:.4f} px over "
                f"{audit.validation.n_folds} folds, ratio "
                f"{audit.validation.ratio:.2f}x",
            )
        )
    _key_values(document, rows)
    document.space(6)
    document.table(
        ["parameter", "value", "std dev", "relative", "weak"],
        [
            [
                name, f"{value:.6g}", f"{deviation:.4g}",
                f"{abs(deviation / value):.2%}" if value else "-",
                f"{participation:.2f}",
            ]
            for name, value, deviation, participation in fit.parameter_table()
        ],
        [110, 110, 90, 80, 62],
    )
    if not fit.conditioning.identifiable:
        document.space(3)
        document.paragraph(
            "A weak value near 1 means that parameter lies in a direction the "
            "capture does not constrain. Its standard deviation is not "
            "meaningful, because a pseudo-inverse assigns zero variance to an "
            "unconstrained direction rather than infinite.",
            7.5, colour=_GREY,
        )

    _heading(document, "Findings")
    document.paragraph(audit.diagnosis.verdict(), 9.0)
    document.space(4)
    for finding in audit.diagnosis.ranked():
        if finding.severity < Severity.NOTE:
            continue
        document.ensure(46)
        document.text(
            f"[{finding.severity.label}] {finding.title}", 8.5, "bold",
            colour=_COLOURS[finding.severity],
        )
        document.paragraph(finding.summary, 8.0, indent=12.0)
        if finding.action:
            document.paragraph(finding.action, 8.0, indent=12.0, colour=_GREY)
        document.space(4)
    clean = audit.diagnosis.passing
    if clean:
        document.paragraph(
            "Clean: " + ", ".join(f.title for f in clean), 8.0, colour=_GREY
        )

    if audit.hand_eye is not None:
        _heading(document, "Hand-eye")
        for line in audit.hand_eye.summary_lines():
            document.text(line, 8.0, "mono")
        if audit.hand_eye_diagnosis is not None:
            document.space(4)
            for finding in audit.hand_eye_diagnosis.ranked():
                if finding.severity < Severity.WARNING:
                    continue
                document.text(
                    f"[{finding.severity.label}] {finding.title}", 8.5, "bold",
                    colour=_COLOURS[finding.severity],
                )
                document.paragraph(finding.summary, 8.0, indent=12.0)
                if finding.action:
                    document.paragraph(finding.action, 8.0, indent=12.0, colour=_GREY)

    if audit.prediction is not None:
        _heading(document, "What more views would buy")
        prediction = audit.prediction
        unit = task.quantities[0].unit if task is not None else "mm"
        document.paragraph(prediction.statement(unit) + ".", 9.0)
        document.space(3)
        _key_values(document, [
            ("Views", f"{prediction.n_views_before} -> {prediction.n_views_after}"),
            (
                "Interval",
                f"+/-{prediction.current_half_width:.3g} -> "
                f"+/-{prediction.forecast_half_width:.3g} {unit}",
            ),
            (
                "Identifiable",
                f"{'yes' if prediction.currently_identifiable else 'no'} -> "
                f"{'yes' if prediction.forecast_identifiable else 'no'}",
            ),
        ])
        document.space(3)
        document.paragraph(
            "This forecast predicts uncertainty, not correctness. The extra views "
            "are synthesised from the calibration already in hand, so they agree "
            "with it by construction: the prediction says how much tighter the "
            "interval would get, not that the answer would be right.",
            7.5, colour=_GREY,
        )

    _heading(document, "The open question")
    document.paragraph(metadata.open_question, 9.5)
    document.space(10)
    document.rule(0.6, (0.8, 0.8, 0.82))
    document.text(f"Questions: {metadata.contact}", 8.5, "bold")
    document.paragraph(
        f"Produced by caltrust {metadata.to_dict()['caltrust_version']}. Every "
        "figure is reproducible from the session bundle and the seed recorded in "
        "the JSON alongside it, to within the last few bits: OpenCV reduces "
        "across threads and floating-point addition is not associative. Run with "
        "--deterministic for byte-identical output.",
        7.5, colour=_GREY,
    )
    return document.render()


def write_pdf(audit: Audit, path: str) -> str:
    """Write an audit as a PDF file.

    Args:
        audit: The audit to render.
        path: Destination path.

    Returns:
        The path written.

    Raises:
        SerializationError: The file could not be written.
    """
    try:
        with open(path, "wb") as handle:
            handle.write(render_pdf(audit))
    except OSError as exc:
        raise SerializationError(f"could not write {path}: {exc}") from exc
    return path


def render_json(audit: Audit, indent: int = 2) -> str:
    """Render an audit as JSON text.

    Args:
        audit: The audit to serialise.
        indent: JSON indentation.

    Returns:
        The JSON document.

    Raises:
        SerializationError: A value in the audit is not JSON-serialisable.
    """
    def default(value: Any) -> Any:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(f"{type(value).__name__} is not JSON serialisable")

    try:
        return json.dumps(audit.to_dict(), indent=indent, default=default, sort_keys=True)
    except TypeError as exc:
        raise SerializationError(f"audit is not JSON-serialisable: {exc}") from exc


def write_json(audit: Audit, path: str, indent: int = 2) -> str:
    """Write an audit as a JSON file.

    Args:
        audit: The audit to serialise.
        path: Destination path.
        indent: JSON indentation.

    Returns:
        The path written.

    Raises:
        SerializationError: The file could not be written.
    """
    try:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(render_json(audit, indent))
            handle.write("\n")
    except OSError as exc:
        raise SerializationError(f"could not write {path}: {exc}") from exc
    return path


def render_text(audit: Audit) -> str:
    """Render an audit as plain text for a terminal.

    Args:
        audit: The audit to render.

    Returns:
        The report.
    """
    lines: List[str] = [audit.metadata.title, "=" * len(audit.metadata.title), ""]
    if not audit.trustworthy:
        lines += [
            "THIS CALIBRATION DOES NOT DETERMINE EVERY PARAMETER.",
            "Every figure below is a lower bound.",
            "",
        ]
    for sentence in audit.headline():
        lines += [sentence, ""]
    lines += ["findings", "-" * 8]
    lines += list(audit.diagnosis.summary_lines())
    if audit.hand_eye is not None:
        lines += ["", "hand-eye", "-" * 8]
        lines += list(audit.hand_eye.summary_lines())
        if audit.hand_eye_diagnosis is not None:
            lines += list(audit.hand_eye_diagnosis.summary_lines())
    lines += ["", "the open question", "-" * 17, audit.metadata.open_question, ""]
    lines.append(f"Questions: {audit.metadata.contact}")
    return "\n".join(lines) + "\n"
