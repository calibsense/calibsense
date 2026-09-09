"""M7 — the report.

Two outputs from one run. `run_audit` produces the numbers; `write_json` emits
every one of them for a machine, and `write_pdf` emits the subset a person has
to sign, leading with the task-space statement and ending with an open question.
"""

from __future__ import annotations

from .audit import (
    DEFAULT_CONTACT,
    DEFAULT_OPEN_QUESTION,
    Audit,
    ReportMetadata,
    default_tasks,
    run_audit,
)
from .forecast import Forecast, Recommendation, forecast, recommend
from .pdf import A4, Document, text_width, wrap
from .render import render_json, render_pdf, render_text, write_json, write_pdf

__all__ = [
    "A4",
    "DEFAULT_CONTACT",
    "DEFAULT_OPEN_QUESTION",
    "Audit",
    "Document",
    "Forecast",
    "Recommendation",
    "ReportMetadata",
    "default_tasks",
    "forecast",
    "recommend",
    "render_json",
    "render_pdf",
    "render_text",
    "run_audit",
    "text_width",
    "wrap",
    "write_json",
    "write_pdf",
]
