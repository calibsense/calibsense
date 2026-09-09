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

"""M7 — the PDF writer, the audit, and the two outputs."""

from __future__ import annotations

import json
import zlib

import numpy as np
import pytest

from caltrust.core.session import CalibrationSession
from caltrust.errors import SerializationError, ValidationError
from caltrust.refit import RefitOptions, instrument
from caltrust.report import (
    DEFAULT_CONTACT,
    DEFAULT_OPEN_QUESTION,
    A4,
    Audit,
    Document,
    ReportMetadata,
    default_tasks,
    forecast,
    recommend,
    render_json,
    render_pdf,
    render_text,
    run_audit,
    text_width,
    wrap,
    write_json,
    write_pdf,
)
from caltrust.report.pdf import FONTS, escape
from caltrust.task import LengthAtDepth, PlaneLocation, propagate

from . import rigs

#: The exact marker this writer emits before a content stream. Searching for a
#: bare `stream` would match compressed bytes and truncate the extraction.
_STREAM_START = b">>\nstream\n"


def pdf_streams(raw: bytes) -> list:
    """Every decompressed content stream in a rendered PDF, in page order."""
    streams = []
    index = 0
    while True:
        found = raw.find(_STREAM_START, index)
        if found < 0:
            break
        start = found + len(_STREAM_START)
        end = raw.index(b"\nendstream", start)
        streams.append(zlib.decompress(raw[start:end]))
        index = end
    return streams


def pdf_text(raw: bytes) -> bytes:
    """All page text from a rendered PDF, concatenated."""
    return b"".join(pdf_streams(raw))


# --------------------------------------------------------------------------
# the PDF writer
# --------------------------------------------------------------------------

def test_helvetica_widths_match_the_adobe_metrics():
    """Hand-checked against the AFM table; wrapping depends on these."""
    assert text_width("Hello", 10.0) == pytest.approx(22.78)
    assert text_width(" ", 10.0) == pytest.approx(2.78)
    assert text_width("W", 10.0) == pytest.approx(9.44)
    assert text_width("W", 10.0, "bold") == pytest.approx(9.44)
    assert text_width("i", 10.0) == pytest.approx(2.22)
    assert text_width("i", 10.0, "bold") == pytest.approx(2.78)


def test_every_font_has_a_full_ascii_width_table():
    for key, (_, widths) in FONTS.items():
        if widths is None:
            continue
        assert len(widths) == 95, key
        assert all(w > 0 for w in widths), key


def test_courier_is_monospaced():
    assert text_width("iiii", 10.0, "mono") == pytest.approx(
        text_width("WWWW", 10.0, "mono")
    )


def test_measuring_with_an_unknown_font_is_rejected():
    with pytest.raises(ValidationError, match="unknown font"):
        text_width("x", 10.0, "comic")


def test_wrapping_respects_the_column_width():
    lines = wrap("word " * 60, 200.0, 9.0)
    assert len(lines) > 1
    assert all(text_width(line, 9.0) <= 200.0 for line in lines)


def test_wrapping_never_drops_an_unbreakable_word():
    long_word = "x" * 200
    lines = wrap(long_word, 50.0, 9.0)
    assert "".join(lines) == long_word


def test_wrapping_empty_text_gives_one_empty_line():
    assert wrap("   ", 100.0, 9.0) == [""]


def test_escaping_uses_cp1252_so_an_em_dash_survives():
    """The fonts are declared WinAnsiEncoding, so Latin-1 would mangle these."""
    assert escape("a — b") == b"a \x97 b"
    assert escape("it’s") == b"it\x92s"
    assert escape("a 中 b") == b"a ? b"


def test_escaping_protects_pdf_syntax():
    assert escape("(a) \\ (b)") == b"\\(a\\) \\\\ \\(b\\)"


def test_a_document_renders_a_valid_pdf():
    document = Document(title="test")
    document.text("heading", 16.0, "bold")
    document.paragraph("body text " * 20)
    document.rule()
    document.table(["a", "b"], [["1", "2"]], [100.0, 100.0])
    document.bar(60.0, 100.0, 8.0)
    raw = document.render()
    assert raw.startswith(b"%PDF-1.4")
    assert raw.rstrip().endswith(b"%%EOF")
    assert b"/Type /Catalog" in raw
    assert b"/Type /Pages" in raw
    assert b"startxref" in raw


def test_page_streams_are_deflated_and_readable_back():
    document = Document()
    document.text("findable text")
    raw = document.render()
    assert b"/Filter /FlateDecode" in raw
    assert b"findable text" in pdf_text(raw)


def test_content_overflowing_a_page_breaks_onto_the_next():
    document = Document()
    for index in range(200):
        document.text(f"line {index}")
    assert len(document.pages) > 1
    assert b"/Type /Page " in document.render()


def test_the_page_size_and_margins_are_respected():
    document = Document(size=A4, margin=50.0)
    assert document.content_width == pytest.approx(A4[0] - 100.0)
    assert document.bottom == pytest.approx(A4[1] - 50.0)


def test_a_table_with_mismatched_columns_is_rejected():
    document = Document()
    with pytest.raises(ValidationError, match="column widths"):
        document.table(["a", "b"], [["1", "2"]], [100.0])
    with pytest.raises(ValidationError, match="against 2 headers"):
        document.table(["a", "b"], [["1"]], [100.0, 100.0])


def test_a_zero_width_bar_draws_nothing():
    document = Document()
    before = len(document.pages[0].operators)
    document.bar(10.0, 0.0, 8.0)
    document.bar(10.0, 10.0, 0.0)
    assert len(document.pages[0].operators) == before


def test_an_empty_table_draws_nothing():
    document = Document()
    before = len(document.pages[0].operators)
    document.table(["a"], [], [100.0])
    assert len(document.pages[0].operators) == before


# --------------------------------------------------------------------------
# the audit
# --------------------------------------------------------------------------

@pytest.fixture
def good_audit():
    capture = rigs.healthy()
    session = CalibrationSession(observations=capture.observations)
    return run_audit(session, n_samples=400, forecast_samples=300)


@pytest.fixture
def degenerate_audit():
    capture = rigs.frontoparallel()
    session = CalibrationSession(observations=capture.observations)
    return run_audit(session, n_samples=400, forecast_samples=300)


@pytest.fixture
def correlated_audit():
    """A capture whose intervals are too tight for a reason other than rank."""
    capture = rigs.with_correlated_noise()
    session = CalibrationSession(observations=capture.observations)
    return run_audit(session, n_samples=400, forecast_samples=300)


@pytest.mark.slow
def test_an_audit_gathers_every_milestone(good_audit):
    assert good_audit.fit is not None
    assert good_audit.validation is not None
    assert good_audit.diagnosis.findings
    assert good_audit.tasks
    assert good_audit.headline_task is good_audit.tasks[0]


@pytest.mark.slow
def test_the_default_task_uses_a_round_working_distance(good_audit):
    """A report that quotes "at 867.186 mm" implies the distance was chosen."""
    depth = good_audit.tasks[0].task.depth_mm
    assert depth % 50 == 0
    assert depth > 0


@pytest.mark.slow
def test_the_headline_states_the_error_in_task_units(good_audit):
    headline = good_audit.headline()
    assert headline
    assert "expected error" in headline[0]
    assert "mm" in headline[0]


@pytest.mark.slow
def test_the_headline_never_blames_a_symptom(degenerate_audit):
    """Out-of-sample error is what is being explained, not the explanation."""
    cause = degenerate_audit.dominant_cause()
    assert cause is not None
    assert cause.cause != "out_of_sample_error"


@pytest.mark.slow
def test_a_degenerate_audit_is_flagged_untrustworthy(degenerate_audit):
    assert not degenerate_audit.trustworthy
    joined = " ".join(degenerate_audit.headline())
    assert "lower bound" in joined
    assert "honest" not in joined


@pytest.mark.slow
def test_a_healthy_audit_calls_its_reprojection_error_honest(good_audit):
    assert good_audit.trustworthy
    assert good_audit.caveats() == ()
    assert any("honest error estimate" in s for s in good_audit.headline())


@pytest.mark.slow
def test_correlated_noise_makes_the_whole_audit_untrustworthy(correlated_audit):
    """`trustworthy` has to cover every way the figures become a bound.

    A reader who branches on this flag would otherwise ship a calibration whose
    intervals the report itself says are several times too tight, because the
    flag used to mean identifiability alone and this capture is identifiable.
    """
    assert correlated_audit.fit.conditioning.identifiable
    assert not correlated_audit.trustworthy
    caveats = correlated_audit.caveats()
    assert len(caveats) == 1
    assert "corner noise is correlated" in caveats[0]
    assert "too tight" in caveats[0]


@pytest.mark.slow
def test_the_caveats_reach_the_json_and_both_renderers(correlated_audit):
    import json

    payload = json.loads(render_json(correlated_audit))
    assert payload["trustworthy"] is False
    assert payload["caveats"] == list(correlated_audit.caveats())

    assert "CORNER NOISE IS CORRELATED" in render_text(correlated_audit)
    assert b"CORNER NOISE IS CORRELATED" in pdf_streams(render_pdf(correlated_audit))[0]


@pytest.mark.slow
def test_a_banner_line_fits_the_pdf_content_width(correlated_audit, degenerate_audit):
    """The banner is one line by design, so it must not silently overflow."""
    from caltrust.report.pdf import A4, text_width

    available = A4[0] - 2 * 56.0
    for audit in (correlated_audit, degenerate_audit):
        for caveat in audit.caveats():
            assert text_width("  " + caveat.upper(), 8.5, "bold") <= available, caveat


@pytest.mark.slow
def test_the_headline_ends_with_a_forecast_when_there_is_one(degenerate_audit):
    assert degenerate_audit.prediction is not None
    assert degenerate_audit.prediction.worthwhile
    assert any("would" in s for s in degenerate_audit.headline())


@pytest.mark.slow
def test_the_audit_severity_covers_the_hand_eye_diagnosis():
    session = rigs.hand_eye_session("eye_in_hand", roll_only=True)
    audit = run_audit(
        session, n_samples=200, forecast_samples=200, mounting="eye_in_hand"
    )
    assert audit.hand_eye is not None
    assert audit.hand_eye_diagnosis is not None
    assert audit.severity.name == "CRITICAL"


@pytest.mark.slow
def test_an_audit_runs_without_cross_validation_when_there_are_too_few_views():
    capture = rigs.context(pose_set=rigs.poses(n=4))
    session = CalibrationSession(observations=capture.observations)
    audit = run_audit(session, n_samples=200, forecast_samples=200)
    assert audit.validation is None
    finding = audit.diagnosis.by_cause("out_of_sample_error")
    assert finding.metrics["measured"] is False


@pytest.mark.slow
def test_explicit_tasks_are_honoured():
    capture = rigs.healthy()
    session = CalibrationSession(observations=capture.observations)
    audit = run_audit(
        session, tasks=[PlaneLocation(700.0, 30.0), LengthAtDepth(500.0, 40.0)],
        n_samples=200, forecast_samples=200,
    )
    assert [result.task.kind for result in audit.tasks] == [
        "plane_location", "length_at_depth"
    ]
    assert audit.headline_task.task.kind == "plane_location"


def test_default_tasks_needs_only_a_fit_and_observations():
    capture = rigs.healthy()
    tasks = default_tasks(capture.fit, capture.observations)
    assert len(tasks) == 1
    assert tasks[0].kind == "length_at_depth"


def test_report_metadata_fills_in_defaults():
    metadata = ReportMetadata()
    assert metadata.created.endswith("+00:00")
    assert metadata.contact == DEFAULT_CONTACT
    assert metadata.open_question == DEFAULT_OPEN_QUESTION
    assert "caltrust_version" in metadata.to_dict()


# --------------------------------------------------------------------------
# the forecast
# --------------------------------------------------------------------------

@pytest.mark.slow
def test_the_forecast_makes_a_degenerate_capture_identifiable(degenerate_audit):
    prediction = degenerate_audit.prediction
    assert prediction.recommendation.cause == "pose_diversity"
    assert not prediction.currently_identifiable
    assert prediction.forecast_identifiable
    assert prediction.n_views_after > prediction.n_views_before
    assert "identifiable" in prediction.statement()


@pytest.mark.slow
def test_the_forecast_narrows_the_interval_dramatically(degenerate_audit):
    prediction = degenerate_audit.prediction
    assert prediction.forecast_half_width < 0.05 * prediction.current_half_width


@pytest.mark.slow
def test_the_forecast_says_it_predicts_uncertainty_not_correctness(degenerate_audit):
    assert degenerate_audit.prediction.to_dict()["predicts_uncertainty_only"] is True


def test_recommendations_put_tilt_before_depth():
    """Measured: depth variation alone does not make a focal length knowable."""
    from caltrust.diagnose import diagnose

    capture = rigs.frontoparallel()
    diagnosis = diagnose(capture.fit, capture.observations)
    suggestion = recommend(diagnosis, capture.board_distances_mm)
    assert suggestion.cause == "pose_diversity"
    assert "tilt" in suggestion.description.lower()


def test_a_depth_problem_recommends_distances():
    from caltrust.diagnose import diagnose

    capture = rigs.single_depth()
    diagnosis = diagnose(capture.fit, capture.observations)
    suggestion = recommend(diagnosis, capture.board_distances_mm)
    assert suggestion.cause == "depth_variation"
    assert "mm" in suggestion.description


def test_no_recommendation_when_nothing_needs_fixing():
    from caltrust.diagnose.base import Finding, Severity
    from caltrust.diagnose.report import Diagnosis

    clean = Diagnosis((Finding("pose_diversity", "Pose diversity", Severity.OK, "fine"),))
    assert recommend(clean, [800.0, 900.0]) is None


@pytest.mark.slow
def test_the_forecast_reports_no_gain_on_an_already_good_capture():
    from caltrust.diagnose import diagnose
    from caltrust.report.forecast import forecast as run_forecast

    capture = rigs.healthy()
    session = CalibrationSession(observations=capture.observations)
    diagnosis = diagnose(capture.fit, capture.observations)
    suggestion = recommend(diagnosis, capture.board_distances_mm)
    if suggestion is None:
        pytest.skip("nothing to recommend on this capture")
    task = LengthAtDepth(800.0, 100.0)
    current = propagate(capture.fit, task, 300)
    prediction = run_forecast(
        session, capture.fit, task, current, suggestion, n_samples=300
    )
    assert prediction.improvement > 0.5
    assert prediction.forecast_identifiable


# --------------------------------------------------------------------------
# the two outputs
# --------------------------------------------------------------------------

@pytest.mark.slow
def test_the_pdf_renders_and_carries_the_headline(good_audit):
    raw = render_pdf(good_audit)
    assert raw.startswith(b"%PDF-1.4")
    text = pdf_text(raw)
    assert b"WHAT THIS CAMERA CAN MEASURE" in text
    assert b"THE OPEN QUESTION" in text
    assert b"expected error" in text


@pytest.mark.slow
def test_the_pdf_leads_with_the_warning_when_untrustworthy(degenerate_audit):
    raw = render_pdf(degenerate_audit)
    assert b"DOES NOT DETERMINE EVERY PARAMETER" in pdf_streams(raw)[0]


@pytest.mark.slow
def test_the_pdf_ends_with_the_contact(good_audit):
    import dataclasses

    audit = dataclasses.replace(
        good_audit,
        metadata=dataclasses.replace(good_audit.metadata, contact="me@example.com"),
    )
    assert b"me@example.com" in pdf_streams(render_pdf(audit))[-1]


@pytest.mark.slow
def test_the_json_carries_every_section(good_audit):
    payload = json.loads(render_json(good_audit))
    for key in ("metadata", "headline", "severity", "trustworthy", "session",
                "fit", "diagnosis", "tasks", "forecast"):
        assert key in payload
    assert payload["tasks"][0]["quantities"][0]["statement"]
    assert payload["fit"]["cross_validation"] is not None


@pytest.mark.slow
def test_the_json_and_the_pdf_agree_on_the_headline(good_audit):
    payload = json.loads(render_json(good_audit))
    assert payload["headline"] == list(good_audit.headline())


@pytest.mark.slow
def test_the_json_includes_hand_eye_when_solved():
    session = rigs.hand_eye_session("eye_in_hand")
    audit = run_audit(
        session, n_samples=200, forecast_samples=200, mounting="eye_in_hand"
    )
    payload = json.loads(render_json(audit))
    assert payload["hand_eye"]["mounting"] == "eye_in_hand"
    assert payload["hand_eye_diagnosis"]["findings"]


@pytest.mark.slow
def test_both_outputs_write_to_disk(good_audit, tmp_path):
    pdf = write_pdf(good_audit, str(tmp_path / "report.pdf"))
    js = write_json(good_audit, str(tmp_path / "report.json"))
    assert open(pdf, "rb").read(8) == b"%PDF-1.4"
    assert json.load(open(js))["severity"]


@pytest.mark.slow
def test_writing_to_an_unwritable_path_is_reported(good_audit, tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    for writer in (write_pdf, write_json):
        with pytest.raises(SerializationError, match="could not write"):
            writer(good_audit, str(blocker / "inner" / "out"))


@pytest.mark.slow
def test_the_text_report_covers_the_same_ground(good_audit):
    text = render_text(good_audit)
    assert good_audit.metadata.title in text
    assert "findings" in text
    assert "the open question" in text
    assert good_audit.metadata.contact in text


@pytest.mark.slow
def test_the_text_report_warns_first_when_untrustworthy(degenerate_audit):
    text = render_text(degenerate_audit)
    assert text.index("DOES NOT DETERMINE") < text.index("findings")
