"""M4 — the diagnostic framework: context, severity, aggregation."""

from __future__ import annotations

import numpy as np
import pytest

from caltrust.diagnose import (
    DIAGNOSTICS,
    Diagnosis,
    Diagnostic,
    DiagnosticContext,
    Finding,
    Severity,
    diagnose,
    worst,
)
from caltrust.errors import ValidationError

from . import rigs


def test_severity_is_ordered_so_the_worst_wins():
    assert Severity.OK < Severity.NOTE < Severity.WARNING < Severity.CRITICAL
    assert worst(()) is Severity.OK
    findings = (
        Finding("a", "A", Severity.NOTE, "note"),
        Finding("b", "B", Severity.CRITICAL, "bad"),
        Finding("c", "C", Severity.OK, "fine"),
    )
    assert worst(findings) is Severity.CRITICAL


def test_finding_serialises_its_severity_by_name():
    payload = Finding("a", "A", Severity.WARNING, "s", "do x", {"n": 1}).to_dict()
    assert payload["severity"] == "WARNING"
    assert payload["metrics"] == {"n": 1}


def test_every_diagnostic_has_a_unique_cause_and_a_title():
    causes = [cls.cause for cls in DIAGNOSTICS]
    assert len(set(causes)) == len(causes)
    assert all(cls.cause and cls.title for cls in DIAGNOSTICS)


def test_every_cause_from_the_plan_has_a_diagnostic():
    expected = {
        "pose_diversity",
        "depth_variation",
        "frontoparallel_dominance",
        "image_coverage",
        "target_scale",
        "distortion_model",
        "outlier_views",
    }
    assert expected <= {cls.cause for cls in DIAGNOSTICS}


def test_diagnose_returns_one_finding_per_diagnostic():
    context = rigs.healthy()
    diagnosis = diagnose(context.fit, context.observations)
    assert len(diagnosis.findings) == len(DIAGNOSTICS)
    assert [f.cause for f in diagnosis.findings] == [c.cause for c in DIAGNOSTICS]


def test_diagnose_rejects_a_fit_from_a_different_capture():
    a, b = rigs.healthy(), rigs.tiny_board()
    trimmed = b.observations.select(range(b.observations.n_views - 1))
    with pytest.raises(ValidationError, match="not the same capture"):
        diagnose(a.fit, trimmed)


def test_board_normals_are_unit_length():
    context = rigs.healthy()
    assert np.allclose(np.linalg.norm(context.board_normals, axis=1), 1.0)
    assert context.board_normals.shape == (context.n_views, 3)


def test_tilt_is_folded_into_zero_to_ninety():
    tilts = rigs.healthy().tilt_degrees
    assert np.all(tilts >= 0) and np.all(tilts <= 90)


def test_frontoparallel_rig_has_near_zero_tilt():
    assert rigs.frontoparallel().tilt_degrees.max() < 5.0


def test_board_distance_is_measured_to_the_centre_not_the_origin():
    """A 200 mm board's origin corner is over 100 mm from its centre."""
    context = rigs.context(pose_set=rigs.poses(distances_mm=(700.0,), n=6))
    centre_distances = context.board_distances_mm
    origin_distances = context.fit.working_distances_mm()
    assert np.abs(centre_distances - 700.0).max() < 60.0
    assert not np.allclose(centre_distances, origin_distances, atol=1.0)


def test_weak_parameters_is_empty_when_identifiable():
    assert rigs.healthy().weak_parameters == set()


def test_weak_parameters_names_the_focal_lengths_when_degenerate():
    assert rigs.frontoparallel().weak_parameters >= {"fx", "fy"}


def test_pooled_image_points_cover_every_view():
    context = rigs.healthy()
    assert context.all_image_points.shape[0] == context.observations.total_points


def test_diagnosis_partitions_findings_by_severity():
    findings = (
        Finding("a", "A", Severity.CRITICAL, "x"),
        Finding("b", "B", Severity.WARNING, "y"),
        Finding("c", "C", Severity.NOTE, "z"),
        Finding("d", "D", Severity.OK, "w"),
    )
    diagnosis = Diagnosis(findings)
    assert len(diagnosis.critical) == 1
    assert len(diagnosis.warnings) == 1
    assert len(diagnosis.notes) == 1
    assert len(diagnosis.passing) == 1
    assert diagnosis.severity is Severity.CRITICAL


def test_ranked_puts_the_worst_first():
    findings = (
        Finding("a", "A", Severity.OK, "x"),
        Finding("b", "B", Severity.CRITICAL, "y"),
        Finding("c", "C", Severity.WARNING, "z"),
    )
    assert [f.cause for f in Diagnosis(findings).ranked()] == ["b", "c", "a"]


def test_by_cause_finds_and_reports_a_miss():
    diagnosis = Diagnosis((Finding("a", "A", Severity.OK, "x"),))
    assert diagnosis.by_cause("a").title == "A"
    with pytest.raises(ValidationError, match="no finding for cause"):
        diagnosis.by_cause("nope")


def test_verdict_names_the_critical_findings_rather_than_condemning_everything():
    diagnosis = Diagnosis((Finding("a", "Image coverage", Severity.CRITICAL, "x"),))
    verdict = diagnosis.verdict()
    assert "image coverage" in verdict
    assert "should not be trusted" not in verdict


def test_verdict_is_clean_when_nothing_is_wrong():
    diagnosis = Diagnosis((Finding("a", "A", Severity.OK, "x"),))
    assert "no problems found" in diagnosis.verdict()


def test_summary_hides_passing_findings_by_default():
    findings = (
        Finding("a", "Alpha", Severity.OK, "fine"),
        Finding("b", "Beta", Severity.WARNING, "iffy", "do x"),
    )
    diagnosis = Diagnosis(findings)
    hidden = "\n".join(diagnosis.summary_lines())
    shown = "\n".join(diagnosis.summary_lines(include_ok=True))
    assert "Alpha" not in hidden and "Beta" in hidden
    assert "Alpha" in shown
    assert "do x" in hidden


def test_diagnosis_serialises():
    diagnosis = Diagnosis((Finding("a", "A", Severity.WARNING, "x", "y", {"n": 2}),))
    payload = diagnosis.to_dict()
    assert payload["severity"] == "WARNING"
    assert payload["findings"][0]["metrics"]["n"] == 2
    assert "verdict" in payload


def test_a_custom_diagnostic_set_is_honoured():
    class Always(Diagnostic):
        cause = "always"
        title = "Always"

        def run(self, context):
            return self._finding(Severity.NOTE, "hello", "world", n=1)

    context = rigs.healthy()
    diagnosis = diagnose(context.fit, context.observations, diagnostics=(Always,))
    assert len(diagnosis.findings) == 1
    assert diagnosis.findings[0].cause == "always"
    assert diagnosis.findings[0].metrics["n"] == 1


def test_every_finding_with_a_problem_carries_an_action():
    for builder in (rigs.frontoparallel, rigs.tiny_board, rigs.centre_only):
        context = builder()
        diagnosis = diagnose(context.fit, context.observations)
        for finding in diagnosis.findings:
            if finding.severity >= Severity.WARNING:
                assert finding.action, f"{finding.cause} has no action"


def test_every_finding_carries_numbers_behind_its_words():
    context = rigs.healthy()
    for finding in diagnose(context.fit, context.observations).findings:
        assert finding.metrics, f"{finding.cause} reported no metrics"
        assert finding.summary
