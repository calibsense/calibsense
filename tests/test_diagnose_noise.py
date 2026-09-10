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

"""The noise model check.

The point of this diagnostic is that it fires on correlated corner noise and on
nothing else. Every other diagnostic in the suite owns some other fault, and a
check that also fired on those would just be a second voice saying the same
thing. So these tests are as much about what stays quiet as about what speaks.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from calibsense.diagnose import Severity, diagnose
from calibsense.diagnose.noise import (
    INFLATION_CRITICAL,
    INFLATION_WARNING,
    NoiseModelValidity,
)
from calibsense.refit.covariance import MIN_CLUSTERS_FOR_ROBUST

from . import rigs


def test_a_healthy_capture_passes_the_noise_check():
    finding = NoiseModelValidity().run(rigs.healthy())
    assert finding.severity is Severity.OK
    assert finding.metrics["measured"] is True
    assert finding.metrics["worst_inflation"] < INFLATION_WARNING
    assert finding.metrics["degrees_of_freedom"] == finding.metrics["n_clusters"] - 1


@pytest.mark.parametrize("length_px", [60.0, 200.0])
def test_correlated_corner_noise_is_called_out(length_px):
    """The one violation that matters, at two correlation lengths."""
    context = rigs.with_correlated_noise(length_px=length_px)
    finding = NoiseModelValidity().run(context)
    assert finding.severity is Severity.CRITICAL
    assert finding.metrics["worst_inflation"] > INFLATION_CRITICAL
    assert "correlat" in finding.summary
    assert finding.action


def test_the_residual_looks_better_under_the_fault_it_catches():
    """Why a residual statistic cannot stand in for this check.

    Correlated noise is partly absorbed by the pose and distortion parameters,
    so the fit reports a *smaller* sigma than the noise actually present. Any
    reader watching the RMS would conclude the capture had got better.
    """
    healthy = rigs.healthy().fit.covariance.sigma
    correlated = rigs.with_correlated_noise().fit.covariance.sigma
    assert correlated < healthy


def test_one_noisy_view_is_left_to_the_outlier_diagnostic():
    """Noise that is worse in some views than others does not break the model.

    Measured: per-view scale variation leaves the classical covariance within
    about fifteen per cent. `OutlierViews` owns this fault, and this check
    staying quiet is what keeps the two from reporting the same thing twice.
    """
    finding = NoiseModelValidity().run(rigs.with_bad_view())
    assert finding.severity is Severity.OK
    assert finding.metrics["worst_inflation"] < INFLATION_WARNING


def test_short_range_correlation_is_a_warning_not_a_critical():
    """Correlation at the sub-pixel window scale is real but mild.

    A 20 px correlation length is about the size of the `cornerSubPix` window,
    which is the one place neighbouring corners genuinely could share image
    gradients. It costs a factor under two rather than a factor of six, so it
    warns instead of condemning.
    """
    finding = NoiseModelValidity().run(rigs.with_correlated_noise(length_px=20.0))
    assert finding.severity is Severity.WARNING
    assert INFLATION_WARNING < finding.metrics["worst_inflation"] <= INFLATION_CRITICAL


def test_an_unidentifiable_capture_defers_to_the_identifiability_finding():
    """Both estimates drop the same null space, so their agreement means nothing."""
    finding = NoiseModelValidity().run(rigs.frontoparallel())
    assert finding.severity is Severity.NOTE
    assert finding.metrics["measured"] is False
    assert "determine every parameter" in finding.summary


def test_too_few_views_reports_the_limit_rather_than_a_ratio():
    context = rigs.context(pose_set=rigs.poses(n=6))
    finding = NoiseModelValidity().run(context)
    assert finding.severity is Severity.NOTE
    assert finding.metrics["measured"] is False
    assert finding.metrics["n_clusters"] == 6
    assert finding.metrics["min_clusters"] == MIN_CLUSTERS_FOR_ROBUST


def test_a_fit_away_from_its_optimum_cannot_be_checked():
    """The scores sum to zero only at a minimum; elsewhere the ratio is noise."""
    context = rigs.healthy()
    fit = dataclasses.replace(context.fit, relative_decrement=1e-2)
    assert not fit.at_optimum
    finding = NoiseModelValidity().run(dataclasses.replace(context, fit=fit))
    assert finding.severity is Severity.NOTE
    assert finding.metrics["measured"] is False
    assert "optimum" in finding.summary


def test_a_fit_without_per_view_scores_says_so():
    context = rigs.healthy()
    equations = dataclasses.replace(
        context.fit.equations, gradient_intrinsic_view=None
    )
    covariance = dataclasses.replace(context.fit.covariance, robust=None)
    fit = dataclasses.replace(context.fit, equations=equations, covariance=covariance)
    finding = NoiseModelValidity().run(dataclasses.replace(context, fit=fit))
    assert finding.severity is Severity.NOTE
    assert finding.metrics["measured"] is False


def test_the_check_appears_in_the_full_diagnosis():
    context = rigs.with_correlated_noise()
    diagnosis = diagnose(context.fit, context.observations)
    finding = diagnosis.by_cause("noise_model")
    assert finding.severity is Severity.CRITICAL
    assert finding in diagnosis.critical
    assert diagnosis.severity is Severity.CRITICAL


def test_the_metrics_carry_both_deviation_sets():
    finding = NoiseModelValidity().run(rigs.healthy())
    names = rigs.healthy().fit.covariance.intrinsic_names
    assert set(finding.metrics["inflation"]) == set(names)
    classical = np.asarray(finding.metrics["classical_std"])
    robust = np.asarray(finding.metrics["robust_std"])
    assert classical.shape == robust.shape == (len(names),)
    assert np.all(classical > 0) and np.all(robust > 0)
    # The deviation arrays are positional while `inflation` is a dict, so the
    # row order has to be stated rather than guessed from the dict's keys.
    assert tuple(finding.metrics["names"]) == tuple(names)
    for i, name in enumerate(finding.metrics["names"]):
        assert finding.metrics["inflation"][name] == pytest.approx(
            robust[i] / classical[i]
        )


def test_the_summary_reports_the_worst_parameter_by_name():
    finding = NoiseModelValidity().run(rigs.with_correlated_noise())
    worst = finding.metrics["worst_parameter"]
    assert worst in finding.metrics["inflation"]
    assert worst in finding.summary
    assert finding.metrics["inflation"][worst] == pytest.approx(
        finding.metrics["worst_inflation"]
    )


def test_the_summary_line_agrees_with_the_diagnostic():
    """The one-line summary must not quote a ratio the finding would refuse to.

    `summary_lines` is read far more often than the full report, so a number
    printed there without the diagnostic's guards would be exactly the silent
    wrongness this tool exists to catch.
    """
    healthy = rigs.healthy()
    line = next(l for l in healthy.fit.summary_lines() if "noise model" in l)
    assert "view-clustered deviations are" in line

    # Away from an optimum the scores carry a gradient that inflates the ratio.
    off = dataclasses.replace(healthy.fit, relative_decrement=1e-2)
    line = next(l for l in off.summary_lines() if "noise model" in l)
    assert "not checked" in line and "optimum" in line

    # On an unidentifiable fit both estimates drop the same null space.
    line = next(
        l for l in rigs.frontoparallel().fit.summary_lines() if "noise model" in l
    )
    assert "not checked" in line and "determine" in line


def test_a_capture_with_too_few_views_prints_no_noise_line():
    """Below the cluster floor there is nothing honest to say, so nothing is said."""
    fit = rigs.context(pose_set=rigs.poses(n=6)).fit
    assert not fit.covariance.robust.usable
    assert not any("noise model" in l for l in fit.summary_lines())
