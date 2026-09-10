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

"""Turning a session or a fit into text a person can read, or JSON a script can.

The text layout is deliberately blunt about ordering. Identifiability comes
before standard deviations, because a standard deviation from a rank-deficient
system is worse than no number at all, and the reader has to hit the warning
before the table that would mislead them.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from ..core.session import CalibrationSession
from ..diagnose.report import Diagnosis
from ..diagnose.views import view_influences
from ..refit.result import InstrumentedFit
from ..validate.result import CrossValidation


def table(
    headers: Sequence[str], rows: Sequence[Sequence[str]], indent: str = "  "
) -> List[str]:
    """Render a fixed-width table.

    Args:
        headers: Column titles.
        rows: Cell text, already formatted.
        indent: Prefix for every line.

    Returns:
        One string per line, header and separator included. An empty `rows`
        yields an empty list rather than a lonely header.
    """
    if not rows:
        return []
    widths = [
        max(len(str(headers[i])), *(len(str(r[i])) for r in rows))
        for i in range(len(headers))
    ]
    def line(cells):
        return indent + "  ".join(str(c).rjust(w) for c, w in zip(cells, widths))
    return [line(headers), indent + "  ".join("-" * w for w in widths)] + [
        line(r) for r in rows
    ]


def render_session(session: CalibrationSession) -> str:
    """Render a session summary as text.

    Args:
        session: The session to describe.

    Returns:
        The report.
    """
    observations = session.observations
    title = "calibsense session"
    lines = [title, "=" * len(title), ""]
    lines += [f"  {line}" for line in session.summary_lines()]
    lines.append(f"  created      {session.created} by calibsense {session.calibsense_version}")

    failures = observations.summary.failures
    if failures:
        lines += ["", f"detection failures ({len(failures)})"]
        for failure in failures[:10]:
            lines.append(f"  {failure.source}: {failure.reason}")
        if len(failures) > 10:
            lines.append(f"  ... and {len(failures) - 10} more")

    counts = observations.points_per_view()
    coverage = counts / observations.target.num_points
    lines += ["", "per-view coverage"]
    lines += table(
        ["view", "points", "of target"],
        [
            [v.view_id, str(int(n)), f"{c:.0%}"]
            for v, n, c in zip(observations.views[:12], counts[:12], coverage[:12])
        ],
    )
    if observations.n_views > 12:
        lines.append(f"  ... and {observations.n_views - 12} more views")
    out_of_frame = observations.out_of_frame()
    if out_of_frame:
        lines.append(
            f"  note: {out_of_frame} detected point(s) lie outside the declared "
            f"{observations.image_size[0]}x{observations.image_size[1]} frame"
        )
    return "\n".join(lines) + "\n"


def render_fit(fit: InstrumentedFit, verbose: bool = False) -> str:
    """Render an instrumented fit as text.

    Args:
        fit: The fit to describe.
        verbose: Include every view rather than the worst few, and print the
            full intrinsic correlation matrix.

    Returns:
        The report.
    """
    covariance = fit.covariance
    conditioning = fit.conditioning
    title = "calibsense instrumented refit"
    lines = [title, "=" * len(title), ""]
    lines += [f"  {line}" for line in fit.summary_lines()]
    if fit.initial_guess:
        lines.append(f"  init         started from {fit.initial_guess}")

    lines += ["", "conditioning"]
    lines += table(
        ["quantity", "value"],
        [
            ["scaled condition number", f"{conditioning.scaled_condition_number:.4e}"],
            ["raw condition number", f"{conditioning.condition_number:.4e}"],
            ["rank", f"{conditioning.rank} / {conditioning.n_intrinsic}"],
            ["identifiable", str(conditioning.identifiable)],
        ],
    )
    if conditioning.singular_views:
        lines.append(
            f"  {len(conditioning.singular_views)} view(s) have a rank-deficient "
            f"pose block: {list(conditioning.singular_views)[:8]}"
        )

    lines += ["", "parameters"]
    rows = []
    for name, value, deviation, participation in fit.parameter_table():
        relative = abs(deviation / value) if value else float("inf")
        rows.append([
            name,
            f"{value:.6g}",
            f"{deviation:.4g}",
            f"{relative:.2%}" if np.isfinite(relative) else "-",
            f"{participation:.2f}",
        ])
    lines += table(["name", "value", "std dev", "relative", "weak"], rows)
    if not conditioning.identifiable:
        lines.append(
            "  a 'weak' value near 1 means that parameter lies in an "
            "unconstrained direction and its std dev is not meaningful"
        )

    correlations = covariance.strongest_correlations(6, threshold=0.3)
    if correlations:
        lines += ["", "strongest intrinsic correlations"]
        lines += table(
            ["pair", "correlation"],
            [[f"{a} ~ {b}", f"{v:+.3f}"] for a, b, v in correlations],
        )

    focal = covariance.correlation_with_poses("fx")[:, 5] if "fx" in covariance.intrinsic_names else None
    if focal is not None and focal.size:
        lines += ["", "focal length against working distance"]
        lines.append(
            f"  corr(fx, tz) mean {focal.mean():+.3f}, "
            f"range {focal.min():+.3f} to {focal.max():+.3f} over {focal.size} views"
        )

    views = fit.residuals.per_view if verbose else fit.residuals.worst_views(6)
    lines += ["", f"residuals by view ({'all' if verbose else 'worst'} "
                  f"{len(views)} of {fit.n_views})"]
    lines += table(
        ["view", "n", "rms", "p95", "max", "bias x", "bias y", "robust z"],
        [
            [
                v.view_id, str(v.n_points), f"{v.rms:.4f}", f"{v.p95:.4f}",
                f"{v.maximum:.4f}", f"{v.bias[0]:+.4f}", f"{v.bias[1]:+.4f}",
                f"{v.robust_z:.2f}" + ("*" if v.is_outlier else ""),
            ]
            for v in views
        ],
    )

    radial = fit.residuals.radial
    lines += ["", "residuals by image radius"]
    lines += table(
        ["radius px", "n", "radial mean", "radial rms", "tangential rms"],
        [
            [
                f"{radial.edges[i]:.0f}-{radial.edges[i + 1]:.0f}",
                str(int(radial.counts[i])),
                f"{radial.radial_mean[i]:+.4f}",
                f"{radial.radial_rms[i]:.4f}",
                f"{radial.tangential_rms[i]:.4f}",
            ]
            for i in range(radial.n_bins)
        ],
    )
    lines.append(
        "  a radial mean that grows with radius is the signature of a "
        "distortion model that does not match the lens"
    )

    validation = fit.cross_validation
    if validation is not None:
        lines += ["", "out-of-sample error"]
        lines += [f"  {line}" for line in validation.summary_lines()]
        lines += table(
            ["fold", "train", "held out", "train rms", "held-out rms", "identifiable"],
            [
                [
                    str(fold.index),
                    str(len(fold.train_view_ids)),
                    str(len(fold.held_out_view_ids)),
                    f"{fold.train_rms:.4f}",
                    f"{fold.held_out_rms:.4f}",
                    str(fold.identifiable),
                ]
                for fold in validation.folds
            ],
        )
        spread = [
            [name, f"{fold:.4g}", "-" if not np.isfinite(predicted) else f"{predicted:.4g}"]
            for name, fold, predicted in validation.spread_table()
        ]
        lines += ["", "parameter spread across folds, against the covariance"]
        lines += table(["name", "fold spread", "predicted"], spread)
        lines.append(
            "  folds share most of their training views, so the fold spread is a "
            "relative indicator rather than an unbiased sampling deviation"
        )

    if verbose:
        matrix = covariance.intrinsic_correlation()
        names = covariance.intrinsic_names
        lines += ["", "intrinsic correlation matrix"]
        lines += table(
            [""] + list(names),
            [[names[i]] + [f"{matrix[i, j]:+.2f}" for j in range(len(names))]
             for i in range(len(names))],
        )
    return "\n".join(lines) + "\n"


def render_diagnosis(diagnosis: Diagnosis, include_ok: bool = False) -> str:
    """Render a diagnosis as text, worst finding first.

    Args:
        diagnosis: The findings to render.
        include_ok: Also list the diagnostics that found nothing wrong.

    Returns:
        The report.
    """
    title = "calibsense diagnosis"
    lines = [title, "=" * len(title), ""]
    lines += [f"  {line}" if line else "" for line in
              diagnosis.summary_lines(include_ok=include_ok)]
    if not include_ok and diagnosis.passing:
        names = ", ".join(f.title for f in diagnosis.passing)
        lines += ["", f"  clean: {names}"]
    return "\n".join(lines) + "\n"


def diagnosis_to_json(diagnosis: Diagnosis) -> Dict[str, Any]:
    """Serialise a diagnosis to JSON-compatible data.

    Args:
        diagnosis: The findings to serialise.

    Returns:
        A dictionary suitable for `json.dumps`.
    """
    return diagnosis.to_dict()


def session_to_json(session: CalibrationSession) -> Dict[str, Any]:
    """Serialise a session summary to JSON-compatible data.

    Args:
        session: The session to describe.

    Returns:
        A dictionary suitable for `json.dumps`.
    """
    observations = session.observations
    return {
        "created": session.created,
        "calibsense_version": session.calibsense_version,
        "target": observations.target.to_dict(),
        "target_description": observations.target.describe(),
        "image_size": list(observations.image_size),
        "n_views": observations.n_views,
        "total_points": observations.total_points,
        "points_per_view": observations.points_per_view().tolist(),
        "out_of_frame": observations.out_of_frame(),
        "detection": observations.summary.to_dict(),
        "prior": session.prior.to_dict() if session.prior else None,
        "has_hand_eye": session.has_hand_eye,
        "metadata": dict(session.metadata),
    }


def fit_to_json(fit: InstrumentedFit) -> Dict[str, Any]:
    """Serialise a fit to JSON-compatible data.

    Large arrays — the per-corner residuals and the covariance factors — are
    left out. They live in the `.npz` bundle; this is the summary a dashboard or
    a CI check consumes.

    Args:
        fit: The fit to describe.

    Returns:
        A dictionary suitable for `json.dumps`.
    """
    covariance = fit.covariance
    depths = fit.working_distances_mm()
    focal = (
        covariance.correlation_with_poses("fx")[:, 5].tolist()
        if "fx" in covariance.intrinsic_names
        else []
    )
    return {
        "created": fit.created,
        "calibsense_version": fit.calibsense_version,
        "camera": fit.camera.to_dict(),
        "image_size": list(fit.image_size),
        "refitted": fit.refitted,
        "initial_guess": fit.initial_guess,
        "options": fit.options.to_dict(),
        "n_views": fit.n_views,
        "total_points": fit.residuals.total_points,
        "rms": fit.rms,
        "prior_rms": fit.prior_rms,
        "sigma": covariance.sigma,
        "degrees_of_freedom": covariance.degrees_of_freedom,
        "at_optimum": fit.at_optimum,
        "relative_decrement": fit.relative_decrement,
        "working_distance_mm": {"min": float(depths.min()), "max": float(depths.max())},
        "conditioning": fit.conditioning.to_dict(),
        "parameters": [
            {"name": n, "value": v, "std_dev": s, "weak_participation": p}
            for n, v, s, p in fit.parameter_table()
        ],
        "intrinsic_correlation": covariance.intrinsic_correlation().tolist(),
        "strongest_correlations": [
            {"a": a, "b": b, "correlation": v}
            for a, b, v in covariance.strongest_correlations(10)
        ],
        "focal_distance_correlation": focal,
        "views": [
            {
                "view_id": v.view_id,
                "n_points": v.n_points,
                "rms": v.rms,
                "bias": list(v.bias),
                "median": v.median,
                "p95": v.p95,
                "max": v.maximum,
                "robust_z": v.robust_z,
                "is_outlier": v.is_outlier,
                "distance_mm": float(depths[i]),
            }
            for i, v in enumerate(fit.residuals.per_view)
        ],
        "cross_validation": (
            fit.cross_validation.to_dict() if fit.cross_validation else None
        ),
        "radial_profile": {
            "edges": fit.residuals.radial.edges.tolist(),
            "counts": fit.residuals.radial.counts.tolist(),
            "radial_mean": fit.residuals.radial.radial_mean.tolist(),
            "radial_rms": fit.residuals.radial.radial_rms.tolist(),
            "tangential_rms": fit.residuals.radial.tangential_rms.tolist(),
        },
    }
