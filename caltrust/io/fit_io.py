"""Reading and writing instrumented fits.

The covariance is not stored. It is a deterministic function of the normal
equations and the `rcond` cut, both of which are stored, so it is rebuilt on
load. That keeps the bundle small and, more usefully, makes it impossible for a
saved covariance to disagree with the equations it came from. The bundle records
the caltrust version that wrote it, so a rebuild under different code is visible
rather than silent.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from ..core.camera import camera_from_dict
from ..core.poses import poses_from_array, poses_to_array
from ..errors import SerializationError
from ..refit.covariance import covariance_from_normal_equations
from ..refit.normal import NormalEquations
from ..refit.residuals import RadialProfile, ResidualStatistics, ViewResiduals
from ..refit.result import (
    InstrumentedFit,
    RefitOptions,
    conditioning_from_covariance,
)
from .bundle import check_format, read_bundle, write_bundle

FORMAT = "caltrust.fit"
FORMAT_VERSION = 1


def save_fit(fit: InstrumentedFit, path: str) -> str:
    """Write an instrumented fit to a single file.

    Args:
        fit: The fit to store.
        path: Destination path; `.npz` is appended when missing.

    Returns:
        The path actually written.
    """
    equations = fit.equations
    radial = fit.residuals.radial
    manifest: Dict[str, Any] = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "created": fit.created,
        "caltrust_version": fit.caltrust_version,
        "camera": fit.camera.to_dict(),
        "view_ids": list(fit.view_ids),
        "image_size": list(fit.image_size),
        "options": fit.options.to_dict(),
        "refitted": fit.refitted,
        "prior_rms": fit.prior_rms,
        "relative_decrement": fit.relative_decrement,
        "initial_guess": fit.initial_guess,
        "intrinsic_names": list(equations.intrinsic_names),
        "cost": equations.cost,
        "n_residuals": equations.n_residuals,
    }
    arrays = {
        "poses": poses_to_array(list(fit.poses)),
        "residuals": fit.residuals.residuals,
        "view_offsets": fit.residuals.view_offsets,
        "eq_u": equations.u,
        "eq_w": equations.w,
        "eq_v": equations.v,
        "eq_gradient_intrinsic": equations.gradient_intrinsic,
        "eq_gradient_pose": equations.gradient_pose,
        "radial_edges": radial.edges,
        "radial_counts": radial.counts,
        "radial_mean": radial.radial_mean,
        "radial_rms": radial.radial_rms,
        "tangential_rms": radial.tangential_rms,
    }
    return write_bundle(path, manifest, arrays)


def load_fit(path: str) -> InstrumentedFit:
    """Read an instrumented fit written by `save_fit`.

    Args:
        path: Path to the bundle.

    Returns:
        The reconstructed fit, with its covariance and conditioning rebuilt from
        the stored normal equations.

    Raises:
        SerializationError: The file is not a fit bundle or is inconsistent.
    """
    manifest, arrays = read_bundle(path)
    check_format(manifest, FORMAT, FORMAT_VERSION)
    try:
        equations = NormalEquations(
            u=arrays["eq_u"],
            w=arrays["eq_w"],
            v=arrays["eq_v"],
            gradient_intrinsic=arrays["eq_gradient_intrinsic"],
            gradient_pose=arrays["eq_gradient_pose"],
            cost=float(manifest["cost"]),
            n_residuals=int(manifest["n_residuals"]),
            intrinsic_names=tuple(manifest["intrinsic_names"]),
        )
        options = RefitOptions.from_dict(manifest["options"])
        residual_array = arrays["residuals"]
        offsets = arrays["view_offsets"]
        view_ids = tuple(manifest["view_ids"])
        radial = RadialProfile(
            edges=arrays["radial_edges"],
            counts=arrays["radial_counts"],
            radial_mean=arrays["radial_mean"],
            radial_rms=arrays["radial_rms"],
            tangential_rms=arrays["tangential_rms"],
        )
    except KeyError as exc:
        raise SerializationError(f"fit bundle is missing {exc}") from exc

    if len(view_ids) != len(offsets) - 1:
        raise SerializationError(
            f"{len(view_ids)} view ids but {len(offsets) - 1} residual spans"
        )
    covariance = covariance_from_normal_equations(equations, options.rcond)
    statistics = _rebuild_residuals(view_ids, residual_array, offsets, radial)
    return InstrumentedFit(
        camera=camera_from_dict(manifest["camera"]),
        poses=poses_from_array(arrays["poses"]),
        view_ids=view_ids,
        image_size=tuple(int(v) for v in manifest["image_size"]),
        covariance=covariance,
        residuals=statistics,
        conditioning=conditioning_from_covariance(covariance),
        equations=equations,
        refitted=bool(manifest.get("refitted", True)),
        options=options,
        prior_rms=manifest.get("prior_rms"),
        relative_decrement=float(manifest.get("relative_decrement", 0.0)),
        initial_guess=manifest.get("initial_guess"),
        created=manifest.get("created", ""),
        caltrust_version=manifest.get("caltrust_version", ""),
    )


def _rebuild_residuals(
    view_ids: Tuple[str, ...],
    residuals: np.ndarray,
    offsets: np.ndarray,
    radial: RadialProfile,
) -> ResidualStatistics:
    """Recompute the per-view summaries from the stored per-corner residuals."""
    from ..refit.residuals import _robust_z, _summarise_view

    summaries: List[ViewResiduals] = []
    for index, view_id in enumerate(view_ids):
        start, stop = int(offsets[index]), int(offsets[index + 1])
        summaries.append(_summarise_view(view_id, residuals[start:stop]))
    zeds = _robust_z(np.array([s.rms for s in summaries]))
    summaries = [
        ViewResiduals(
            s.view_id, s.n_points, s.rms, s.bias, s.median, s.p95, s.maximum, float(z)
        )
        for s, z in zip(summaries, zeds)
    ]
    magnitude = np.linalg.norm(residuals, axis=1)
    return ResidualStatistics(
        per_view=tuple(summaries),
        residuals=residuals,
        view_offsets=offsets,
        rms=float(np.sqrt(np.mean(magnitude ** 2))),
        bias=(float(residuals[:, 0].mean()), float(residuals[:, 1].mean())),
        radial=radial,
    )
