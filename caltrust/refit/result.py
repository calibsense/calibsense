"""What an instrumented refit produces.

Everything the standard calibration call computes and discards, in one object:
the full covariance in factored form, the per-view and per-corner residual
distributions, the conditioning of the normal equations, and the parameter
correlation matrix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .._version import __version__
from ..core.camera import CameraModel
from ..core.poses import Pose
from ..errors import ValidationError
from .covariance import CalibrationCovariance, WeakDirection
from .normal import NormalEquations
from .residuals import ResidualStatistics

#: Predicted relative cost reduction below which a parameter set counts as
#: sitting at the optimum of its own objective.
OPTIMUM_DECREMENT_TOLERANCE = 1e-6


@dataclass(frozen=True)
class RefitOptions:
    """How the refit should be run.

    Attributes:
        model: `"pinhole"`, `"fisheye"`, or `None` to take the model from the
            session's existing calibration and fall back to pinhole.
        distortion_terms: Number of Brown-Conrady coefficients to estimate: 5,
            8, 12 or 14. Ignored for fisheye, which always has four. Passing 4
            is accepted and means five coefficients with `k3` held at zero.
        fixed: Intrinsic parameter names to hold rather than estimate. What
            "hold" means differs by model, and the difference is OpenCV's, not
            caltrust's: `cv2.calibrateCamera` keeps a fixed pinhole coefficient
            at the value it was handed, while `cv2.fisheye.calibrate` sets a
            fixed Kannala-Brandt coefficient to zero and ignores the value
            supplied for it. Either way the parameter leaves the estimate and
            the covariance, so the reported uncertainty is correct; only the
            resulting value differs.
        tie_aspect: Hold `fy / fx` constant rather than estimating both.
        refit: Re-estimate the parameters. When `False`, the session's existing
            calibration is instrumented in place and only the poses are solved,
            which is what auditing a shipped calibration means.
        use_prior_as_guess: Start the optimiser from the session's existing
            calibration when there is one.
        rcond: Relative eigenvalue cut for every inverse taken.
        radial_bins: Number of bins in the radial residual profile.
        check_condition: Ask OpenCV's fisheye path to reject ill-conditioned
            views instead of fitting through them.
    """

    model: Optional[str] = None
    distortion_terms: int = 5
    fixed: Tuple[str, ...] = ()
    tie_aspect: bool = False
    refit: bool = True
    use_prior_as_guess: bool = True
    rcond: float = 1e-12
    radial_bins: int = 10
    check_condition: bool = False

    def __post_init__(self) -> None:
        if self.model not in (None, "pinhole", "fisheye"):
            raise ValidationError(
                f"model must be 'pinhole', 'fisheye' or None, got {self.model!r}"
            )
        if self.distortion_terms not in (4, 5, 8, 12, 14):
            raise ValidationError(
                f"distortion_terms must be one of 4, 5, 8, 12, 14, "
                f"got {self.distortion_terms}"
            )
        if not 0 < self.rcond < 1:
            raise ValidationError(f"rcond must be in (0, 1), got {self.rcond}")
        if self.radial_bins < 1:
            raise ValidationError(f"radial_bins must be positive, got {self.radial_bins}")
        object.__setattr__(self, "fixed", tuple(self.fixed))

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dictionary."""
        return {
            "model": self.model,
            "distortion_terms": self.distortion_terms,
            "fixed": list(self.fixed),
            "tie_aspect": self.tie_aspect,
            "refit": self.refit,
            "use_prior_as_guess": self.use_prior_as_guess,
            "rcond": self.rcond,
            "radial_bins": self.radial_bins,
            "check_condition": self.check_condition,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "RefitOptions":
        """Rebuild from `to_dict` output."""
        return cls(
            model=payload.get("model"),
            distortion_terms=int(payload.get("distortion_terms", 5)),
            fixed=tuple(payload.get("fixed", ())),
            tie_aspect=bool(payload.get("tie_aspect", False)),
            refit=bool(payload.get("refit", True)),
            use_prior_as_guess=bool(payload.get("use_prior_as_guess", True)),
            rcond=float(payload.get("rcond", 1e-12)),
            radial_bins=int(payload.get("radial_bins", 10)),
            check_condition=bool(payload.get("check_condition", False)),
        )


@dataclass(frozen=True)
class Conditioning:
    """How well posed the estimation problem was.

    Attributes:
        rank: Rank of the intrinsic Schur complement.
        n_intrinsic: Number of free intrinsic parameters.
        condition_number: Raw condition number. Depends on the units the
            parameters happen to be measured in, so it is reported for
            completeness rather than for judgement.
        scaled_condition_number: Condition number after Jacobi scaling. This is
            the one to read: it is unit-free, and it is what separates a
            well-posed capture (order 1e3) from a degenerate one (1e10 and up).
        weak_directions: Parameter combinations the data does not determine.
        participation: Per-parameter share of the weak subspace, in `[0, 1]`.
        singular_views: Views whose own pose block was rank deficient.
    """

    rank: int
    n_intrinsic: int
    condition_number: float
    scaled_condition_number: float
    weak_directions: Tuple[WeakDirection, ...]
    participation: np.ndarray
    singular_views: Tuple[int, ...]

    @property
    def identifiable(self) -> bool:
        """Whether every intrinsic combination is determined by the data.

        When this is `False`, the reported standard deviations describe only the
        subspace the data does constrain, and taking them at face value would
        overstate confidence badly. This flag outranks every other number in the
        report.
        """
        return not self.weak_directions

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dictionary."""
        return {
            "rank": self.rank,
            "n_intrinsic": self.n_intrinsic,
            "condition_number": self.condition_number,
            "scaled_condition_number": self.scaled_condition_number,
            "identifiable": self.identifiable,
            "weak_directions": [
                {"eigenvalue_ratio": d.eigenvalue_ratio, "terms": d.terms}
                for d in self.weak_directions
            ],
            "participation": np.asarray(self.participation).tolist(),
            "singular_views": list(self.singular_views),
        }


def conditioning_from_covariance(covariance: CalibrationCovariance) -> Conditioning:
    """Extract the conditioning report from a covariance.

    Args:
        covariance: The covariance produced by the refit.

    Returns:
        The conditioning summary.
    """
    return Conditioning(
        rank=covariance.spectrum.rank,
        n_intrinsic=covariance.n_intrinsic,
        condition_number=covariance.spectrum.condition_number,
        scaled_condition_number=covariance.spectrum.scaled_condition_number,
        weak_directions=tuple(covariance.weak_directions()),
        participation=covariance.parameter_participation(),
        singular_views=tuple(np.flatnonzero(covariance.singular_views).tolist()),
    )


@dataclass(frozen=True)
class InstrumentedFit:
    """A calibration together with everything needed to judge it.

    Attributes:
        camera: The intrinsic model, either re-estimated or taken from the
            session depending on `options.refit`.
        poses: Board-to-camera pose per view, in view order.
        view_ids: The views, in the same order as `poses`.
        image_size: Frame size as `(width, height)`.
        covariance: Full parameter covariance, in factored form.
        residuals: Per-view and per-corner residual distributions.
        conditioning: How well posed the problem was.
        equations: The assembled normal equations.
        refitted: Whether the parameters were re-estimated.
        options: The settings this fit was produced under.
        prior_rms: The RMS the session's existing calibration claimed, if any.
        created: ISO-8601 UTC timestamp.
        caltrust_version: Version that produced the fit.
    """

    camera: CameraModel
    poses: Tuple[Pose, ...]
    view_ids: Tuple[str, ...]
    image_size: Tuple[int, int]
    covariance: CalibrationCovariance
    residuals: ResidualStatistics
    conditioning: Conditioning
    equations: NormalEquations
    refitted: bool
    options: RefitOptions
    prior_rms: Optional[float] = None
    relative_decrement: float = 0.0
    initial_guess: Optional[str] = None
    created: str = ""
    caltrust_version: str = __version__

    def __post_init__(self) -> None:
        if not self.created:
            object.__setattr__(
                self, "created", datetime.now(timezone.utc).isoformat(timespec="seconds")
            )

    @property
    def rms(self) -> float:
        """In-sample reprojection RMS in pixels.

        This is the number the engineer already has. It is reported here so the
        rest of the audit can be compared against it, not because it means much
        on its own.
        """
        return self.residuals.rms

    @property
    def n_views(self) -> int:
        """Number of views."""
        return len(self.view_ids)

    @property
    def at_optimum(self) -> bool:
        """Whether the parameters sit at a minimum of their own objective.

        Measured as the cost reduction a full Newton step would predict,
        relative to the current cost. When this is `False` — which happens when
        auditing a calibration produced elsewhere, or when the optimiser gave up
        in a flat valley — the covariance describes the curvature at the point it
        was evaluated rather than an estimator's sampling spread.
        """
        return self.relative_decrement < OPTIMUM_DECREMENT_TOLERANCE

    def working_distances_mm(self) -> np.ndarray:
        """Distance from the camera to each view's board centre, in millimetres."""
        return np.array([float(np.linalg.norm(p.translation)) for p in self.poses])

    def summary_lines(self) -> Tuple[str, ...]:
        """A short human summary, used by the CLI and by reports."""
        covariance = self.covariance
        depths = self.working_distances_mm()
        lines = [
            f"model        {self.camera.kind} "
            f"({'refitted' if self.refitted else 'instrumented in place'})",
            f"views        {self.n_views}, {self.residuals.total_points} points",
            f"RMS          {self.rms:.4f} px in-sample"
            + (
                f"  (calibration file reported {self.prior_rms:.4f} px)"
                if self.prior_rms is not None
                else ""
            ),
            f"sigma        {covariance.sigma:.4f} px per coordinate, "
            f"{covariance.degrees_of_freedom} degrees of freedom",
            f"depth range  {depths.min():.0f} to {depths.max():.0f} mm",
            f"conditioning scaled condition number "
            f"{self.conditioning.scaled_condition_number:.3e}, "
            f"rank {self.conditioning.rank}/{self.conditioning.n_intrinsic}",
        ]
        if not self.conditioning.identifiable:
            lines.append(
                "IDENTIFIABILITY  this capture does not determine every parameter; "
                "the standard deviations below understate the true uncertainty"
            )
            for direction in self.conditioning.weak_directions:
                lines.append(f"   unconstrained: {direction.describe()}")
        if not self.at_optimum:
            lines.append(
                f"NOT AT OPTIMUM  a Newton step would cut the cost by "
                f"{self.relative_decrement:.2%}; these parameters are not the "
                "best fit to these detections"
            )
        outliers = self.residuals.outlier_views()
        if outliers:
            worst = ", ".join(f"{v.view_id} ({v.rms:.2f} px)" for v in outliers[:3])
            lines.append(f"outlier views {len(outliers)}: {worst}")
        return tuple(lines)

    def parameter_table(self) -> List[Tuple[str, float, float, float]]:
        """Value, standard deviation and weak-subspace share per free intrinsic.

        Returns:
            One row per free intrinsic, as
            `(name, value, standard_deviation, participation)`. A participation
            near one means the standard deviation in that row is not meaningful.
        """
        values = dict(zip(self.camera.parameter_names(), self.camera.to_vector()))
        deviations = self.covariance.intrinsic_std()
        participation = self.conditioning.participation
        return [
            (name, float(values[name]), float(deviations[i]), float(participation[i]))
            for i, name in enumerate(self.covariance.intrinsic_names)
        ]
