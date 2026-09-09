"""What cross-validation produces.

The headline is a ratio. In-sample reprojection RMS is a training error, and the
only honest way to find out how much it flatters the model is to hold views back
and ask the fit to predict them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from ..core.camera import CameraModel, camera_from_dict
from ..errors import ValidationError


@dataclass(frozen=True)
class HeldOutView:
    """One view predicted by a fit that never saw it.

    Attributes:
        view_id: The view's identifier.
        n_points: Points contributing.
        rms: Reprojection RMS in pixels, at the fold's intrinsics with this
            view's own pose solved by PnP.
        distance_mm: Distance from the camera to the board centre.
        fold: Which fold held this view out.
    """

    view_id: str
    n_points: int
    rms: float
    distance_mm: float
    fold: int

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dictionary."""
        return {
            "view_id": self.view_id,
            "n_points": self.n_points,
            "rms": self.rms,
            "distance_mm": self.distance_mm,
            "fold": self.fold,
        }


@dataclass(frozen=True)
class Fold:
    """One train/test split.

    Attributes:
        index: Zero-based fold number.
        train_view_ids: Views the fold was fitted on.
        held_out_view_ids: Views kept back.
        camera: The intrinsics this fold estimated.
        train_rms: In-sample RMS on the training views.
        held_out_rms: Pooled RMS over the held-out views.
        held_out_points: Points in the held-out set.
        identifiable: Whether the fold's own training set determined every
            intrinsic. A fold that says `False` produced a held-out number from
            a meaningless fit, and pooling it in would be dishonest.
    """

    index: int
    train_view_ids: Tuple[str, ...]
    held_out_view_ids: Tuple[str, ...]
    camera: CameraModel
    train_rms: float
    held_out_rms: float
    held_out_points: int
    identifiable: bool

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dictionary."""
        return {
            "index": self.index,
            "train_view_ids": list(self.train_view_ids),
            "held_out_view_ids": list(self.held_out_view_ids),
            "camera": self.camera.to_dict(),
            "train_rms": self.train_rms,
            "held_out_rms": self.held_out_rms,
            "held_out_points": self.held_out_points,
            "identifiable": self.identifiable,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Fold":
        """Rebuild from `to_dict` output."""
        return cls(
            index=int(payload["index"]),
            train_view_ids=tuple(payload["train_view_ids"]),
            held_out_view_ids=tuple(payload["held_out_view_ids"]),
            camera=camera_from_dict(payload["camera"]),
            train_rms=float(payload["train_rms"]),
            held_out_rms=float(payload["held_out_rms"]),
            held_out_points=int(payload["held_out_points"]),
            identifiable=bool(payload["identifiable"]),
        )


@dataclass(frozen=True)
class CrossValidation:
    """K-fold cross-validation over views.

    Attributes:
        folds: Every train/test split, in order.
        held_out: Per-view held-out results, pooled across folds.
        in_sample_rms: RMS of the full fit on all views. This is the number the
            engineer already has.
        out_of_sample_rms: RMS pooled over every held-out point, weighted by
            point count rather than averaged across folds.
        n_views: Views in the session.
        fold_spread: Standard deviation of each parameter across the folds. This
            is model-free — it resamples the actual views and assumes nothing
            about the noise — and it catches degeneracies the ratio cannot see.
            It is not an unbiased estimate of the sampling standard deviation,
            because folds share most of their training data, so read it as a
            relative indicator beside `predicted_spread` rather than as a
            replacement for it.
        predicted_spread: The standard deviation the full fit's covariance
            predicts for each parameter, `nan` where a parameter was not
            estimated. Present so the two can be compared in one place.
        parameter_names: Names matching both spread arrays, over the full
            parameter vector rather than just the free part.
        degenerate_folds: Folds whose training set was not identifiable.
    """

    folds: Tuple[Fold, ...]
    held_out: Tuple[HeldOutView, ...]
    in_sample_rms: float
    out_of_sample_rms: float
    n_views: int
    fold_spread: np.ndarray
    predicted_spread: np.ndarray
    parameter_names: Tuple[str, ...]
    degenerate_folds: Tuple[int, ...] = ()

    @property
    def n_folds(self) -> int:
        """Number of folds."""
        return len(self.folds)

    @property
    def ratio(self) -> float:
        """How much the in-sample RMS understates out-of-sample error.

        This is the single most useful number in a calibration report. A value
        near one means the reprojection error is an honest error estimate. A
        value of three means the calibration is fitting its own images and will
        not generalise to the next one.
        """
        if self.in_sample_rms <= 0:
            return float("inf")
        return self.out_of_sample_rms / self.in_sample_rms

    @property
    def held_out_points(self) -> int:
        """Total points evaluated out of sample."""
        return int(sum(view.n_points for view in self.held_out))

    @property
    def trustworthy(self) -> bool:
        """Whether every fold's training set was identifiable.

        When this is `False` the ratio is still reported, but it mixes folds
        whose intrinsics were not determined by their own data, and it should be
        read as a lower bound at best.
        """
        return not self.degenerate_folds

    def worst_views(self, count: int = 5) -> Tuple[HeldOutView, ...]:
        """The held-out views the fit predicted worst, worst first.

        Args:
            count: How many to return.

        Returns:
            Up to `count` held-out results.
        """
        return tuple(sorted(self.held_out, key=lambda v: -v.rms)[:count])

    def spread_of(self, parameter: str) -> float:
        """Across-fold standard deviation of one parameter.

        Args:
            parameter: Name of a camera parameter.

        Returns:
            The standard deviation of that parameter over the folds.

        Raises:
            ValidationError: The parameter is not one of this camera's.
        """
        return float(self.fold_spread[self._index_of(parameter)])

    def predicted_of(self, parameter: str) -> float:
        """The standard deviation the covariance predicts for one parameter.

        Args:
            parameter: Name of a camera parameter.

        Returns:
            The predicted standard deviation, or `nan` if it was not estimated.

        Raises:
            ValidationError: The parameter is not one of this camera's.
        """
        return float(self.predicted_spread[self._index_of(parameter)])

    def spread_table(self) -> List[Tuple[str, float, float]]:
        """Fold spread against predicted spread, one row per parameter.

        Returns:
            Triples of `(name, fold_spread, predicted_spread)`.
        """
        return [
            (name, float(self.fold_spread[i]), float(self.predicted_spread[i]))
            for i, name in enumerate(self.parameter_names)
        ]

    def _index_of(self, parameter: str) -> int:
        try:
            return self.parameter_names.index(parameter)
        except ValueError:
            raise ValidationError(
                f"{parameter!r} is not a parameter of this camera; "
                f"have {list(self.parameter_names)}"
            ) from None

    def summary_lines(self) -> Tuple[str, ...]:
        """A short human summary, used by the CLI and by reports.

        Identifiability gates the verdict, for the same reason it gates the
        standard deviations in M2: a ratio near one means "the reprojection
        error is honest" only when the folds were fitting a determined model.
        Otherwise it means the held-out poses absorbed the error.
        """
        lines = [
            f"folds         {self.n_folds} over {self.n_views} views, "
            f"{self.held_out_points} points held out",
            f"in-sample     {self.in_sample_rms:.4f} px",
            f"out-of-sample {self.out_of_sample_rms:.4f} px",
        ]
        if self.degenerate_folds:
            lines.append(
                f"ratio         {self.ratio:.2f}x, but it means nothing here: "
                f"{len(self.degenerate_folds)} of {self.n_folds} folds trained on "
                "a set that does not determine every intrinsic"
            )
            lines.append(
                "              a held-out view solves its own pose, so it absorbs "
                "exactly the errors an undetermined fit makes; read the fold "
                "spread below instead"
            )
        elif self.ratio < 1.25:
            lines.append(
                f"ratio         {self.ratio:.2f}x, so the in-sample error is honest"
            )
        else:
            lines.append(
                f"ratio         {self.ratio:.2f}x, so the in-sample error understates "
                f"out-of-sample error by {self.ratio:.1f}x"
            )
        worst = self._worst_spread()
        if worst is not None:
            name, fold, predicted = worst
            comparison = (
                f", against {predicted:.4g} predicted by the covariance"
                if np.isfinite(predicted)
                else ""
            )
            lines.append(
                f"fold spread   {name} varies by {fold:.4g} across folds{comparison}"
            )
        return tuple(lines)

    def _worst_spread(self) -> Optional[Tuple[str, float, float]]:
        """The parameter whose fold spread is largest relative to its prediction."""
        rows = self.spread_table()
        if not rows:
            return None
        def key(row):
            _, fold, predicted = row
            if not np.isfinite(predicted) or predicted <= 0:
                return -np.inf
            return fold / predicted
        ranked = max(rows, key=key)
        return ranked if np.isfinite(key(ranked)) else None

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain JSON-compatible dictionary.

        The derived fields — `n_folds`, `held_out_points`, `ratio` and
        `trustworthy` — are written out too, so a consumer reading the JSON does
        not have to re-derive the numbers the report is actually about. They are
        ignored on the way back in.
        """
        return {
            "folds": [fold.to_dict() for fold in self.folds],
            "held_out": [view.to_dict() for view in self.held_out],
            "in_sample_rms": self.in_sample_rms,
            "out_of_sample_rms": self.out_of_sample_rms,
            "ratio": self.ratio,
            "n_folds": self.n_folds,
            "held_out_points": self.held_out_points,
            "trustworthy": self.trustworthy,
            "n_views": self.n_views,
            "fold_spread": np.asarray(self.fold_spread).tolist(),
            "predicted_spread": np.asarray(self.predicted_spread).tolist(),
            "parameter_names": list(self.parameter_names),
            "degenerate_folds": list(self.degenerate_folds),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CrossValidation":
        """Rebuild from `to_dict` output."""
        return cls(
            folds=tuple(Fold.from_dict(f) for f in payload["folds"]),
            held_out=tuple(
                HeldOutView(
                    view_id=v["view_id"], n_points=int(v["n_points"]),
                    rms=float(v["rms"]), distance_mm=float(v["distance_mm"]),
                    fold=int(v["fold"]),
                )
                for v in payload["held_out"]
            ),
            in_sample_rms=float(payload["in_sample_rms"]),
            out_of_sample_rms=float(payload["out_of_sample_rms"]),
            n_views=int(payload["n_views"]),
            fold_spread=np.asarray(payload["fold_spread"], dtype=float),
            predicted_spread=np.asarray(payload["predicted_spread"], dtype=float),
            parameter_names=tuple(payload["parameter_names"]),
            degenerate_folds=tuple(int(i) for i in payload.get("degenerate_folds", ())),
        )
