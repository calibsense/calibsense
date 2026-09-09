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

"""What a task is, and what propagating one produces.

A task is a measurement an engineer actually makes. It knows how to observe its
own scene through a camera and how to measure the scene back from those pixels.
Everything else — sampling, propagation, reporting — is generic over that pair.

The split matters and it is not arbitrary. Observations are what the sensor saw;
the calibration is what is uncertain about interpreting them. So the scene is
projected *once* through the fitted camera to fix the pixels, and then measured
back many times through sampled cameras. Re-projecting the scene with each
sampled camera instead would ask a different and much less useful question.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..errors import ValidationError
from .sampling import ParameterSample

#: Confidence level used wherever a report says "95% interval".
DEFAULT_LEVEL = 0.95


@dataclass(frozen=True)
class Quantity:
    """One number a task measures.

    Attributes:
        name: Machine-readable slug, for example `"length_mm"`.
        unit: Physical unit, `"mm"` or `"deg"`.
        description: How to say it in a sentence, for example
            `"the distance between two features at 800 mm"`.
    """

    name: str
    unit: str
    description: str

    def to_dict(self) -> Dict[str, str]:
        """Serialise to a plain dictionary."""
        return {"name": self.name, "unit": self.unit, "description": self.description}


class Task(ABC):
    """A measurement whose error is to be propagated from the calibration."""

    #: Stable slug identifying the task type.
    kind: ClassVar[str] = ""
    #: Short human name.
    title: ClassVar[str] = ""

    @abstractmethod
    def quantities(self) -> Tuple[Quantity, ...]:
        """The numbers this task measures, in the order `measure` returns them."""

    def view_indices(self) -> Tuple[int, ...]:
        """Views whose poses the task needs sampled alongside the intrinsics.

        Empty for tasks that depend on the intrinsics only, which is most of
        them: a length measured at a known depth does not care where the
        calibration target happened to be.
        """
        return ()

    def hand_eye(self):
        """The hand-eye result this task needs sampled, or `None`.

        Returns:
            A `HandEyeResult`, or `None` for tasks that stay in the camera frame.
        """
        return None

    @abstractmethod
    def observe(self, sample: ParameterSample) -> np.ndarray:
        """Project this task's scene into the image.

        Called once, with the fitted calibration, to fix what the sensor saw.

        Args:
            sample: The nominal parameter set.

        Returns:
            An `(n, 2)` array of pixel coordinates.
        """

    @abstractmethod
    def measure(
        self, sample: ParameterSample, image_points: np.ndarray
    ) -> np.ndarray:
        """Measure the scene back from pixels, under one parameter set.

        Args:
            sample: The parameter set to measure under.
            image_points: The `(n, 2)` pixels from `observe`.

        Returns:
            One value per entry of `quantities`.
        """

    @abstractmethod
    def describe(self) -> str:
        """A one-line description of the task and its geometry."""

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the task's configuration."""
        return {
            "kind": self.kind,
            "title": self.title,
            "description": self.describe(),
            "quantities": [q.to_dict() for q in self.quantities()],
        }


@dataclass(frozen=True)
class MeasurementDistribution:
    """The distribution of one measured quantity over the sampled calibrations.

    Attributes:
        quantity: What is being measured.
        nominal: The value at the fitted calibration.
        samples: The measured value under each sampled parameter set.
        bounded: Whether the sampling covered the whole parameter space. When
            `False`, every spread below is a lower bound.
    """

    quantity: Quantity
    nominal: float
    samples: np.ndarray
    bounded: bool = True

    @property
    def errors(self) -> np.ndarray:
        """Deviation of each sample from the nominal measurement."""
        return self.samples - self.nominal

    @property
    def n_samples(self) -> int:
        """Number of samples."""
        return int(self.samples.size)

    @property
    def bias(self) -> float:
        """Mean signed error.

        A non-zero bias means the measurement is non-linear in the parameters
        over their uncertainty range, so the fitted calibration is not the mean
        of the distribution it implies.
        """
        return float(np.mean(self.errors))

    @property
    def std(self) -> float:
        """Standard deviation of the measurement."""
        return float(np.std(self.samples, ddof=1)) if self.n_samples > 1 else 0.0

    @property
    def expected_error(self) -> float:
        """Mean absolute error.

        The number to quote as "expected error": it is what a single measurement
        is typically off by, which is what an engineer is asking.
        """
        return float(np.mean(np.abs(self.errors)))

    @property
    def rms_error(self) -> float:
        """Root-mean-square error, which includes the bias."""
        return float(np.sqrt(np.mean(self.errors ** 2)))

    def interval(self, level: float = DEFAULT_LEVEL) -> Tuple[float, float]:
        """Percentile interval of the error.

        Args:
            level: Coverage, for example `0.95`.

        Returns:
            The lower and upper error bounds.

        Raises:
            ValidationError: The level is not in `(0, 1)`.
        """
        if not 0.0 < level < 1.0:
            raise ValidationError(f"level must be in (0, 1), got {level}")
        tail = 100.0 * (1.0 - level) / 2.0
        low, high = np.percentile(self.errors, [tail, 100.0 - tail])
        return float(low), float(high)

    def half_width(self, level: float = DEFAULT_LEVEL) -> float:
        """Half-width of the error interval, for a plus-or-minus statement.

        Args:
            level: Coverage.

        Returns:
            Half the width of the percentile interval.
        """
        low, high = self.interval(level)
        return float((high - low) / 2.0)

    def statement(self, level: float = DEFAULT_LEVEL) -> str:
        """The sentence this whole package exists to produce.

        Args:
            level: Coverage for the interval.

        Returns:
            One sentence naming the expected error and the interval, with the
            caveat attached when the sampling was not bounded.
        """
        unit = self.quantity.unit
        sentence = (
            f"{self.quantity.description} has an expected error of "
            f"{self.expected_error:.3g} {unit}, with a {level:.0%} interval of "
            f"+/-{self.half_width(level):.3g} {unit}"
        )
        if not self.bounded:
            sentence += (
                ", and that is a lower bound: this calibration leaves a parameter "
                "direction undetermined, so the true spread is unbounded"
            )
        return sentence

    def to_dict(self, level: float = DEFAULT_LEVEL) -> Dict[str, Any]:
        """Serialise to a plain JSON-compatible dictionary."""
        low, high = self.interval(level)
        return {
            "quantity": self.quantity.to_dict(),
            "nominal": self.nominal,
            "n_samples": self.n_samples,
            "bounded": self.bounded,
            "bias": self.bias,
            "std": self.std,
            "expected_error": self.expected_error,
            "rms_error": self.rms_error,
            "level": level,
            "interval": [low, high],
            "half_width": self.half_width(level),
            "statement": self.statement(level),
        }


@dataclass(frozen=True)
class TaskResult:
    """Propagated task-space error, split by where it comes from.

    Attributes:
        task: The task that was propagated.
        nominal: The measurement at the fitted calibration, one per quantity.
        combined: Samples with both the calibration and pixel noise varying,
            shape `(n, k)`. This is the number a measurement actually carries.
        parameters_only: Samples with the calibration varying and the pixels
            fixed. Isolates what the calibration costs.
        noise_only: Samples with the pixels varying and the calibration fixed.
            Isolates what a better calibration could never fix.
        bounded: Whether the parameter sampling covered the whole space.
        observation_noise_px: Per-coordinate pixel noise used, in pixels.
        sampler_rank: Rank of the sampled covariance.
        sampler_size: Dimension of the sampled covariance.
    """

    task: Task
    nominal: np.ndarray
    combined: np.ndarray
    parameters_only: np.ndarray
    noise_only: np.ndarray
    bounded: bool
    observation_noise_px: float
    sampler_rank: int
    sampler_size: int

    @property
    def n_samples(self) -> int:
        """Number of Monte Carlo samples."""
        return int(self.combined.shape[0])

    @property
    def quantities(self) -> Tuple[Quantity, ...]:
        """The measured quantities."""
        return self.task.quantities()

    @property
    def parameter_source_label(self) -> str:
        """What the non-pixel share of the error is attributed to.

        Named rather than always "the calibration", because a task that goes
        through a hand-eye transform carries that transform's uncertainty too,
        and it usually dominates.
        """
        return (
            "the calibration and hand-eye transform"
            if self.task.hand_eye() is not None
            else "the calibration"
        )

    def _index_of(self, quantity: str) -> int:
        names = [q.name for q in self.quantities]
        try:
            return names.index(quantity)
        except ValueError:
            raise ValidationError(
                f"{quantity!r} is not measured by this task; have {names}"
            ) from None

    def distribution(
        self, quantity: str, source: str = "combined"
    ) -> MeasurementDistribution:
        """The distribution of one quantity.

        Args:
            quantity: Name of the quantity.
            source: `"combined"`, `"parameters"` or `"noise"`.

        Returns:
            That quantity's distribution.

        Raises:
            ValidationError: The quantity or the source is unknown.
        """
        samples = {
            "combined": self.combined,
            "parameters": self.parameters_only,
            "noise": self.noise_only,
        }
        if source not in samples:
            raise ValidationError(
                f"unknown source {source!r}; expected one of {sorted(samples)}"
            )
        index = self._index_of(quantity)
        return MeasurementDistribution(
            quantity=self.quantities[index],
            nominal=float(self.nominal[index]),
            samples=samples[source][:, index],
            bounded=self.bounded or source == "noise",
        )

    def distributions(self, source: str = "combined") -> Tuple[MeasurementDistribution, ...]:
        """Every quantity's distribution, in task order.

        Args:
            source: `"combined"`, `"parameters"` or `"noise"`.

        Returns:
            One distribution per quantity.
        """
        return tuple(
            self.distribution(q.name, source) for q in self.quantities
        )

    def variance_share(self, quantity: str) -> Tuple[float, float]:
        """How the error splits between the calibration and pixel noise.

        The two contributions are independent, so their variances add. The
        shares are normalised to sum to one, which makes the answer to "would a
        better calibration help" a single number.

        Args:
            quantity: Name of the quantity.

        Returns:
            The calibration's share and the pixel noise's share.
        """
        parameters = self.distribution(quantity, "parameters").std ** 2
        noise = self.distribution(quantity, "noise").std ** 2
        total = parameters + noise
        if total <= 0:
            return 0.0, 0.0
        return parameters / total, noise / total

    def summary_lines(self, level: float = DEFAULT_LEVEL) -> Tuple[str, ...]:
        """A human summary, leading with the task-space statement."""
        lines: List[str] = [self.task.describe(), ""]
        for quantity in self.quantities:
            distribution = self.distribution(quantity.name)
            lines.append(distribution.statement(level))
            calibration, noise = self.variance_share(quantity.name)
            lines.append(
                f"    of that, {calibration:.0%} comes from "
                f"{self.parameter_source_label} and {noise:.0%} from "
                f"{self.observation_noise_px:.3g} px of pixel noise"
            )
            if abs(distribution.bias) > 0.1 * max(distribution.std, 1e-12):
                lines.append(
                    f"    the fitted value is offset from the distribution mean by "
                    f"{distribution.bias:+.3g} {quantity.unit}, so the measurement "
                    "is non-linear over this uncertainty range"
                )
        return tuple(lines)

    def to_dict(self, level: float = DEFAULT_LEVEL) -> Dict[str, Any]:
        """Serialise to a plain JSON-compatible dictionary."""
        return {
            "task": self.task.to_dict(),
            "n_samples": self.n_samples,
            "bounded": self.bounded,
            "observation_noise_px": self.observation_noise_px,
            "sampler_rank": self.sampler_rank,
            "sampler_size": self.sampler_size,
            "parameter_source": self.parameter_source_label,
            "quantities": [
                {
                    **self.distribution(q.name).to_dict(level),
                    "variance_share": {
                        "calibration": self.variance_share(q.name)[0],
                        "pixel_noise": self.variance_share(q.name)[1],
                    },
                    "parameters_only": self.distribution(q.name, "parameters").to_dict(level),
                    "noise_only": self.distribution(q.name, "noise").to_dict(level),
                }
                for q in self.quantities
            ],
        }
