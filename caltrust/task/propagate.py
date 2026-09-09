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

"""M5 — propagating parameter uncertainty into task space.

Monte Carlo rather than a linearised push-through, for two reasons. Tasks like
triangulation and plane fitting are non-linear enough in the parameters that a
first-order propagation understates the tails, and more importantly a Monte
Carlo run reports the *distribution*, so a bias — the fitted calibration not
sitting at the mean of what it implies — becomes visible instead of being
assumed away.

Three runs, not one. The calibration and the pixel noise are independent, so
their variances add, and separating them answers the question that follows every
error figure: would a better calibration help, or is this the sensor's floor?
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..errors import ValidationError
from ..refit.result import InstrumentedFit
from .base import Task, TaskResult
from .sampling import CovarianceSampler, ParameterSample

#: Samples drawn when the caller does not say. Enough for a stable 95%
#: percentile interval without making a report slow to produce.
DEFAULT_SAMPLES = 2000

#: Fraction of samples that may fail to measure before the result is refused.
MAX_FAILURE_FRACTION = 0.05


def propagate(
    fit: InstrumentedFit,
    task: Task,
    n_samples: int = DEFAULT_SAMPLES,
    observation_noise_px: Optional[float] = None,
    seed: Optional[int] = 0,
) -> TaskResult:
    """Propagate a calibration's uncertainty into a task's units.

    Args:
        fit: The instrumented refit whose covariance is being propagated.
        task: The measurement to evaluate.
        n_samples: Monte Carlo samples.
        observation_noise_px: Per-coordinate pixel noise at measurement time.
            Defaults to the fit's own residual sigma, which is the noise
            caltrust actually measured on this camera. Pass `0.0` to isolate the
            calibration's contribution.
        seed: Seed, so a report is reproducible.

    Returns:
        The measurement distribution, split by source.

    Raises:
        ValidationError: The sample count or noise level is invalid, or too many
            samples failed to measure.
    """
    if n_samples < 2:
        raise ValidationError(f"need at least two samples, got {n_samples}")
    noise_px = (
        float(fit.covariance.sigma)
        if observation_noise_px is None
        else float(observation_noise_px)
    )
    if not np.isfinite(noise_px) or noise_px < 0:
        raise ValidationError(
            f"observation noise must be finite and non-negative, got {noise_px}"
        )

    sampler = CovarianceSampler(fit, task.view_indices(), seed, task.hand_eye())
    nominal_sample = sampler.nominal()
    observations = task.observe(nominal_sample)
    nominal = np.asarray(task.measure(nominal_sample, observations), dtype=float)
    n_quantities = nominal.size
    if n_quantities != len(task.quantities()):
        raise ValidationError(
            f"{task.kind} declares {len(task.quantities())} quantities but measured "
            f"{n_quantities}"
        )

    draws = sampler.draw(n_samples)
    rng = np.random.default_rng(None if seed is None else seed + 1)
    perturbations = (
        rng.normal(0.0, noise_px, (n_samples,) + observations.shape)
        if noise_px > 0
        else np.zeros((n_samples,) + observations.shape)
    )

    combined = np.full((n_samples, n_quantities), np.nan)
    parameters_only = np.full((n_samples, n_quantities), np.nan)
    noise_only = np.full((n_samples, n_quantities), np.nan)
    for index, draw in enumerate(draws):
        noisy = observations + perturbations[index]
        combined[index] = _attempt(task, draw, noisy, n_quantities)
        parameters_only[index] = _attempt(task, draw, observations, n_quantities)
        noise_only[index] = _attempt(task, nominal_sample, noisy, n_quantities)

    combined, parameters_only, noise_only = _drop_failures(
        task, combined, parameters_only, noise_only
    )
    return TaskResult(
        task=task,
        nominal=nominal,
        combined=combined,
        parameters_only=parameters_only,
        noise_only=noise_only,
        bounded=sampler.bounded,
        observation_noise_px=noise_px,
        sampler_rank=sampler.rank,
        sampler_size=sampler.n_free,
    )


def propagate_all(
    fit: InstrumentedFit,
    tasks: Sequence[Task],
    n_samples: int = DEFAULT_SAMPLES,
    observation_noise_px: Optional[float] = None,
    seed: Optional[int] = 0,
) -> Tuple[TaskResult, ...]:
    """Propagate several tasks against one calibration.

    Args:
        fit: The instrumented refit.
        tasks: The measurements to evaluate.
        n_samples: Monte Carlo samples per task.
        observation_noise_px: Pixel noise, defaulting to the fit's sigma.
        seed: Base seed; each task is offset from it so the tasks do not share
            the same draws.

    Returns:
        One result per task, in order.
    """
    return tuple(
        propagate(
            fit, task, n_samples, observation_noise_px,
            None if seed is None else seed + 7919 * index,
        )
        for index, task in enumerate(tasks)
    )


def _attempt(
    task: Task, sample: ParameterSample, observations: np.ndarray, size: int
) -> np.ndarray:
    """Measure, returning `nan` where the geometry degenerated."""
    try:
        value = np.asarray(task.measure(sample, observations), dtype=float)
    except (ValidationError, Exception):  # noqa: BLE001 - OpenCV raises its own
        return np.full(size, np.nan)
    if value.size != size or not np.all(np.isfinite(value)):
        return np.full(size, np.nan)
    return value


def _drop_failures(
    task: Task, *arrays: np.ndarray
) -> Tuple[np.ndarray, ...]:
    """Remove samples any of the three runs failed to measure.

    A draw is dropped from all three runs or none, so the variance decomposition
    stays comparable across them.

    Raises:
        ValidationError: More than `MAX_FAILURE_FRACTION` of samples failed,
            which means the task geometry is not viable under this calibration's
            uncertainty rather than that a few draws were unlucky.
    """
    usable = np.ones(arrays[0].shape[0], dtype=bool)
    for array in arrays:
        usable &= np.all(np.isfinite(array), axis=1)
    failed = int((~usable).sum())
    total = usable.size
    if failed > MAX_FAILURE_FRACTION * total:
        raise ValidationError(
            f"{failed} of {total} samples could not be measured for "
            f"{task.kind!r}; the task geometry is not viable under this "
            "calibration's uncertainty, which is itself the finding"
        )
    if failed == 0:
        return arrays
    return tuple(array[usable] for array in arrays)
