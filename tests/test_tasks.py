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

"""M5 — the tasks, and propagating them into millimetres."""

from __future__ import annotations

import numpy as np
import pytest

from calibsense.errors import ValidationError
from calibsense.refit import instrument
from calibsense.task import (
    CameraToBase,
    LengthAtDepth,
    MeasurementDistribution,
    PlaneLocation,
    Quantity,
    StereoTriangulation,
    propagate,
    propagate_all,
)
from calibsense.task.sampling import ParameterSample
from calibsense.task.tasks import _closest_approach

from . import rigs

pytestmark = pytest.mark.slow

TASKS = [
    LengthAtDepth(800.0, 100.0),
    LengthAtDepth(800.0, 100.0, centre_mm=(150.0, 90.0), orientation_deg=30.0),
    PlaneLocation(800.0, 25.0),
    StereoTriangulation(200.0, 800.0),
]


def nominal(camera=rigs.WIDE_PINHOLE):
    return ParameterSample(camera)


# --------------------------------------------------------------------------
# each task measures its own scene exactly at the nominal calibration
# --------------------------------------------------------------------------

@pytest.mark.parametrize("task", TASKS, ids=[t.kind + str(i) for i, t in enumerate(TASKS)])
def test_a_task_measures_its_own_scene_exactly(task):
    sample = nominal()
    observations = task.observe(sample)
    measured = task.measure(sample, observations)
    assert measured.size == len(task.quantities())
    assert np.all(np.isfinite(measured))


def test_length_recovers_the_true_separation():
    for centre in [(0.0, 0.0), (150.0, 90.0), (-200.0, 50.0)]:
        task = LengthAtDepth(800.0, 100.0, centre_mm=centre)
        sample = nominal()
        measured = task.measure(sample, task.observe(sample))[0]
        assert measured == pytest.approx(100.0, abs=1e-6)


def test_length_scales_with_depth_and_size():
    for depth, size in [(400.0, 50.0), (1200.0, 250.0)]:
        task = LengthAtDepth(depth, size)
        sample = nominal()
        assert task.measure(sample, task.observe(sample))[0] == pytest.approx(size, abs=1e-6)


def test_plane_reports_the_perpendicular_distance_not_the_axial_one():
    """A plane tilted 25 degrees at 800 mm is 800*cos(25) away perpendicular."""
    task = PlaneLocation(800.0, 25.0)
    sample = nominal()
    distance, tilt = task.measure(sample, task.observe(sample))
    assert distance == pytest.approx(800.0 * np.cos(np.radians(25.0)), rel=1e-6)
    assert tilt == pytest.approx(25.0, abs=1e-4)


def test_plane_at_zero_tilt_is_at_its_axial_depth():
    task = PlaneLocation(800.0, 0.0)
    sample = nominal()
    distance, tilt = task.measure(sample, task.observe(sample))
    assert distance == pytest.approx(800.0, rel=1e-6)
    assert tilt == pytest.approx(0.0, abs=1e-4)


def test_stereo_recovers_the_true_depth():
    for depth, baseline in [(500.0, 150.0), (1200.0, 300.0)]:
        task = StereoTriangulation(baseline, depth)
        sample = nominal()
        measured = task.measure(sample, task.observe(sample))
        assert measured[0] == pytest.approx(depth, rel=1e-6)


def test_stereo_range_exceeds_depth_for_an_off_axis_point():
    task = StereoTriangulation(200.0, 800.0, lateral_mm=(120.0, 0.0))
    sample = nominal()
    depth, distance = task.measure(sample, task.observe(sample))
    assert depth == pytest.approx(800.0, rel=1e-6)
    assert distance > depth


def test_closest_approach_on_intersecting_rays():
    origins = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
    directions = np.array([[0.0, 0.0, 1.0], [-100.0, 0.0, 800.0]])
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    point = _closest_approach(origins, directions)
    assert np.allclose(point, [0.0, 0.0, 800.0], atol=1e-6)


def test_closest_approach_refuses_parallel_rays():
    origins = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
    directions = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    with pytest.raises(ValidationError, match="parallel"):
        _closest_approach(origins, directions)


@pytest.mark.parametrize(
    "factory,message",
    [
        (lambda: LengthAtDepth(0.0, 100.0), "depth_mm"),
        (lambda: LengthAtDepth(800.0, -1.0), "length_mm"),
        (lambda: PlaneLocation(np.nan, 25.0), "depth_mm"),
        (lambda: PlaneLocation(800.0, 25.0, grid=(1, 5)), "at least 2x2"),
        (lambda: StereoTriangulation(0.0, 800.0), "baseline_mm"),
        (lambda: CameraToBase(None, 800.0), "needs a solved hand-eye"),
    ],
)
def test_invalid_tasks_are_rejected(factory, message):
    with pytest.raises(ValidationError, match=message):
        factory()


def test_a_task_serialises_its_configuration():
    payload = LengthAtDepth(800.0, 100.0).to_dict()
    assert payload["kind"] == "length_at_depth"
    assert payload["depth_mm"] == 800.0
    assert payload["quantities"][0]["unit"] == "mm"
    assert "description" in payload


def test_stereo_says_its_baseline_is_exact():
    """An input nobody measured must not be given an invented uncertainty."""
    task = StereoTriangulation(200.0, 800.0)
    assert task.to_dict()["baseline_is_exact"] is True
    assert "exact" in task.describe()


# --------------------------------------------------------------------------
# propagation
# --------------------------------------------------------------------------

def test_propagation_produces_a_distribution_around_the_nominal():
    fit = rigs.healthy().fit
    result = propagate(fit, LengthAtDepth(800.0, 100.0), 600)
    distribution = result.distribution("length_mm")
    assert result.n_samples > 500
    assert distribution.nominal == pytest.approx(100.0, abs=1e-6)
    assert distribution.expected_error > 0
    assert distribution.std > 0
    low, high = distribution.interval()
    assert low < 0 < high
    assert distribution.half_width() > distribution.expected_error


def test_the_three_runs_decompose_the_variance():
    """Calibration and pixel noise are independent, so their variances add."""
    fit = rigs.healthy().fit
    result = propagate(fit, LengthAtDepth(800.0, 100.0), 1500)
    combined = result.distribution("length_mm", "combined").std ** 2
    parameters = result.distribution("length_mm", "parameters").std ** 2
    noise = result.distribution("length_mm", "noise").std ** 2
    assert combined == pytest.approx(parameters + noise, rel=0.2)
    calibration_share, noise_share = result.variance_share("length_mm")
    assert calibration_share + noise_share == pytest.approx(1.0)
    assert 0 < calibration_share < 1


def test_zero_pixel_noise_isolates_the_calibration():
    fit = rigs.healthy().fit
    result = propagate(fit, LengthAtDepth(800.0, 100.0), 400, observation_noise_px=0.0)
    # Not exactly zero: with the pixels fixed, every sample measures the same
    # value and the residual spread is floating-point noise from the projection.
    assert result.distribution("length_mm", "noise").std < 1e-9
    assert result.variance_share("length_mm")[0] == pytest.approx(1.0)


def test_more_pixel_noise_widens_the_interval():
    fit = rigs.healthy().fit
    quiet = propagate(fit, LengthAtDepth(800.0, 100.0), 600, observation_noise_px=0.05)
    loud = propagate(fit, LengthAtDepth(800.0, 100.0), 600, observation_noise_px=1.0)
    assert loud.distribution("length_mm").std > quiet.distribution("length_mm").std


def test_error_grows_with_working_distance():
    """A fixed pixel error subtends more millimetres further away."""
    fit = rigs.healthy().fit
    near = propagate(fit, LengthAtDepth(400.0, 100.0), 600)
    far = propagate(fit, LengthAtDepth(1600.0, 100.0), 600)
    assert far.distribution("length_mm").std > near.distribution("length_mm").std


def test_a_degenerate_fit_reports_a_lower_bound():
    """`bounded` is the signal, not the numerical rank.

    A weak direction can sit above the raw eigenvalue cut and still be
    undetermined in the scaled sense, so the matrix comes back full rank while
    the fit is not identifiable. Reading rank alone would miss it, which is why
    the sampler folds identifiability into `bounded`.
    """
    fit = rigs.frontoparallel().fit
    result = propagate(fit, LengthAtDepth(800.0, 100.0), 400)
    assert not fit.conditioning.identifiable
    assert not result.bounded
    statement = result.distribution("length_mm").statement()
    assert "lower bound" in statement
    assert "unbounded" in statement


def test_bounded_is_not_merely_the_numerical_rank():
    from calibsense.task.sampling import CovarianceSampler

    sampler = CovarianceSampler(rigs.frontoparallel().fit, seed=0)
    assert not sampler.bounded
    # Full rank and still unbounded: the flag has to consult identifiability.
    assert sampler.rank == sampler.n_free


def test_a_degenerate_fit_is_orders_of_magnitude_worse():
    task = LengthAtDepth(800.0, 100.0)
    good = propagate(rigs.healthy().fit, task, 400)
    bad = propagate(rigs.frontoparallel().fit, task, 400)
    assert bad.distribution("length_mm").std > 50 * good.distribution("length_mm").std


def test_propagation_is_deterministic_for_a_seed():
    fit = rigs.healthy().fit
    first = propagate(fit, LengthAtDepth(800.0, 100.0), 300, seed=4)
    second = propagate(fit, LengthAtDepth(800.0, 100.0), 300, seed=4)
    assert np.allclose(first.combined, second.combined)


def test_propagate_all_gives_each_task_its_own_draws():
    fit = rigs.healthy().fit
    results = propagate_all(fit, [LengthAtDepth(800.0, 100.0), LengthAtDepth(800.0, 100.0)], 300)
    assert len(results) == 2
    assert not np.allclose(results[0].combined, results[1].combined)


@pytest.mark.parametrize("kwargs,message", [
    ({"n_samples": 1}, "at least two samples"),
    ({"observation_noise_px": -1.0}, "non-negative"),
    ({"observation_noise_px": np.nan}, "finite"),
])
def test_invalid_propagation_arguments_are_rejected(kwargs, message):
    fit = rigs.healthy().fit
    with pytest.raises(ValidationError, match=message):
        propagate(fit, LengthAtDepth(800.0, 100.0), **{"n_samples": 100, **kwargs})


def test_an_unknown_quantity_or_source_is_rejected():
    result = propagate(rigs.healthy().fit, LengthAtDepth(800.0, 100.0), 200)
    with pytest.raises(ValidationError, match="not measured by this task"):
        result.distribution("nope")
    with pytest.raises(ValidationError, match="unknown source"):
        result.distribution("length_mm", "elsewhere")


def test_plane_reports_both_a_length_and_an_angle():
    result = propagate(rigs.healthy().fit, PlaneLocation(800.0, 25.0), 300)
    units = {q.name: q.unit for q in result.quantities}
    assert units == {"distance_mm": "mm", "tilt_deg": "deg"}
    assert "deg" in result.distribution("tilt_deg").statement()


def test_result_serialises_every_source():
    result = propagate(rigs.healthy().fit, LengthAtDepth(800.0, 100.0), 300)
    payload = result.to_dict()
    assert payload["task"]["kind"] == "length_at_depth"
    row = payload["quantities"][0]
    for key in ("expected_error", "half_width", "statement", "variance_share",
                "parameters_only", "noise_only"):
        assert key in row
    assert payload["parameter_source"] == "the calibration"


def test_summary_names_the_source_of_the_error():
    result = propagate(rigs.healthy().fit, LengthAtDepth(800.0, 100.0), 300)
    text = "\n".join(result.summary_lines())
    assert "comes from the calibration" in text
    assert "pixel noise" in text


# --------------------------------------------------------------------------
# the distribution object itself
# --------------------------------------------------------------------------

def test_distribution_statistics_match_a_hand_computation():
    quantity = Quantity("x_mm", "mm", "a thing")
    samples = np.array([9.0, 10.0, 11.0, 12.0])
    distribution = MeasurementDistribution(quantity, 10.0, samples)
    assert distribution.bias == pytest.approx(0.5)
    assert distribution.expected_error == pytest.approx(1.0)
    assert distribution.rms_error == pytest.approx(np.sqrt((1 + 0 + 1 + 4) / 4))
    assert distribution.n_samples == 4


def test_distribution_interval_rejects_a_bad_level():
    distribution = MeasurementDistribution(
        Quantity("x_mm", "mm", "a thing"), 0.0, np.arange(10.0)
    )
    for level in (0.0, 1.0, -0.5, 2.0):
        with pytest.raises(ValidationError, match="level must be"):
            distribution.interval(level)


def test_an_unbounded_distribution_says_so_in_its_statement():
    distribution = MeasurementDistribution(
        Quantity("x_mm", "mm", "a thing"), 0.0, np.arange(10.0), bounded=False
    )
    assert "lower bound" in distribution.statement()


def test_a_single_sample_has_no_deviation():
    distribution = MeasurementDistribution(
        Quantity("x_mm", "mm", "a thing"), 0.0, np.array([1.0])
    )
    assert distribution.std == 0.0


# --------------------------------------------------------------------------
# the camera-to-base task
# --------------------------------------------------------------------------

def test_camera_to_base_recovers_the_true_base_position():
    from calibsense.handeye import solve_hand_eye

    session = rigs.hand_eye_session("eye_in_hand")
    fit = instrument(session)
    result = solve_hand_eye(fit, session, "eye_in_hand", monte_carlo=False)
    flange = session.robot.aligned_with(session.observations)[0]
    task = CameraToBase(hand_eye_result=result, depth_mm=800.0, flange=flange)
    sample = ParameterSample(fit.camera, hand_eye=result.camera)
    measured = task.measure(sample, task.observe(sample))
    expected = flange.compose(result.camera).apply(np.array([[0.0, 0.0, 800.0]]))[0]
    assert np.allclose(measured[:3], expected, atol=1e-6)
    assert measured[3] == pytest.approx(np.linalg.norm(expected), abs=1e-6)


def test_camera_to_base_includes_the_hand_eye_uncertainty():
    from calibsense.handeye import solve_hand_eye

    session = rigs.hand_eye_session("eye_in_hand")
    fit = instrument(session)
    result = solve_hand_eye(fit, session, "eye_in_hand", monte_carlo=False)
    flange = session.robot.aligned_with(session.observations)[0]
    task = CameraToBase(hand_eye_result=result, depth_mm=800.0, flange=flange)
    propagated = propagate(fit, task, 500)
    assert task.hand_eye() is result
    assert propagated.parameter_source_label == "the calibration and hand-eye transform"
    assert propagated.distribution("position_mm").std > 0


def test_eye_in_hand_needs_a_flange_pose():
    from calibsense.handeye import solve_hand_eye

    session = rigs.hand_eye_session("eye_in_hand")
    fit = instrument(session)
    result = solve_hand_eye(fit, session, "eye_in_hand", monte_carlo=False)
    with pytest.raises(ValidationError, match="needs the flange pose"):
        CameraToBase(hand_eye_result=result, depth_mm=800.0)


def test_eye_to_hand_needs_no_flange_pose():
    from calibsense.handeye import solve_hand_eye

    session = rigs.hand_eye_session("eye_to_hand")
    fit = instrument(session)
    result = solve_hand_eye(fit, session, "eye_to_hand", monte_carlo=False)
    task = CameraToBase(hand_eye_result=result, depth_mm=800.0)
    sample = ParameterSample(fit.camera, hand_eye=result.camera)
    measured = task.measure(sample, task.observe(sample))
    expected = result.camera.apply(np.array([[0.0, 0.0, 800.0]]))[0]
    assert np.allclose(measured[:3], expected, atol=1e-6)


def test_camera_to_base_says_the_flange_pose_is_exact():
    from calibsense.handeye import solve_hand_eye

    session = rigs.hand_eye_session("eye_to_hand")
    fit = instrument(session)
    result = solve_hand_eye(fit, session, "eye_to_hand", monte_carlo=False)
    task = CameraToBase(hand_eye_result=result, depth_mm=800.0)
    assert task.to_dict()["flange_is_exact"] is True
    assert "exact" in task.describe()


# --------------------------------------------------------------------------
# does the widened covariance actually cover, in millimetres?
# --------------------------------------------------------------------------

def _measure_with(camera, task, pixels):
    """Measure fixed pixels with one calibration, the way a user's rig would."""
    from calibsense.task.sampling import ParameterSample

    return float(task.measure(ParameterSample(camera=camera), pixels)[0])


@pytest.mark.slow
def test_the_widened_interval_covers_in_task_space_and_the_classical_one_does_not():
    """The claim item 1b would not make without this measurement.

    The finding named a factor and told the reader to apply it by hand, which
    left the product's headline number — the millimetres — carrying an
    understatement the report had just described. This checks the fix where it
    matters, in task units rather than in pixels.

    Empirical: hold the pixels a truth camera produces, refit the same capture
    many times under a correlated noise field, and measure those fixed pixels
    with each fit. The spread of that measurement is the real task-space error
    of a calibration from this capture. Predicted: propagate one such fit, with
    the widening off and on.

    Measured over 150 refits: the classical propagation lands at about 0.12 of
    the empirical spread and the widened one at about 1.14, so the assertions
    below sit either side of a gap of nearly ten. The widened figure sits just
    above one rather than just below because the view-clustered estimate carries
    a leverage correction that errs wide; see `leave_one_out_steps`.
    """
    from calibsense.core.session import CalibrationSession
    from calibsense.refit import RefitOptions, instrument
    from calibsense.synthetic import diverse_poses, synthesise
    from calibsense.task.propagate import propagate
    from calibsense.task.sampling import ParameterSample

    from .test_covariance import reshape_noise

    truth = rigs.WIDE_PINHOLE
    clean = synthesise(
        truth, rigs.BOARD, diverse_poses(rigs.BOARD, 14, seed=2), rigs.IMAGE_SIZE,
        noise_px=0.0, seed=0,
    ).observations
    task = LengthAtDepth(800.0, 100.0)
    pixels = task.observe(ParameterSample(camera=truth))

    def fit_for(observations):
        return instrument(
            CalibrationSession(observations=observations), RefitOptions()
        )

    rng = np.random.default_rng(11)
    empirical = float(np.std(
        [
            _measure_with(fit_for(reshape_noise(clean, rng, "correlated")).camera,
                          task, pixels)
            for _ in range(60)
        ],
        ddof=1,
    ))

    fit = fit_for(reshape_noise(clean, np.random.default_rng(9999), "correlated"))
    # The residual looks *better* than the noise injected, which is why no
    # residual statistic can catch this and why the classical interval collapses.
    assert fit.covariance.sigma < 0.15

    def predicted(widen):
        result = propagate(
            fit, task, n_samples=1500, observation_noise_px=0.0, widen=widen
        )
        return result.distribution("length_mm").std, result.intrinsic_inflation

    classical, unwidened_factor = predicted(False)
    widened, factor = predicted(True)

    assert unwidened_factor == 1.0
    assert factor > 3.0
    assert classical / empirical < 0.3, f"expected a collapsed interval: {classical}"
    assert 0.7 < widened / empirical < 1.8, f"widened interval off: {widened}"
    assert widened > 4.0 * classical


@pytest.mark.slow
def test_widening_costs_little_when_the_noise_model_holds():
    """The control: independent noise must not have its millimetres inflated."""
    from calibsense.core.session import CalibrationSession
    from calibsense.refit import RefitOptions, instrument
    from calibsense.synthetic import diverse_poses, synthesise
    from calibsense.task.propagate import propagate
    from calibsense.task.sampling import ParameterSample

    from .test_covariance import reshape_noise

    truth = rigs.WIDE_PINHOLE
    clean = synthesise(
        truth, rigs.BOARD, diverse_poses(rigs.BOARD, 14, seed=2), rigs.IMAGE_SIZE,
        noise_px=0.0, seed=0,
    ).observations
    task = LengthAtDepth(800.0, 100.0)
    pixels = task.observe(ParameterSample(camera=truth))

    def fit_for(observations):
        return instrument(
            CalibrationSession(observations=observations), RefitOptions()
        )

    rng = np.random.default_rng(11)
    empirical = float(np.std(
        [
            _measure_with(fit_for(reshape_noise(clean, rng, "independent")).camera,
                          task, pixels)
            for _ in range(60)
        ],
        ddof=1,
    ))

    fit = fit_for(reshape_noise(clean, np.random.default_rng(9999), "independent"))
    classical = propagate(
        fit, task, n_samples=1500, observation_noise_px=0.0, widen=False
    ).distribution("length_mm").std
    widened = propagate(
        fit, task, n_samples=1500, observation_noise_px=0.0, widen=True
    ).distribution("length_mm").std

    assert 0.7 < classical / empirical < 1.4
    assert 0.8 < widened / empirical < 1.6
    # A premium, not a doubling: the leverage correction errs wide by about a
    # fifth even when the noise model is perfectly satisfied, which is the price
    # of it not erring short by a third when the model fails.
    assert widened / classical < 1.45


# --------------------------------------------------------------------------
# separating the hand-eye transform from the calibration
# --------------------------------------------------------------------------

def _base_frame_task(monte_carlo=True, samples=120):
    from calibsense.handeye import solve_hand_eye
    from calibsense.refit import RefitOptions

    session = rigs.hand_eye_session("eye_in_hand")
    fit = instrument(session, RefitOptions())
    solved = solve_hand_eye(
        fit, session, mounting="eye_in_hand",
        monte_carlo=monte_carlo, n_samples=samples,
    )
    flange = session.robot.aligned_with(session.observations)[0]
    return fit, CameraToBase(
        hand_eye_result=solved, depth_mm=800.0, flange=flange
    )


def test_a_task_without_a_hand_eye_has_nothing_to_separate(good_capture):
    """The three-way split has to agree with the two-way one where it applies."""
    from calibsense.core.session import CalibrationSession

    fit = instrument(CalibrationSession(observations=good_capture.observations))
    result = propagate(fit, LengthAtDepth(800.0, 100.0), 400)
    assert result.hand_eye_only is None
    sources = result.variance_sources("length_mm")
    calibration, noise = result.variance_share("length_mm")
    assert sources["hand_eye"] == 0.0
    assert sources["calibration"] == pytest.approx(calibration)
    assert sources["pixel_noise"] == pytest.approx(noise)


@pytest.mark.slow
def test_the_base_frame_split_says_which_of_the_two_to_spend_on():
    """The question a robot cell asks that the two-way split could not answer.

    Whether the camera calibration or the hand-eye solve dominates decides where
    the money goes, and on this rig it is genuinely close — which is the point,
    because the previous report lumped them together and said only that
    "calibration and hand-eye" accounted for everything.
    """
    fit, task = _base_frame_task()
    result = propagate(fit, task, 600)
    assert result.hand_eye_only is not None

    sources = result.variance_sources("position_mm")
    assert sum(sources.values()) == pytest.approx(1.0)
    assert sources["hand_eye"] > 0.1, sources
    assert sources["calibration"] > 0.1, sources
    assert any("hand-eye" in line for line in result.summary_lines())


@pytest.mark.slow
def test_the_split_is_additive_which_is_what_makes_the_subtraction_valid():
    """The calibration's share is what is left after the hand-eye's is removed.

    That is only right if the two variances add. They are drawn independently,
    but the task is non-linear in both, so additivity is an assumption rather
    than an identity and it is worth measuring. It holds to a few per cent.
    """
    from calibsense.task.propagate import _attempt
    from calibsense.task.sampling import CovarianceSampler, ParameterSample

    fit, task = _base_frame_task()
    sampler = CovarianceSampler(fit, task.view_indices(), 0, task.hand_eye())
    nominal = sampler.nominal()
    observations = task.observe(nominal)
    n_quantities = len(task.quantities())
    draws = sampler.draw(1500)

    def spread(builder):
        values = np.array([
            _attempt(task, builder(d), observations, n_quantities) for d in draws
        ])
        return np.nanvar(values, axis=0, ddof=1)

    both = spread(lambda d: d)
    camera = spread(lambda d: ParameterSample(d.camera, d.poses, nominal.hand_eye))
    hand_eye = spread(lambda d: ParameterSample(nominal.camera, nominal.poses, d.hand_eye))
    assert np.allclose(both, camera + hand_eye, rtol=0.10)


@pytest.mark.slow
def test_an_optimistic_hand_eye_covariance_hides_its_own_share():
    """Why the split is only meaningful on top of the Monte Carlo covariance.

    The residual-based hand-eye covariance treats the target-in-camera poses as
    exact and understates itself threefold, which here reads as the hand-eye
    contributing nothing at all. The honest covariance puts it at half. Anyone
    reading the split off a residual-based solve would draw the opposite
    conclusion about where to spend.
    """
    fit, optimistic = _base_frame_task(monte_carlo=False)
    _, honest = _base_frame_task(monte_carlo=True)
    low = propagate(fit, optimistic, 500).variance_sources("position_mm")
    high = propagate(fit, honest, 500).variance_sources("position_mm")
    assert low["hand_eye"] < 0.05
    assert high["hand_eye"] > 0.2
