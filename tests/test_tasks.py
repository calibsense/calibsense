"""M5 — the tasks, and propagating them into millimetres."""

from __future__ import annotations

import numpy as np
import pytest

from caltrust.errors import ValidationError
from caltrust.refit import instrument
from caltrust.task import (
    CameraToBase,
    LengthAtDepth,
    MeasurementDistribution,
    PlaneLocation,
    Quantity,
    StereoTriangulation,
    propagate,
    propagate_all,
)
from caltrust.task.sampling import ParameterSample
from caltrust.task.tasks import _closest_approach

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
    from caltrust.task.sampling import CovarianceSampler

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
    from caltrust.handeye import solve_hand_eye

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
    from caltrust.handeye import solve_hand_eye

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
    from caltrust.handeye import solve_hand_eye

    session = rigs.hand_eye_session("eye_in_hand")
    fit = instrument(session)
    result = solve_hand_eye(fit, session, "eye_in_hand", monte_carlo=False)
    with pytest.raises(ValidationError, match="needs the flange pose"):
        CameraToBase(hand_eye_result=result, depth_mm=800.0)


def test_eye_to_hand_needs_no_flange_pose():
    from caltrust.handeye import solve_hand_eye

    session = rigs.hand_eye_session("eye_to_hand")
    fit = instrument(session)
    result = solve_hand_eye(fit, session, "eye_to_hand", monte_carlo=False)
    task = CameraToBase(hand_eye_result=result, depth_mm=800.0)
    sample = ParameterSample(fit.camera, hand_eye=result.camera)
    measured = task.measure(sample, task.observe(sample))
    expected = result.camera.apply(np.array([[0.0, 0.0, 800.0]]))[0]
    assert np.allclose(measured[:3], expected, atol=1e-6)


def test_camera_to_base_says_the_flange_pose_is_exact():
    from caltrust.handeye import solve_hand_eye

    session = rigs.hand_eye_session("eye_to_hand")
    fit = instrument(session)
    result = solve_hand_eye(fit, session, "eye_to_hand", monte_carlo=False)
    task = CameraToBase(hand_eye_result=result, depth_mm=800.0)
    assert task.to_dict()["flange_is_exact"] is True
    assert "exact" in task.describe()
