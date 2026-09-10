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

"""The measured noise floor: recover injected noise whose truth is known.

Every test here builds a static capture with one property deliberately set --
a scale, an axis ratio, a correlation length, a drift -- and asserts that the
measurement gets that property back. That is the only way to know the tool
reports the noise rather than an artefact of how it was computed.
"""

from __future__ import annotations

import numpy as np
import pytest

from calibsense.core.camera import PinholeBrownConrady
from calibsense.core.observations import ObservationSet, ViewObservations
from calibsense.core.target import Checkerboard
from calibsense.errors import ValidationError
from calibsense.noisefloor import (
    ANISOTROPY_EXCESS_WARNING,
    CORRELATION_SPACINGS_CRITICAL,
    CORRELATION_SPACINGS_WARNING,
    EDGE_ALIGNED_BY_CHANCE,
    MIN_FRAMES,
    anisotropy_floor,
    measure_noise_floor,
)
from calibsense.refit.projection import projector_for
from calibsense.synthetic import pose_for_view

IMAGE_SIZE = (1280, 720)
BOARD = Checkerboard(9, 6, 25.0)
CAMERA = PinholeBrownConrady(900.0, 905.0, 639.5, 359.5, [-0.21, 0.06, 0.001, -0.002, 0.01])


@pytest.fixture(scope="module")
def truth() -> np.ndarray:
    """Where the corners sit when nothing is moving."""
    pose = pose_for_view(BOARD, 700.0, tilt_rad=0.25)
    return projector_for(CAMERA).project(CAMERA, pose, BOARD.object_points())


def static_capture(truth, noise, n_frames=120, seed=0, presence=None):
    """A static capture whose per-frame deviation comes from `noise`.

    Args:
        truth: Noise-free corner positions.
        noise: Callable taking `(rng, frame_index)` and returning an
            `(n_points, 2)` deviation in pixels.
        n_frames: Frames to synthesise.
        seed: Seed for the deviations.
        presence: Optional callable taking `(rng, frame_index)` and returning a
            boolean mask of which points were detected in that frame.

    Returns:
        An `ObservationSet` with one view per frame.
    """
    rng = np.random.default_rng(seed)
    views = []
    for index in range(n_frames):
        points = truth + noise(rng, index)
        keep = (
            np.ones(len(truth), dtype=bool) if presence is None else presence(rng, index)
        )
        views.append(
            ViewObservations(
                f"f{index:04d}", np.flatnonzero(keep), points[keep]
            )
        )
    return ObservationSet(BOARD, IMAGE_SIZE, tuple(views))


def iid(sigma):
    return lambda rng, index: rng.normal(0.0, sigma, (BOARD.num_points, 2))


def anisotropic(sigma_x, sigma_y):
    return lambda rng, index: rng.normal(0.0, 1.0, (BOARD.num_points, 2)) * np.array(
        [sigma_x, sigma_y]
    )


def correlated(truth, sigma, length, n_centres=6):
    """A smooth random field per frame, sampled at the corners.

    Squared-exponential weights around random centres give a field whose
    correlation falls off over roughly `length` pixels, which is the one noise
    property that actually breaks the classical covariance.
    """

    def noise(rng, index):
        centres = rng.uniform([0, 0], IMAGE_SIZE, (n_centres, 2))
        amplitude = rng.normal(0.0, 1.0, (n_centres, 2))
        distance = np.linalg.norm(truth[:, None, :] - centres[None, :, :], axis=2)
        weight = np.exp(-(distance ** 2) / (2.0 * length ** 2))
        field = np.stack([weight @ amplitude[:, 0], weight @ amplitude[:, 1]], axis=1)
        return field * (sigma / max(field.std(), 1e-12))

    return noise


def drifting(sigma, total, n_frames=120):
    def noise(rng, index):
        shift = np.array([total * index / (n_frames - 1.0), 0.0])
        return rng.normal(0.0, sigma, (BOARD.num_points, 2)) + shift

    return noise


class TestScale:
    """The headline number: how big is the noise."""

    @pytest.mark.parametrize("sigma", [0.02, 0.05, 0.2, 0.8])
    def test_recovers_injected_sigma(self, truth, sigma):
        floor = measure_noise_floor(static_capture(truth, iid(sigma)))
        assert floor.total_sigma_px == pytest.approx(sigma, rel=0.05)

    def test_irreducible_sits_just_below_total_for_clean_noise(self, truth):
        floor = measure_noise_floor(static_capture(truth, iid(0.2)))
        # Six affine parameters over 54 corners can only absorb a small share,
        # so a clean capture must not look mostly reducible.
        assert 0.0 < floor.absorbed_fraction < 0.15
        assert floor.irreducible_sigma_px < floor.total_sigma_px
        assert floor.sigma_for_propagation() == floor.irreducible_sigma_px

    def test_scale_is_linear_in_the_injected_noise(self, truth):
        small = measure_noise_floor(static_capture(truth, iid(0.05), seed=4))
        large = measure_noise_floor(static_capture(truth, iid(0.50), seed=4))
        assert large.total_sigma_px / small.total_sigma_px == pytest.approx(10.0, rel=0.05)


class TestShape:
    """The scatter ellipse, and the floor it has to beat."""

    def test_closed_form_floor_matches_simulation(self):
        rng = np.random.default_rng(11)
        for n_frames in (10, 30, 120):
            ratios = []
            for _ in range(3000):
                sample = np.cov(rng.normal(0.0, 1.0, (n_frames, 2)).T, ddof=1)
                values = np.linalg.eigvalsh(sample)
                ratios.append(np.sqrt(values[1] / values[0]))
            for quantile in (0.5, 0.9):
                assert np.quantile(ratios, quantile) == pytest.approx(
                    anisotropy_floor(n_frames, quantile), rel=0.02
                )

    def test_floor_falls_towards_one_with_more_frames(self):
        floors = [anisotropy_floor(n) for n in (10, 30, 120, 1000)]
        assert floors == sorted(floors, reverse=True)
        assert floors[0] > 1.5
        assert floors[-1] < 1.05

    def test_floor_is_infinite_when_frames_cannot_determine_it(self):
        assert anisotropy_floor(3) == float("inf")

    def test_floor_rejects_an_impossible_quantile(self):
        with pytest.raises(ValidationError):
            anisotropy_floor(30, quantile=1.0)

    @pytest.mark.parametrize("ratio", [2.0, 3.0, 5.0])
    def test_recovers_injected_axis_ratio(self, truth, ratio):
        floor = measure_noise_floor(
            static_capture(truth, anisotropic(0.1 * ratio, 0.1))
        )
        assert floor.anisotropy_median == pytest.approx(ratio, rel=0.12)
        assert floor.anisotropy_excess > ANISOTROPY_EXCESS_WARNING
        assert any(line.startswith("SHAPE") for line in floor.summary_lines())

    def test_isotropic_noise_is_not_called_anisotropic_at_any_frame_count(self, truth):
        # The whole point of the floor: a raw anisotropy of 1.5 is expected at
        # ten frames, so a fixed threshold would fire on a perfectly clean rig.
        for n_frames in (MIN_FRAMES, 15, 40, 120):
            floor = measure_noise_floor(
                static_capture(truth, iid(0.2), n_frames=n_frames, seed=n_frames)
            )
            assert floor.anisotropy_excess < ANISOTROPY_EXCESS_WARNING, n_frames
            assert not any(
                line.startswith("SHAPE") for line in floor.summary_lines()
            ), n_frames

    def test_orientation_points_along_the_injected_long_axis(self, truth):
        floor = measure_noise_floor(static_capture(truth, anisotropic(0.4, 0.08)))
        angles = np.array([corner.orientation_deg for corner in floor.corners])
        folded = np.minimum(angles, 180.0 - angles)
        assert np.median(folded) < 8.0

    def test_edge_alignment_is_at_chance_for_direction_free_noise(self, truth):
        floor = measure_noise_floor(static_capture(truth, iid(0.2)))
        assert floor.edge_aligned_fraction == pytest.approx(
            EDGE_ALIGNED_BY_CHANCE, abs=0.15
        )


class TestCorrelation:
    """The one property that actually invalidates the covariance."""

    def test_independent_noise_has_no_correlation_length(self, truth):
        floor = measure_noise_floor(static_capture(truth, iid(0.2)))
        assert floor.correlation.length_px == 0.0
        assert floor.trustworthy_covariance

    def test_the_sigma_advice_does_not_contradict_the_verdict(self, truth):
        """Telling someone to propagate a sigma it has just called unquotable.

        A correct noise scale does not rescue a covariance whose independence
        assumption has failed, so the `use` line has to say so when the
        `COVARIANCE` line is present.
        """
        clean = measure_noise_floor(static_capture(truth, iid(0.2)))
        correlated_floor = measure_noise_floor(
            static_capture(truth, correlated(truth, 0.2, 60.0))
        )

        def use_line(floor):
            return [l for l in floor.summary_lines() if l.startswith("use")][0]

        assert "COVARIANCE" not in use_line(clean)
        assert "COVARIANCE" in use_line(correlated_floor)

    @pytest.mark.parametrize("length", [40.0, 60.0, 90.0])
    def test_recovers_the_injected_correlation_length(self, truth, length):
        floor = measure_noise_floor(
            static_capture(truth, correlated(truth, 0.2, length))
        )
        assert (
            floor.total_correlation.length_px / floor.spacing_px
            > CORRELATION_SPACINGS_WARNING
        )
        assert not floor.trustworthy_covariance
        assert any(line.startswith("COVARIANCE") for line in floor.summary_lines())

    def test_longer_fields_read_as_longer_correlation(self, truth):
        short = measure_noise_floor(static_capture(truth, correlated(truth, 0.2, 40.0)))
        long = measure_noise_floor(static_capture(truth, correlated(truth, 0.2, 120.0)))
        assert long.total_correlation.length_px > short.total_correlation.length_px

    def test_correlation_saturates_at_the_board_rather_than_extrapolating(self, truth):
        # A field smoother than the board cannot have its length measured, so
        # the profile must stop at the widest separation seen instead of
        # inventing a number beyond it.
        floor = measure_noise_floor(static_capture(truth, correlated(truth, 0.2, 400.0)))
        widest = float(floor.total_correlation.separation_px.max())
        assert floor.total_correlation.length_px <= widest * 1.01

    @pytest.mark.parametrize("length", [40.0, 90.0, 250.0, 800.0])
    def test_smoothing_the_field_does_not_defeat_the_check(self, truth, length):
        # The obvious way to slip past this check is to make the field so smooth
        # that the affine removal takes all of it. It does not work: removing an
        # affine from a very smooth field leaves the residual curvature, whose
        # scale is set by the board rather than the field, so the irreducible
        # length settles near the corner spacing instead of falling to zero.
        floor = measure_noise_floor(
            static_capture(truth, correlated(truth, 0.2, length))
        )
        assert not floor.trustworthy_covariance, length
        assert floor.correlation_spacings > CORRELATION_SPACINGS_WARNING, length

    def test_the_threshold_is_in_corner_spacings_not_pixels(self):
        """The same fault must read the same on boards of different density.

        Injecting one field into a 9x6 board gives a 51 px correlation length
        and into a 21x14 board 79 px, so a fixed pixel threshold judges the two
        rigs differently. In corner spacings the denser board reads *higher*,
        which is the correct ordering: more corners share the field, so more of
        the independent observations the covariance counts are not independent.
        """
        sparse = Checkerboard(9, 6, 25.0)
        dense = Checkerboard(21, 14, 15.0)
        ratios = []
        floors = []
        for board in (sparse, dense):
            pose = pose_for_view(board, 650.0, tilt_rad=0.2)
            points = projector_for(CAMERA).project(CAMERA, pose, board.object_points())
            rng = np.random.default_rng(5)
            views = []
            for index in range(120):
                centres = rng.uniform([0, 0], IMAGE_SIZE, (6, 2))
                amplitude = rng.normal(0.0, 1.0, (6, 2))
                distance = np.linalg.norm(
                    points[:, None, :] - centres[None, :, :], axis=2
                )
                weight = np.exp(-(distance ** 2) / (2.0 * 120.0 ** 2))
                field = np.stack(
                    [weight @ amplitude[:, 0], weight @ amplitude[:, 1]], axis=1
                )
                field *= 0.2 / max(field.std(), 1e-12)
                views.append(
                    ViewObservations(
                        f"f{index:04d}", np.arange(len(points)), points + field
                    )
                )
            floor = measure_noise_floor(
                ObservationSet(board, IMAGE_SIZE, tuple(views))
            )
            assert not floor.trustworthy_covariance
            ratios.append(floor.correlation_spacings)
            floors.append(floor)

        assert ratios[1] > ratios[0]
        assert ratios[1] >= CORRELATION_SPACINGS_CRITICAL
        assert ratios[0] < CORRELATION_SPACINGS_CRITICAL
        # And the critical tier says something different from the warning tier.
        assert ratios[0] > CORRELATION_SPACINGS_WARNING

        warning, critical = (
            [l for l in floor.summary_lines() if l.startswith("COVARIANCE")][0]
            for floor in floors
        )
        assert "no interval from this capture is quotable" in critical
        assert "no interval from this capture is quotable" not in warning
        assert "RobustCovariance" in warning
        # The severe message names candidate causes; the milder one does not,
        # because at one corner spacing more views is often the whole answer.
        assert "rolling shutter" in critical

    def test_the_spacing_matches_the_projected_geometry(self, truth):
        """`spacing_px` is the unit everything else is judged in, so check it.

        A 25 mm square at 700 mm through a 900 px focal length subtends about
        900 * 25 / 700 = 32 px, and the board is tilted so the real figure sits
        a little either side of that.
        """
        floor = measure_noise_floor(static_capture(truth, iid(0.2)))
        expected = CAMERA.fx * BOARD.square_size / 700.0
        assert floor.spacing_px == pytest.approx(expected, rel=0.15)

        # And it really is the nearest-neighbour distance, not some average.
        positions = np.array([corner.mean_px for corner in floor.corners])
        gaps = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=2)
        np.fill_diagonal(gaps, np.inf)
        assert floor.spacing_px == pytest.approx(np.median(gaps.min(axis=1)))

    def test_profile_bins_are_ordered_and_populated(self, truth):
        profile = measure_noise_floor(static_capture(truth, iid(0.2))).correlation
        assert profile.n_bins > 1
        assert np.all(np.diff(profile.separation_px) > 0)
        assert np.all(profile.pair_counts > 0)


class TestRigMovement:
    """Whole-frame movement, which is a mount problem and not a noise problem."""

    def test_drift_is_detected_and_reported(self, truth):
        floor = measure_noise_floor(static_capture(truth, drifting(0.2, 2.0)))
        assert floor.drift_px == pytest.approx(1.0, rel=0.15)
        assert floor.absorbed_fraction > 0.5
        assert any(line.startswith("RIG") for line in floor.summary_lines())

    def test_drift_leaves_the_irreducible_scale_untouched(self, truth):
        clean = measure_noise_floor(static_capture(truth, iid(0.2), seed=9))
        drifted = measure_noise_floor(static_capture(truth, drifting(0.2, 2.0), seed=9))
        assert drifted.total_sigma_px > clean.total_sigma_px * 1.5
        assert drifted.irreducible_sigma_px == pytest.approx(
            clean.irreducible_sigma_px, rel=0.05
        )

    def test_drift_does_not_masquerade_as_detector_anisotropy(self, truth):
        # Drift is along x only, so the raw ellipse is elongated in x for a
        # reason that has nothing to do with the detector.
        floor = measure_noise_floor(static_capture(truth, drifting(0.2, 2.0)))
        assert floor.anisotropy_excess < ANISOTROPY_EXCESS_WARNING

        def ratio(matrix):
            values = np.linalg.eigvalsh(matrix)
            return float(np.sqrt(values[1] / values[0]))

        raw = np.median([ratio(c.covariance) for c in floor.corners])
        reported = np.median([c.anisotropy for c in floor.corners])
        assert raw > 3.0
        assert reported == pytest.approx(floor.anisotropy_floor, rel=0.15)

    def test_drift_does_not_masquerade_as_a_correlated_field(self, truth):
        floor = measure_noise_floor(static_capture(truth, drifting(0.2, 2.0)))
        assert floor.correlation.length_px == 0.0
        assert floor.trustworthy_covariance


class TestGuards:
    """What the tool refuses, and why."""

    def test_refuses_too_few_frames(self, truth):
        with pytest.raises(ValidationError, match="frames"):
            measure_noise_floor(
                static_capture(truth, iid(0.2), n_frames=MIN_FRAMES - 1)
            )

    def test_accepts_exactly_the_minimum(self, truth):
        floor = measure_noise_floor(
            static_capture(truth, iid(0.2), n_frames=MIN_FRAMES)
        )
        assert floor.n_frames == MIN_FRAMES
        assert any(line.startswith("CAUTION") for line in floor.summary_lines())

    def test_refuses_scatter_that_cannot_be_a_static_scene(self, truth):
        with pytest.raises(ValidationError, match="motion"):
            measure_noise_floor(static_capture(truth, iid(12.0)))

    def test_drops_corners_seen_too_rarely(self, truth):
        # Half the board vanishes after the first few frames, as it would if
        # something moved into the light.
        def presence(rng, index):
            keep = np.ones(BOARD.num_points, dtype=bool)
            if index > 3:
                keep[: BOARD.num_points // 2] = False
            return keep

        floor = measure_noise_floor(
            static_capture(truth, iid(0.2), presence=presence)
        )
        assert floor.n_corners == BOARD.num_points // 2

    @pytest.mark.parametrize(
        "kwargs", [{"min_presence": 1.5}, {"min_presence": 0.0},
                   {"min_presence": -0.1}, {"n_bins": 1}]
    )
    def test_refuses_arguments_outside_their_range(self, truth, kwargs):
        with pytest.raises(ValidationError):
            measure_noise_floor(static_capture(truth, iid(0.2)), **kwargs)

    def test_accepts_demanding_every_frame(self, truth):
        floor = measure_noise_floor(
            static_capture(truth, iid(0.2)), min_presence=1.0
        )
        assert floor.n_corners == BOARD.num_points

    def test_says_how_many_corners_it_discarded(self, truth):
        """A tool about hidden assumptions must not hide its own exclusions.

        `MIN_PRESENCE` admits a corner seen in 90% of frames, but a `(2, 2)`
        covariance still needs `MIN_FRAMES` of them, so at exactly ten frames a
        corner missing one is admitted by the filter and then dropped. That is
        correct, and it has to be said out loud.
        """
        def presence(rng, index):
            keep = np.ones(BOARD.num_points, dtype=bool)
            keep[index % BOARD.num_points] = False
            return keep

        floor = measure_noise_floor(
            static_capture(truth, iid(0.2), n_frames=MIN_FRAMES, presence=presence)
        )
        assert floor.n_detected == BOARD.num_points
        assert floor.n_corners == BOARD.num_points - MIN_FRAMES
        first = floor.summary_lines()[0]
        assert f"of {BOARD.num_points} detected" in first
        assert "seen too rarely to measure" in first

        # And it stays quiet when it discarded nothing.
        clean = measure_noise_floor(static_capture(truth, iid(0.2)))
        assert clean.n_detected == clean.n_corners
        assert "detected" not in clean.summary_lines()[0]

    def test_refuses_when_no_corner_is_seen_often_enough(self, truth):
        """Two different refusals, depending on where the shortfall is caught.

        A demanding `min_presence` rejects the capture on the presence filter. A
        lax one lets every corner through, and they are then all dropped for
        having too few frames to form a covariance from. Both must say which it
        was rather than return an empty measurement.
        """
        def presence(rng, index):
            keep = np.zeros(BOARD.num_points, dtype=bool)
            keep[index % BOARD.num_points] = True
            return keep

        observations = static_capture(truth, iid(0.2), n_frames=40, presence=presence)
        with pytest.raises(ValidationError, match="at least 90%"):
            measure_noise_floor(observations)
        with pytest.raises(ValidationError, match="after filtering"):
            measure_noise_floor(observations, min_presence=0.01)


class TestSerialisation:
    """The JSON the CLI writes."""

    def test_round_trips_through_json(self, truth):
        import json

        floor = measure_noise_floor(static_capture(truth, iid(0.2)))
        loaded = json.loads(json.dumps(floor.to_dict()))
        assert loaded["n_frames"] == floor.n_frames
        assert loaded["n_corners"] == floor.n_corners
        assert loaded["n_detected"] == floor.n_detected
        assert loaded["total_sigma_px"] == pytest.approx(floor.total_sigma_px)
        assert loaded["anisotropy_floor"] == pytest.approx(floor.anisotropy_floor)
        assert loaded["spacing_px"] == pytest.approx(floor.spacing_px)
        assert loaded["correlation_spacings"] == pytest.approx(
            floor.correlation_spacings
        )
        assert loaded["trustworthy_covariance"] is True
        assert len(loaded["corners"]) == floor.n_corners
        assert loaded["correlation"]["length_px"] == floor.correlation.length_px

    def test_every_number_is_finite(self, truth):
        floor = measure_noise_floor(static_capture(truth, iid(0.2)))

        def walk(value):
            if isinstance(value, dict):
                for item in value.values():
                    walk(item)
            elif isinstance(value, list):
                for item in value:
                    walk(item)
            elif isinstance(value, float):
                assert np.isfinite(value)

        walk(floor.to_dict())


def test_correlation_is_not_biased_low_by_unequal_presence():
    """Each pair normalises over the frames it shares, not over all frames.

    Two corners detected in different frames share fewer observations than
    either has alone. Normalising by each corner's full power would scale the
    correlation down by the square root of the overlap and report a perfectly
    shared field as partly independent.
    """
    from calibsense.noisefloor import _correlation_profile

    rng = np.random.default_rng(0)
    n_frames, n_corners = 100, 6
    field = rng.normal(0.0, 1.0, (n_frames, 1, 2)) * np.ones((1, n_corners, 1))
    field[60:, 0, :] = np.nan
    means = np.column_stack([np.arange(n_corners) * 30.0, np.zeros(n_corners)])

    profile = _correlation_profile(field, means, n_bins=4)
    assert np.all(profile.correlation > 0.99)


class TestCrossing:
    """The interpolation that turns a profile into the headline length.

    Tested directly because it is a pure function with known answers, and
    because everything the tool concludes about the covariance rests on it.
    """

    def crossing(self, x, y):
        from calibsense.noisefloor import CORRELATION_THRESHOLD, _crossing

        return _crossing(np.asarray(x, float), np.asarray(y, float), CORRELATION_THRESHOLD)

    def test_interpolates_between_the_bracketing_bins(self):
        level = float(np.exp(-1.0))
        # Halfway between a bin at 1.0 and a bin at 0.0 in correlation puts the
        # 1/e crossing at 1/e of the way along in separation.
        assert self.crossing([0.0, 100.0], [1.0, 0.0]) == pytest.approx(
            100.0 * (1.0 - level), rel=1e-9
        )

    def test_returns_the_exact_separation_when_a_bin_sits_on_the_level(self):
        level = float(np.exp(-1.0))
        assert self.crossing([10.0, 40.0, 90.0], [0.9, level, 0.1]) == pytest.approx(40.0)

    def test_reads_zero_when_the_first_bin_is_already_below(self):
        assert self.crossing([20.0, 60.0], [0.2, 0.1]) == 0.0

    def test_saturates_at_the_last_bin_when_it_never_falls(self):
        # A field smoother than the board: the length is at least the widest
        # separation, and must not be extrapolated past it.
        assert self.crossing([20.0, 60.0, 200.0], [0.99, 0.95, 0.9]) == 200.0

    def test_handles_an_empty_profile(self):
        assert self.crossing([], []) == 0.0

    def test_handles_a_single_bin(self):
        level = float(np.exp(-1.0))
        assert self.crossing([50.0], [0.9]) == 50.0
        assert self.crossing([50.0], [level / 2]) == 0.0


def test_an_unusable_profile_is_empty_rather_than_wrong():
    """No corner pair shares enough frames, so there is nothing to bin.

    Reachable only below the entry point's own guards, but the profile has to
    degrade to "measured nothing" rather than to a fabricated length.
    """
    from calibsense.noisefloor import _correlation_profile

    deviations = np.full((MIN_FRAMES, 3, 2), np.nan)
    means = np.array([[0.0, 0.0], [30.0, 0.0], [60.0, 0.0]])
    profile = _correlation_profile(deviations, means, n_bins=4)
    assert profile.n_bins == 0
    assert profile.length_px == 0.0
    assert profile.separation_px.size == 0
