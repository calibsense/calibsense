"""The synthetic capture generator."""

from __future__ import annotations

import numpy as np
import pytest

from caltrust.errors import ValidationError
from caltrust.refit.projection import projector_for
from caltrust.synthetic import (
    diverse_poses,
    frontoparallel_poses,
    pose_for_view,
    synthesise,
)


def test_pose_places_the_board_centre_at_the_requested_distance(checkerboard):
    pose = pose_for_view(checkerboard, 750.0, tilt_rad=0.4, roll_rad=1.0)
    centre = checkerboard.object_points().mean(axis=0)
    assert pose.apply(centre.reshape(1, 3))[0, 2] == pytest.approx(750.0)


def test_lateral_offset_moves_the_centre_sideways(checkerboard):
    pose = pose_for_view(checkerboard, 700.0, offset_mm=(50.0, -30.0))
    centre = checkerboard.object_points().mean(axis=0)
    moved = pose.apply(centre.reshape(1, 3))[0]
    assert moved[0] == pytest.approx(50.0)
    assert moved[1] == pytest.approx(-30.0)


def test_tilt_rotates_the_board_normal(checkerboard):
    pose = pose_for_view(checkerboard, 700.0, tilt_rad=0.5, tilt_axis_rad=0.0)
    normal = pose.rotation @ np.array([0.0, 0.0, 1.0])
    assert np.arccos(abs(normal[2])) == pytest.approx(0.5)


@pytest.mark.parametrize("distance", [0.0, -1.0, np.nan])
def test_a_non_positive_distance_is_rejected(checkerboard, distance):
    with pytest.raises(ValidationError, match="distance must be positive"):
        pose_for_view(checkerboard, distance)


def test_frontoparallel_poses_share_one_depth(checkerboard):
    poses = frontoparallel_poses(checkerboard, 12, 800.0, seed=0)
    depths = np.array([p.translation[2] for p in poses])
    assert len(poses) == 12
    assert depths.std() < 5.0
    tilts = [np.linalg.norm(p.rvec[:2]) for p in poses]
    assert max(tilts) < 0.05


def test_diverse_poses_cycle_the_requested_distances(checkerboard):
    poses = diverse_poses(checkerboard, 9, distances_mm=(400.0, 800.0, 1200.0), seed=0)
    depths = np.array([p.translation[2] for p in poses])
    assert depths.min() < 500.0 and depths.max() > 1100.0


def test_diverse_poses_need_a_distance(checkerboard):
    with pytest.raises(ValidationError, match="at least one working distance"):
        diverse_poses(checkerboard, 4, distances_mm=())


def test_noiseless_capture_reprojects_exactly(pinhole, checkerboard):
    capture = synthesise(
        pinhole, checkerboard, diverse_poses(checkerboard, 6, seed=1),
        (1280, 720), noise_px=0.0, seed=0,
    )
    projector = projector_for(pinhole)
    for view, pose in zip(capture.observations.views, capture.poses):
        predicted = projector.project(
            pinhole, pose, view.object_points(checkerboard)
        )
        assert np.abs(predicted - view.image_points).max() < 1e-9


def test_noise_appears_at_the_requested_scale(pinhole, checkerboard):
    poses = diverse_poses(checkerboard, 10, seed=1)
    clean = synthesise(pinhole, checkerboard, poses, (1280, 720), noise_px=0.0)
    noisy = synthesise(pinhole, checkerboard, poses, (1280, 720), noise_px=0.5, seed=3)
    assert clean.observations.n_views == noisy.observations.n_views
    differences = np.concatenate([
        b.image_points - a.image_points
        for a, b in zip(clean.observations.views, noisy.observations.views)
    ])
    assert differences.std() == pytest.approx(0.5, rel=0.15)


def test_capture_is_deterministic_for_a_seed(pinhole, checkerboard):
    poses = diverse_poses(checkerboard, 6, seed=1)
    first = synthesise(pinhole, checkerboard, poses, (1280, 720), noise_px=0.3, seed=5)
    second = synthesise(pinhole, checkerboard, poses, (1280, 720), noise_px=0.3, seed=5)
    assert np.array_equal(
        first.observations.views[0].image_points, second.observations.views[0].image_points
    )


def test_points_outside_the_frame_are_dropped(pinhole, checkerboard):
    # Close in and pushed well off centre, so part of the board leaves the frame
    # even after the barrel distortion pulls the periphery back inwards.
    poses = [pose_for_view(checkerboard, 250.0, offset_mm=(160.0, 100.0))]
    kept = synthesise(pinhole, checkerboard, poses, (1280, 720), min_points=4)
    assert kept.observations.total_points < checkerboard.num_points
    assert kept.observations.out_of_frame() == 0


def test_keeping_out_of_frame_points_disables_the_visibility_test(pinhole, checkerboard):
    poses = [pose_for_view(checkerboard, 250.0, offset_mm=(160.0, 100.0))]
    kept = synthesise(
        pinhole, checkerboard, poses, (1280, 720), min_points=4, keep_out_of_frame=True
    )
    assert kept.observations.total_points == checkerboard.num_points
    assert kept.observations.out_of_frame() > 0


def test_a_view_with_too_few_visible_points_is_dropped_whole(pinhole, checkerboard):
    poses = diverse_poses(checkerboard, 4, seed=1) + [
        pose_for_view(checkerboard, 200.0, offset_mm=(400.0, 400.0))
    ]
    capture = synthesise(pinhole, checkerboard, poses, (1280, 720), min_points=20)
    assert capture.observations.n_views == 4
    assert len(capture.poses) == 4
    assert capture.observations.summary.attempted == 5


def test_a_board_behind_the_camera_is_skipped(pinhole, checkerboard):
    from caltrust.core.poses import Pose
    from caltrust.core.poses import rotvec_to_matrix

    behind = Pose(rotvec_to_matrix([0.0, 0.0, 0.0]), [0.0, 0.0, -800.0])
    good = diverse_poses(checkerboard, 3, seed=2)
    capture = synthesise(pinhole, checkerboard, [behind] + good, (1280, 720))
    assert capture.observations.n_views == 3


def test_no_usable_view_is_an_error(pinhole, checkerboard):
    poses = [pose_for_view(checkerboard, 100.0, offset_mm=(5000.0, 5000.0))]
    with pytest.raises(ValidationError, match="no view kept"):
        synthesise(pinhole, checkerboard, poses, (1280, 720), min_points=20)


def test_fisheye_capture_works_on_a_charuco_board(fisheye, charuco):
    capture = synthesise(
        fisheye, charuco,
        diverse_poses(charuco, 8, distances_mm=(300.0, 600.0), max_tilt_rad=0.7, seed=3),
        (1280, 720), noise_px=0.1, seed=4,
    )
    assert capture.observations.n_views == 8
    assert capture.observations.target is charuco
    assert capture.noise_px == pytest.approx(0.1)


def test_capture_records_its_truth(good_capture, pinhole):
    assert good_capture.camera is pinhole
    assert len(good_capture.poses) == good_capture.observations.n_views
    assert good_capture.observations.summary.detector.startswith("synthetic:")
