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

"""The session bundle: detections, an existing calibration, robot poses."""

from __future__ import annotations

import numpy as np
import pytest

from caltrust.core.camera import PinholeBrownConrady
from caltrust.core.poses import Pose
from caltrust.core.session import (
    HAND_EYE_CONVENTIONS,
    CalibrationRecord,
    CalibrationSession,
    RobotPoses,
)
from caltrust.errors import ValidationError


def robot_poses(view_ids, convention="gripper2base"):
    matrices = [
        Pose.from_rvec_tvec([0.1 * i, 0.0, 0.0], [10.0 * i, 0.0, 500.0]).matrix
        for i in range(len(view_ids))
    ]
    return RobotPoses.from_matrices(matrices, view_ids, convention)


def test_record_round_trip(pinhole):
    record = CalibrationRecord(pinhole, (1280, 720), "a.yml", 0.24, {"k": "v"})
    back = CalibrationRecord.from_dict(record.to_dict())
    assert back.camera == pinhole
    assert (back.image_size, back.source, back.reported_rms) == ((1280, 720), "a.yml", 0.24)
    assert back.metadata == {"k": "v"}


@pytest.mark.parametrize("rms", [-0.1, np.nan, np.inf])
def test_record_rejects_a_bad_rms(pinhole, rms):
    with pytest.raises(ValidationError, match="reported_rms"):
        CalibrationRecord(pinhole, (1280, 720), reported_rms=rms)


def test_record_rejects_a_non_positive_image_size(pinhole):
    with pytest.raises(ValidationError, match="image size"):
        CalibrationRecord(pinhole, (0, 720))


def test_base2gripper_is_inverted_on_the_way_in():
    forward = robot_poses(["a", "b"], "gripper2base")
    reverse = robot_poses(["a", "b"], "base2gripper")
    assert forward.poses[1].allclose(reverse.poses[1].inverse(), atol=1e-9)


def test_unknown_convention_is_rejected():
    with pytest.raises(ValidationError, match="unknown hand-eye convention"):
        RobotPoses.from_matrices([np.eye(4)], ["a"], "camera2world")
    assert set(HAND_EYE_CONVENTIONS) == {"gripper2base", "base2gripper"}


def test_pose_and_view_counts_must_agree():
    with pytest.raises(ValidationError, match="robot poses for"):
        RobotPoses((Pose.identity(),), ("a", "b"))


def test_empty_robot_poses_are_rejected():
    with pytest.raises(ValidationError, match="cannot be empty"):
        RobotPoses((), ())


def test_repeated_view_id_in_robot_poses_is_rejected():
    with pytest.raises(ValidationError, match="repeated"):
        RobotPoses((Pose.identity(), Pose.identity()), ("a", "a"))


def test_aligned_with_reorders_to_view_order(tiny_observations):
    shuffled = robot_poses(["v2", "v0", "v1"])
    aligned = shuffled.aligned_with(tiny_observations)
    # poses were built in ["v2", "v0", "v1"] order, so v0 gets index 1 and
    # v2 gets index 0 once realigned to the observation order.
    assert aligned[0].allclose(shuffled.poses[1], atol=1e-12)
    assert aligned[1].allclose(shuffled.poses[2], atol=1e-12)
    assert aligned[2].allclose(shuffled.poses[0], atol=1e-12)


def test_a_missing_robot_pose_is_reported_at_ingest(tiny_observations):
    partial = robot_poses(["v0", "v1"])
    with pytest.raises(ValidationError, match="no robot pose"):
        CalibrationSession(observations=tiny_observations, robot=partial)


def test_session_fills_in_a_timestamp(good_session):
    assert good_session.created.endswith("+00:00")
    assert good_session.caltrust_version


def test_session_rejects_a_prior_for_a_different_image_size(good_capture, pinhole):
    prior = CalibrationRecord(pinhole, (640, 480), "a.yml")
    with pytest.raises(ValidationError, match="prior calibration is for"):
        CalibrationSession(observations=good_capture.observations, prior=prior)


def test_session_shortcuts_reach_through_the_observations(good_session):
    assert good_session.target is good_session.observations.target
    assert good_session.image_size == good_session.observations.image_size
    assert good_session.has_hand_eye is False


def test_summary_mentions_the_prior_and_its_rms(session_with_prior):
    text = "\n".join(session_with_prior.summary_lines())
    assert "shipped.yml" in text and "0.2412" in text


def test_summary_mentions_robot_poses(tiny_observations):
    session = CalibrationSession(
        observations=tiny_observations, robot=robot_poses(["v0", "v1", "v2"])
    )
    assert session.has_hand_eye
    assert any("robot poses" in line for line in session.summary_lines())
