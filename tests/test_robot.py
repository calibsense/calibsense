"""Reading robot poses for hand-eye."""

from __future__ import annotations

import json

import numpy as np
import pytest

from caltrust.core.poses import Pose, quaternion_to_matrix
from caltrust.errors import UnsupportedFormatError, ValidationError
from caltrust.ingest.robot import read_robot_poses, write_robot_poses

QUATERNION = [0.0, 0.0, 0.3826834, 0.9238795]  # 45 degrees about z, (x, y, z, w)


def test_json_matrix_form(tmp_path):
    pose = Pose.from_rvec_tvec([0.1, 0.2, 0.3], [10.0, 20.0, 500.0])
    path = tmp_path / "r.json"
    path.write_text(json.dumps({
        "format": "caltrust.robot_poses",
        "poses": [{"view_id": "a", "matrix": pose.matrix.tolist()}],
    }))
    poses = read_robot_poses(str(path))
    assert poses.view_ids == ("a",)
    assert poses.poses[0].allclose(pose, atol=1e-9)


def test_json_quaternion_form(tmp_path):
    path = tmp_path / "r.json"
    path.write_text(json.dumps({
        "poses": [{"view_id": "a", "translation": [1.0, 2.0, 3.0], "quaternion": QUATERNION}],
    }))
    poses = read_robot_poses(str(path))
    assert np.allclose(poses.poses[0].rotation, quaternion_to_matrix(QUATERNION))
    assert poses.poses[0].translation.tolist() == [1.0, 2.0, 3.0]


def test_json_rotation_vector_form(tmp_path):
    path = tmp_path / "r.json"
    path.write_text(json.dumps({
        "poses": [{"view_id": "a", "tvec": [1.0, 2.0, 3.0], "rvec": [0.0, 0.0, 0.5]}],
    }))
    assert read_robot_poses(str(path)).poses[0].rvec[2] == pytest.approx(0.5)


def test_json_rotation_matrix_form(tmp_path):
    path = tmp_path / "r.json"
    path.write_text(json.dumps({
        "poses": [{"view_id": "a", "position": [1.0, 2.0, 3.0],
                   "rotation": np.eye(3).tolist()}],
    }))
    assert read_robot_poses(str(path)).poses[0].allclose(
        Pose(np.eye(3), [1.0, 2.0, 3.0]), atol=1e-12
    )


def test_a_bare_json_list_is_accepted(tmp_path):
    path = tmp_path / "r.json"
    path.write_text(json.dumps([{"matrix": np.eye(4).tolist()}]))
    poses = read_robot_poses(str(path))
    assert poses.view_ids == ("view0000",)


def test_units_convert_translations(tmp_path):
    path = tmp_path / "r.json"
    path.write_text(json.dumps({
        "units": "m",
        "poses": [{"view_id": "a", "matrix": Pose(np.eye(3), [0.5, 0, 0.8]).matrix.tolist()}],
    }))
    assert read_robot_poses(str(path)).poses[0].translation[2] == pytest.approx(800.0)


def test_the_file_convention_overrides_the_argument(tmp_path):
    pose = Pose.from_rvec_tvec([0.3, 0.0, 0.0], [10.0, 0.0, 500.0])
    path = tmp_path / "r.json"
    path.write_text(json.dumps({
        "convention": "base2gripper",
        "poses": [{"view_id": "a", "matrix": pose.matrix.tolist()}],
    }))
    poses = read_robot_poses(str(path), convention="gripper2base")
    assert poses.poses[0].allclose(pose.inverse(), atol=1e-9)


def test_csv_quaternion_form(tmp_path):
    path = tmp_path / "r.csv"
    path.write_text(
        "view_id,x,y,z,qx,qy,qz,qw\n"
        f"a,1,2,3,{QUATERNION[0]},{QUATERNION[1]},{QUATERNION[2]},{QUATERNION[3]}\n"
        f"b,4,5,6,0,0,0,1\n"
    )
    poses = read_robot_poses(str(path))
    assert poses.view_ids == ("a", "b")
    assert np.allclose(poses.poses[1].rotation, np.eye(3))


def test_csv_rotation_vector_form_and_alternate_column_names(tmp_path):
    path = tmp_path / "r.csv"
    path.write_text("image,tx,ty,tz,rx,ry,rz\nfrm1,1,2,3,0,0,0.5\n")
    poses = read_robot_poses(str(path))
    assert poses.view_ids == ("frm1",)
    assert poses.poses[0].rvec[2] == pytest.approx(0.5)


def test_csv_is_case_insensitive_about_headers(tmp_path):
    path = tmp_path / "r.csv"
    path.write_text("View_ID,X,Y,Z,QX,QY,QZ,QW\na,1,2,3,0,0,0,1\n")
    assert read_robot_poses(str(path)).view_ids == ("a",)


def test_tsv_is_accepted(tmp_path):
    path = tmp_path / "r.tsv"
    path.write_text("view_id\tx\ty\tz\trx\try\trz\na\t1\t2\t3\t0\t0\t0\n")
    assert read_robot_poses(str(path)).view_ids == ("a",)


def test_csv_without_a_view_column_says_so(tmp_path):
    path = tmp_path / "r.csv"
    path.write_text("x,y,z,rx,ry,rz\n1,2,3,0,0,0\n")
    with pytest.raises(UnsupportedFormatError, match="no view id column"):
        read_robot_poses(str(path))


def test_csv_without_position_columns_says_so(tmp_path):
    path = tmp_path / "r.csv"
    path.write_text("view_id,rx,ry,rz\na,0,0,0\n")
    with pytest.raises(UnsupportedFormatError, match="no position columns"):
        read_robot_poses(str(path))


def test_csv_without_orientation_columns_says_so(tmp_path):
    path = tmp_path / "r.csv"
    path.write_text("view_id,x,y,z\na,1,2,3\n")
    with pytest.raises(UnsupportedFormatError, match="no orientation columns"):
        read_robot_poses(str(path))


def test_csv_with_no_data_rows_says_so(tmp_path):
    path = tmp_path / "r.csv"
    path.write_text("view_id,x,y,z,rx,ry,rz\n")
    with pytest.raises(UnsupportedFormatError, match="no data rows"):
        read_robot_poses(str(path))


def test_csv_with_an_unparseable_number_names_the_line(tmp_path):
    path = tmp_path / "r.csv"
    path.write_text("view_id,x,y,z,rx,ry,rz\na,1,2,three,0,0,0\n")
    with pytest.raises(ValidationError, match=":2"):
        read_robot_poses(str(path))


def test_a_bad_matrix_is_rejected(tmp_path):
    path = tmp_path / "r.json"
    path.write_text(json.dumps({"poses": [{"matrix": np.eye(3).tolist()}]}))
    with pytest.raises(ValidationError, match="4x4"):
        read_robot_poses(str(path))


def test_a_non_rotation_matrix_is_rejected(tmp_path):
    broken = np.eye(4)
    broken[0, 0] = 2.0
    path = tmp_path / "r.json"
    path.write_text(json.dumps({"poses": [{"matrix": broken.tolist()}]}))
    with pytest.raises(ValidationError, match="orthonormal"):
        read_robot_poses(str(path))


def test_an_entry_without_an_orientation_is_rejected(tmp_path):
    path = tmp_path / "r.json"
    path.write_text(json.dumps({"poses": [{"translation": [1, 2, 3]}]}))
    with pytest.raises(ValidationError, match="quaternion"):
        read_robot_poses(str(path))


def test_an_entry_without_anything_is_rejected(tmp_path):
    path = tmp_path / "r.json"
    path.write_text(json.dumps({"poses": [{"view_id": "a"}]}))
    with pytest.raises(ValidationError, match="'matrix'"):
        read_robot_poses(str(path))


def test_an_unknown_extension_is_refused(tmp_path):
    path = tmp_path / "r.xlsx"
    path.write_bytes(b"x")
    with pytest.raises(UnsupportedFormatError, match="expected .json or .csv"):
        read_robot_poses(str(path))


def test_a_missing_file_says_so(tmp_path):
    with pytest.raises(UnsupportedFormatError, match="no such robot pose file"):
        read_robot_poses(str(tmp_path / "absent.json"))


def test_an_unknown_convention_is_rejected(tmp_path):
    path = tmp_path / "r.json"
    path.write_text(json.dumps({"poses": [{"matrix": np.eye(4).tolist()}]}))
    with pytest.raises(ValidationError, match="unknown hand-eye convention"):
        read_robot_poses(str(path), convention="world2cam")


def test_write_then_read_round_trips(tmp_path):
    pose = Pose.from_rvec_tvec([0.1, -0.2, 0.3], [10.0, 20.0, 500.0])
    path = tmp_path / "r.json"
    original = read_robot_poses(
        str(_write_json(path, [{"view_id": "a", "matrix": pose.matrix.tolist()}]))
    )
    out = write_robot_poses(original, str(tmp_path / "out.json"))
    back = read_robot_poses(out)
    assert back.view_ids == original.view_ids
    assert back.poses[0].allclose(original.poses[0], atol=1e-9)


def _write_json(path, poses):
    path.write_text(json.dumps({"poses": poses}))
    return path
