"""Detected points and their accounting."""

from __future__ import annotations

import numpy as np
import pytest

from caltrust.core.observations import (
    DetectionFailure,
    DetectionSummary,
    ObservationSet,
    ViewObservations,
)
from caltrust.core.target import Checkerboard
from caltrust.errors import ValidationError


def view(view_id="v", ids=(0, 1, 2, 3), points=None):
    points = points if points is not None else np.arange(len(ids) * 2).reshape(-1, 2)
    return ViewObservations(view_id, np.asarray(ids), np.asarray(points, dtype=float))


def test_view_reports_its_size_and_centroid():
    v = view(points=[[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]])
    assert v.n_points == 4
    assert v.centroid().tolist() == [5.0, 5.0]


def test_view_area_fraction_is_the_bounding_box_share():
    v = view(points=[[0.0, 0.0], [100.0, 0.0], [0.0, 50.0], [100.0, 50.0]])
    assert v.area_fraction((200, 100)) == pytest.approx(0.25)


def test_area_fraction_is_clipped_to_one():
    v = view(points=[[0.0, 0.0], [400.0, 0.0], [0.0, 400.0], [400.0, 400.0]])
    assert v.area_fraction((100, 100)) == 1.0


def test_view_object_points_follow_its_ids():
    target = Checkerboard(4, 3, 10.0)
    v = view(ids=(5, 2, 9, 0))
    assert np.array_equal(v.object_points(target), target.object_points([5, 2, 9, 0]))


def test_mismatched_id_and_point_counts_are_rejected():
    with pytest.raises(ValidationError, match="point ids for"):
        ViewObservations("v", [0, 1], [[0.0, 0.0]])


def test_empty_view_is_rejected():
    with pytest.raises(ValidationError, match="no points"):
        ViewObservations("v", [], np.zeros((0, 2)))


def test_repeated_point_id_is_rejected():
    with pytest.raises(ValidationError, match="repeats a point id"):
        ViewObservations("v", [1, 1], [[0.0, 0.0], [1.0, 1.0]])


def test_negative_point_id_is_rejected():
    with pytest.raises(ValidationError, match="negative point id"):
        ViewObservations("v", [-1, 0], [[0.0, 0.0], [1.0, 1.0]])


def test_non_finite_image_points_are_rejected():
    with pytest.raises(ValidationError, match="non-finite"):
        ViewObservations("v", [0, 1], [[0.0, 0.0], [np.nan, 1.0]])


def test_metadata_is_copied_not_aliased():
    metadata = {"a": 1}
    v = ViewObservations("v", [0], [[0.0, 0.0]], metadata=metadata)
    metadata["a"] = 2
    assert v.metadata == {"a": 1}


def test_summary_counts_successes_and_rate():
    summary = DetectionSummary(10, (DetectionFailure("a.png", "blur"),), "checkerboard")
    assert summary.succeeded == 9
    assert summary.success_rate == pytest.approx(0.9)
    assert DetectionSummary().success_rate == 0.0


def test_summary_dict_round_trip():
    summary = DetectionSummary(4, (DetectionFailure("a.png", "blur"),), "charuco")
    back = DetectionSummary.from_dict(summary.to_dict())
    assert back == summary


def test_observation_set_totals(tiny_observations):
    assert tiny_observations.n_views == 3
    assert tiny_observations.total_points == 3 * 54
    assert tiny_observations.points_per_view().tolist() == [54, 54, 54]
    assert tiny_observations.view_ids == ("v0", "v1", "v2")


def test_empty_observation_set_is_rejected(checkerboard):
    with pytest.raises(ValidationError, match="at least one view"):
        ObservationSet(checkerboard, (640, 480), ())


def test_duplicate_view_ids_are_rejected(checkerboard):
    v = ViewObservations("same", [0, 1, 2, 3], np.zeros((4, 2)))
    with pytest.raises(ValidationError, match="duplicate view id"):
        ObservationSet(checkerboard, (640, 480), (v, v))


def test_point_id_beyond_the_target_is_caught_at_ingest(checkerboard):
    v = ViewObservations("v", [0, 9999], np.zeros((2, 2)))
    with pytest.raises(ValidationError, match="out of range"):
        ObservationSet(checkerboard, (640, 480), (v,))


@pytest.mark.parametrize("size", [(0, 480), (640, 0), (-1, 480)])
def test_non_positive_image_size_is_rejected(checkerboard, size):
    v = ViewObservations("v", [0, 1, 2, 3], np.zeros((4, 2)))
    with pytest.raises(ValidationError, match="image size must be positive"):
        ObservationSet(checkerboard, size, (v,))


def test_index_of_finds_a_view(tiny_observations):
    assert tiny_observations.index_of("v1") == 1
    with pytest.raises(ValidationError, match="no view with id"):
        tiny_observations.index_of("nope")


def test_select_keeps_order_and_rejects_bad_indices(tiny_observations):
    assert tiny_observations.select([2, 0]).view_ids == ("v2", "v0")
    with pytest.raises(ValidationError, match="out of range"):
        tiny_observations.select([0, 7])


def test_filter_min_points_drops_thin_views(checkerboard):
    fat = ViewObservations("fat", np.arange(20), np.zeros((20, 2)))
    thin = ViewObservations("thin", np.arange(5), np.zeros((5, 2)))
    observations = ObservationSet(checkerboard, (640, 480), (fat, thin))
    assert observations.filter_min_points(10).view_ids == ("fat",)


def test_filter_min_points_refuses_to_empty_the_set(checkerboard):
    thin = ViewObservations("thin", np.arange(5), np.zeros((5, 2)))
    observations = ObservationSet(checkerboard, (640, 480), (thin,))
    with pytest.raises(ValidationError, match="no view has at least"):
        observations.filter_min_points(10)


def test_out_of_frame_counts_points_beyond_the_edges(checkerboard):
    points = np.array([[10.0, 10.0], [-1.0, 5.0], [700.0, 5.0], [5.0, 500.0]])
    v = ViewObservations("v", [0, 1, 2, 3], points)
    observations = ObservationSet(checkerboard, (640, 480), (v,))
    assert observations.out_of_frame() == 3
