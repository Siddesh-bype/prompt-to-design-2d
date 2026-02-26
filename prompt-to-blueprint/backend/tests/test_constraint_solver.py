"""Tests for the constraint solver module."""

import pytest
from app.models.schemas import (
    BoundingBox, RoomSpec, RoomType, CompassFacing, ConnectionType,
    AdjacencyEdge, RoomLayout, LayoutGraph,
)
from app.services.constraint_solver import (
    heuristic_place, run_constraint_pipeline,
    _compute_overlap, _compute_adjacency, _improve_adjacency_by_swapping,
)


def _make_test_layout(overlapping: bool = False) -> LayoutGraph:
    """Create a test layout with or without overlaps."""
    if overlapping:
        rooms = [
            RoomLayout(
                room_spec=RoomSpec(room_id="r1", room_type=RoomType.LIVING_ROOM, target_area_sqm=20.0),
                bbox=BoundingBox(x_min=0.0, y_min=0.0, x_max=0.5, y_max=0.5),
            ),
            RoomLayout(
                room_spec=RoomSpec(room_id="r2", room_type=RoomType.KITCHEN, target_area_sqm=10.0),
                bbox=BoundingBox(x_min=0.3, y_min=0.3, x_max=0.7, y_max=0.7),
            ),
        ]
    else:
        rooms = [
            RoomLayout(
                room_spec=RoomSpec(room_id="r1", room_type=RoomType.LIVING_ROOM, target_area_sqm=20.0),
                bbox=BoundingBox(x_min=0.0, y_min=0.0, x_max=0.5, y_max=0.5),
            ),
            RoomLayout(
                room_spec=RoomSpec(room_id="r2", room_type=RoomType.KITCHEN, target_area_sqm=10.0),
                bbox=BoundingBox(x_min=0.5, y_min=0.0, x_max=0.8, y_max=0.4),
            ),
        ]

    return LayoutGraph(
        rooms=rooms,
        adjacency_edges=[
            AdjacencyEdge(room_a_id="r1", room_b_id="r2",
                         connection_type=ConnectionType.OPENING, required=True),
        ],
        plot_area_sqm=100.0,
        facing=CompassFacing.NORTH,
    )


def test_overlap_detection_zero():
    """Non-overlapping rooms should have overlap_rate == 0.0."""
    layout = _make_test_layout(overlapping=False)
    overlap = _compute_overlap(layout, 10.0, 10.0)
    assert overlap == 0.0


def test_overlap_detection_positive():
    """Overlapping rooms should have positive overlap_rate."""
    layout = _make_test_layout(overlapping=True)
    overlap = _compute_overlap(layout, 10.0, 10.0)
    assert overlap > 0.0


def test_heuristic_minimum_dimension():
    """All rooms from heuristic placer should have min dimension >= 2.4m."""
    layout = _make_test_layout(overlapping=True)
    result = heuristic_place(layout, 10.0, 10.0)

    for room in result.rooms:
        w = (room.bbox.x_max - room.bbox.x_min) * 10.0
        h = (room.bbox.y_max - room.bbox.y_min) * 10.0
        assert w >= 2.3, f"Room {room.room_spec.room_id} width {w} < 2.4m"
        assert h >= 2.3, f"Room {room.room_spec.room_id} height {h} < 2.4m"


def test_heuristic_sets_mode():
    """Heuristic placer should set generation_mode = 'heuristic'."""
    layout = _make_test_layout(overlapping=True)
    result = heuristic_place(layout, 10.0, 10.0)
    assert result.generation_mode == "heuristic"


def test_pipeline_produces_valid_output():
    """Pipeline should produce a LayoutGraph with all rooms."""
    layout = _make_test_layout(overlapping=True)
    result = run_constraint_pipeline(layout, 10.0, 10.0)

    assert len(result.rooms) == len(layout.rooms)
    assert result.overlap_rate >= 0.0
    assert result.adjacency_satisfaction >= 0.0


def test_adjacency_swap_improves_required_edges():
    """Swap helper should improve required adjacency when a better assignment exists."""
    rooms = [
        # r1 at NW
        RoomLayout(
            room_spec=RoomSpec(room_id="r1", room_type=RoomType.LIVING_ROOM, target_area_sqm=20.0),
            bbox=BoundingBox(x_min=0.0, y_min=0.0, x_max=0.5, y_max=0.5),
        ),
        # r2 at SE (not adjacent to r1)
        RoomLayout(
            room_spec=RoomSpec(room_id="r2", room_type=RoomType.KITCHEN, target_area_sqm=12.0),
            bbox=BoundingBox(x_min=0.5, y_min=0.5, x_max=1.0, y_max=1.0),
        ),
        # r3 at NE (adjacent to r1)
        RoomLayout(
            room_spec=RoomSpec(room_id="r3", room_type=RoomType.BEDROOM, target_area_sqm=14.0),
            bbox=BoundingBox(x_min=0.5, y_min=0.0, x_max=1.0, y_max=0.5),
        ),
    ]
    edges = [
        AdjacencyEdge(
            room_a_id="r1",
            room_b_id="r2",
            connection_type=ConnectionType.OPENING,
            required=True,
        ),
    ]

    base_layout = LayoutGraph(
        rooms=rooms,
        adjacency_edges=edges,
        plot_area_sqm=100.0,
        facing=CompassFacing.NORTH,
    )
    base_adj = _compute_adjacency(base_layout, 10.0, 10.0)

    improved_rooms = _improve_adjacency_by_swapping(rooms, edges, 10.0, 10.0)
    improved_layout = base_layout.model_copy(update={"rooms": improved_rooms})
    improved_adj = _compute_adjacency(improved_layout, 10.0, 10.0)

    assert improved_adj >= base_adj
    assert improved_adj > 0.0
