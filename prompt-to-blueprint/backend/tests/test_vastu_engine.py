"""Tests for the Vastu engine module."""

import pytest
from app.models.schemas import (
    BoundingBox, RoomSpec, RoomType, CompassFacing,
    RoomLayout, LayoutGraph, AdjacencyEdge,
)
from app.services.vastu_engine import (
    get_vastu_zone, check_vastu,
)


def test_zone_assignment_corners():
    """Test zone assignment at boundary positions."""
    assert get_vastu_zone(0.1, 0.1) == "NW"
    assert get_vastu_zone(0.5, 0.1) == "N"
    assert get_vastu_zone(0.9, 0.1) == "NE"
    assert get_vastu_zone(0.1, 0.5) == "W"
    assert get_vastu_zone(0.5, 0.5) == "CENTER"
    assert get_vastu_zone(0.9, 0.5) == "E"
    assert get_vastu_zone(0.1, 0.9) == "SW"
    assert get_vastu_zone(0.5, 0.9) == "S"
    assert get_vastu_zone(0.9, 0.9) == "SE"


def test_vastu_score_range():
    """Vastu score should be between 0 and 100."""
    # Kitchen in SE (ideal) = high score
    layout = LayoutGraph(
        rooms=[
            RoomLayout(
                room_spec=RoomSpec(room_id="r1", room_type=RoomType.KITCHEN, target_area_sqm=10.0),
                bbox=BoundingBox(x_min=0.7, y_min=0.7, x_max=0.95, y_max=0.95),
            ),
            RoomLayout(
                room_spec=RoomSpec(room_id="r2", room_type=RoomType.LIVING_ROOM, target_area_sqm=25.0),
                bbox=BoundingBox(x_min=0.6, y_min=0.0, x_max=0.95, y_max=0.3),
            ),
        ],
        adjacency_edges=[],
        plot_area_sqm=100.0,
        facing=CompassFacing.NORTH,
    )
    result = check_vastu(layout)
    assert 0 <= result.score <= 100
    assert isinstance(result.suggestions, list)
    assert isinstance(result.zone_map, dict)


def test_vastu_good_placement_high_score():
    """Kitchen in SE and Master Bedroom in SW should score well."""
    layout = LayoutGraph(
        rooms=[
            RoomLayout(
                room_spec=RoomSpec(room_id="r1", room_type=RoomType.KITCHEN, target_area_sqm=10.0),
                bbox=BoundingBox(x_min=0.7, y_min=0.7, x_max=0.95, y_max=0.95),  # SE
            ),
            RoomLayout(
                room_spec=RoomSpec(room_id="r2", room_type=RoomType.MASTER_BEDROOM, target_area_sqm=18.0),
                bbox=BoundingBox(x_min=0.05, y_min=0.7, x_max=0.3, y_max=0.95),  # SW
            ),
        ],
        adjacency_edges=[],
        plot_area_sqm=100.0,
        facing=CompassFacing.NORTH,
    )
    result = check_vastu(layout)
    assert result.score >= 80  # Both in ideal positions


def test_vastu_bad_placement_generates_suggestions():
    """Kitchen in NE (bad) should generate a suggestion."""
    layout = LayoutGraph(
        rooms=[
            RoomLayout(
                room_spec=RoomSpec(room_id="r1", room_type=RoomType.KITCHEN, target_area_sqm=10.0),
                bbox=BoundingBox(x_min=0.7, y_min=0.05, x_max=0.95, y_max=0.3),  # NE - bad for kitchen
            ),
            RoomLayout(
                room_spec=RoomSpec(room_id="r2", room_type=RoomType.LIVING_ROOM, target_area_sqm=25.0),
                bbox=BoundingBox(x_min=0.05, y_min=0.05, x_max=0.4, y_max=0.3),  # NW
            ),
        ],
        adjacency_edges=[],
        plot_area_sqm=100.0,
        facing=CompassFacing.NORTH,
    )
    result = check_vastu(layout)
    assert len(result.suggestions) > 0
    assert any("Kitchen" in s for s in result.suggestions)


def test_zone_map_keeps_multiple_rooms_in_same_zone():
    """Multiple rooms in one zone should not overwrite earlier zone_map entries."""
    layout = LayoutGraph(
        rooms=[
            RoomLayout(
                room_spec=RoomSpec(room_id="r1", room_type=RoomType.BEDROOM, target_area_sqm=12.0, label="Bedroom A"),
                bbox=BoundingBox(x_min=0.02, y_min=0.02, x_max=0.20, y_max=0.20),  # NW
            ),
            RoomLayout(
                room_spec=RoomSpec(room_id="r2", room_type=RoomType.STUDY, target_area_sqm=8.0, label="Study B"),
                bbox=BoundingBox(x_min=0.22, y_min=0.02, x_max=0.30, y_max=0.20),  # NW
            ),
        ],
        adjacency_edges=[],
        plot_area_sqm=100.0,
        facing=CompassFacing.NORTH,
    )

    result = check_vastu(layout)
    assert "NW" in result.zone_map
    assert "Bedroom A" in result.zone_map["NW"]
    assert "Study B" in result.zone_map["NW"]
