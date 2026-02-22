"""Unit tests for Pydantic v2 schemas."""

import pytest
from app.models.schemas import (
    RoomType, CompassFacing, ConnectionType,
    RoomSpec, AdjacencyEdge, ParsedLayout,
    BoundingBox, RoomLayout, LayoutGraph,
    ErrorDetail, GenerateRequest,
)
from pydantic import ValidationError


def test_valid_room_spec_construction():
    """Test that a valid RoomSpec is constructed correctly."""
    room = RoomSpec(
        room_id="room_1",
        room_type=RoomType.LIVING_ROOM,
        target_area_sqm=25.0,
        compass_preference=CompassFacing.NORTH,
        label="Main Living Room",
    )
    assert room.room_id == "room_1"
    assert room.room_type == RoomType.LIVING_ROOM
    assert room.target_area_sqm == 25.0
    assert room.compass_preference == CompassFacing.NORTH
    assert room.label == "Main Living Room"


def test_invalid_area_range():
    """Test that area outside valid range raises ValidationError."""
    with pytest.raises(ValidationError):
        RoomSpec(
            room_id="room_bad",
            room_type=RoomType.BEDROOM,
            target_area_sqm=2.0,  # Below minimum of 4.0
        )
    with pytest.raises(ValidationError):
        RoomSpec(
            room_id="room_bad",
            room_type=RoomType.BEDROOM,
            target_area_sqm=100.0,  # Above maximum of 80.0
        )


def test_bbox_area_property():
    """Test that BoundingBox.area returns correct value."""
    bbox = BoundingBox(x_min=0.1, y_min=0.2, x_max=0.5, y_max=0.7)
    expected = (0.5 - 0.1) * (0.7 - 0.2)  # 0.4 * 0.5 = 0.2
    assert abs(bbox.area - expected) < 1e-10


def test_bbox_to_pixel_coords():
    """Test BoundingBox.to_pixel_coords conversion."""
    bbox = BoundingBox(x_min=0.0, y_min=0.0, x_max=0.5, y_max=0.5)
    x_min, y_min, x_max, y_max = bbox.to_pixel_coords(800, 800)
    assert x_min == 0
    assert y_min == 0
    assert x_max == 400
    assert y_max == 400

    bbox2 = BoundingBox(x_min=0.25, y_min=0.25, x_max=0.75, y_max=0.75)
    x_min, y_min, x_max, y_max = bbox2.to_pixel_coords(1000, 600)
    assert x_min == 250
    assert y_min == 150
    assert x_max == 750
    assert y_max == 450


def test_error_detail_serialisation():
    """Test ErrorDetail serialisation and deserialisation."""
    error = ErrorDetail(
        error_code="PARSE_FAILURE",
        message="Failed to parse NLP output",
        fallback_used=True,
        details={"attempts": 3, "last_error": "Invalid JSON"},
    )
    data = error.model_dump()
    assert data["error_code"] == "PARSE_FAILURE"
    assert data["message"] == "Failed to parse NLP output"
    assert data["fallback_used"] is True
    assert data["details"]["attempts"] == 3

    # Round-trip
    error2 = ErrorDetail.model_validate(data)
    assert error2 == error
