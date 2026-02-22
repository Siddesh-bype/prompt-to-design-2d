"""Tests for the renderer module."""

import pytest
import xml.etree.ElementTree as ET

from app.models.schemas import (
    BoundingBox, RoomSpec, RoomType, CompassFacing,
    ConnectionType, AdjacencyEdge, RoomLayout, LayoutGraph,
)
from app.services.renderer import (
    layout_to_svg,
    compute_overlap_rate,
    compute_adjacency_satisfaction,
    HAS_EZDXF,
)


def _make_non_overlapping_layout() -> LayoutGraph:
    """Two rooms side by side, no overlap."""
    return LayoutGraph(
        rooms=[
            RoomLayout(
                room_spec=RoomSpec(room_id="r1", room_type=RoomType.LIVING_ROOM, target_area_sqm=25.0),
                bbox=BoundingBox(x_min=0.0, y_min=0.0, x_max=0.5, y_max=0.5),
            ),
            RoomLayout(
                room_spec=RoomSpec(room_id="r2", room_type=RoomType.KITCHEN, target_area_sqm=10.0),
                bbox=BoundingBox(x_min=0.5, y_min=0.0, x_max=0.8, y_max=0.4),
            ),
        ],
        adjacency_edges=[
            AdjacencyEdge(room_a_id="r1", room_b_id="r2",
                         connection_type=ConnectionType.OPENING, required=True),
        ],
        plot_area_sqm=100.0,
        facing=CompassFacing.NORTH,
    )


def _make_overlapping_layout() -> LayoutGraph:
    """Two rooms overlapping."""
    return LayoutGraph(
        rooms=[
            RoomLayout(
                room_spec=RoomSpec(room_id="r1", room_type=RoomType.LIVING_ROOM, target_area_sqm=25.0),
                bbox=BoundingBox(x_min=0.0, y_min=0.0, x_max=0.6, y_max=0.6),
            ),
            RoomLayout(
                room_spec=RoomSpec(room_id="r2", room_type=RoomType.KITCHEN, target_area_sqm=10.0),
                bbox=BoundingBox(x_min=0.3, y_min=0.3, x_max=0.8, y_max=0.8),
            ),
        ],
        adjacency_edges=[],
        plot_area_sqm=100.0,
        facing=CompassFacing.NORTH,
    )


def test_svg_output_is_valid_xml():
    """SVG output should be parseable XML."""
    layout = _make_non_overlapping_layout()
    svg_str = layout_to_svg(layout)

    # Should not raise
    root = ET.fromstring(svg_str)
    assert root.tag == "svg" or root.tag.endswith("}svg")


def test_svg_contains_all_rooms():
    """SVG should contain a rect for each room."""
    layout = _make_non_overlapping_layout()
    svg_str = layout_to_svg(layout)
    root = ET.fromstring(svg_str)

    ns = {"svg": "http://www.w3.org/2000/svg"}
    # Find all rects with data-room-id
    rects = [
        el for el in root.iter()
        if el.tag.endswith("rect") or el.tag == "rect"
    ]
    room_rects = [r for r in rects if r.get("data-room-id")]
    assert len(room_rects) == 2


def test_dxf_layers_created():
    """DXF output should have expected layers."""
    if not HAS_EZDXF:
        pytest.skip("ezdxf not installed")

    import ezdxf
    from app.services.renderer import layout_to_dxf

    layout = _make_non_overlapping_layout()
    dxf_bytes = layout_to_dxf(layout)

    # Parse the DXF
    stream = __import__("io").BytesIO(dxf_bytes)
    doc = ezdxf.read(stream)

    layer_names = [layer.dxf.name for layer in doc.layers]
    assert "WALLS" in layer_names
    assert "DOORS" in layer_names
    assert "DIMENSIONS" in layer_names
    assert "ANNOTATIONS" in layer_names
    assert "PLOT_BOUNDARY" in layer_names


def test_overlap_rate_zero_for_non_overlapping():
    """Non-overlapping rooms should have overlap_rate == 0."""
    layout = _make_non_overlapping_layout()
    rate = compute_overlap_rate(layout)
    assert rate == 0.0


def test_overlap_rate_positive_for_overlapping():
    """Overlapping rooms should have positive overlap_rate."""
    layout = _make_overlapping_layout()
    rate = compute_overlap_rate(layout)
    assert rate > 0.0
