"""
Renderer — SVG and DXF export service for floor plan layouts.

Converts a LayoutGraph into visual outputs:
- SVG: in-browser rendering with room colours, labels, dimensions, doors
- DXF: CAD-compatible format using ezdxf
"""

from __future__ import annotations

import io
import logging
import math
import xml.etree.ElementTree as ET
from typing import Optional

from app.models.schemas import (
    BoundingBox,
    LayoutGraph,
    RoomLayout,
    RoomType,
)

logger = logging.getLogger(__name__)

# Try to import ezdxf
try:
    import ezdxf
    from ezdxf.enums import TextEntityAlignment
    HAS_EZDXF = True
except ImportError:
    HAS_EZDXF = False
    logger.warning("ezdxf not installed — DXF export disabled")

# Try to import Shapely
try:
    from shapely.geometry import box as shapely_box
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False


# ─── Room Colour Palette ────────────────────────────────────────────────────

ROOM_COLOURS: dict[RoomType, str] = {
    RoomType.LIVING_ROOM: "#E8F4FD",
    RoomType.KITCHEN: "#FFF9E6",
    RoomType.MASTER_BEDROOM: "#F0F4FF",
    RoomType.BEDROOM: "#F5F0FF",
    RoomType.BATHROOM: "#E6F9F5",
    RoomType.TOILET: "#E6F9F5",
    RoomType.CORRIDOR: "#F5F5F5",
    RoomType.BALCONY: "#E8F8E8",
    RoomType.STUDY: "#FFF0E8",
    RoomType.DINING: "#FFF5F0",
    RoomType.UTILITY: "#F0F0F0",
    RoomType.GARAGE: "#EDEDED",
}

DEFAULT_COLOUR = "#FAFAFA"


# ─── FUNCTION 1: SVG Rendering ──────────────────────────────────────────────


def layout_to_svg(
    layout: LayoutGraph,
    canvas_width_px: int = 800,
    canvas_height_px: int = 800,
    show_dimensions: bool = True,
    show_room_labels: bool = True,
) -> str:
    """Convert a LayoutGraph into an SVG string.

    Args:
        layout: The layout graph to render
        canvas_width_px: Canvas width in pixels
        canvas_height_px: Canvas height in pixels
        show_dimensions: Whether to show dimension lines
        show_room_labels: Whether to show room labels

    Returns:
        Complete SVG as a UTF-8 string
    """
    # Calculate plot dimensions
    plot_area = layout.plot_area_sqm
    aspect = 1.0  # Assume square plot
    plot_w = math.sqrt(plot_area * aspect)
    plot_h = plot_area / plot_w

    # Padding
    pad = 30
    draw_w = canvas_width_px - 2 * pad
    draw_h = canvas_height_px - 2 * pad

    # SVG root
    svg = ET.Element("svg", {
        "xmlns": "http://www.w3.org/2000/svg",
        "viewBox": f"0 0 {canvas_width_px} {canvas_height_px}",
        "width": str(canvas_width_px),
        "height": str(canvas_height_px),
        "style": "background: #FFFFFF; font-family: Arial, sans-serif;",
    })

    # Defs (could add patterns/gradients here)
    defs = ET.SubElement(svg, "defs")

    # Shadow filter
    shadow_filter = ET.SubElement(defs, "filter", {"id": "shadow", "x": "-2%", "y": "-2%", "width": "104%", "height": "104%"})
    ET.SubElement(shadow_filter, "feDropShadow", {
        "dx": "1", "dy": "1", "stdDeviation": "2", "flood-opacity": "0.1",
    })

    # Background
    ET.SubElement(svg, "rect", {
        "x": "0", "y": "0",
        "width": str(canvas_width_px), "height": str(canvas_height_px),
        "fill": "#FAFBFC",
    })

    # Plot border
    ET.SubElement(svg, "rect", {
        "x": str(pad), "y": str(pad),
        "width": str(draw_w), "height": str(draw_h),
        "fill": "none",
        "stroke": "#0D1F3C",
        "stroke-width": "3",
        "rx": "2",
    })

    # Compass indicator
    _add_compass(svg, canvas_width_px - 50, 50, layout.facing.value)

    # Room rectangles
    for room in layout.rooms:
        bbox = room.bbox
        x = pad + bbox.x_min * draw_w
        y = pad + bbox.y_min * draw_h
        w = (bbox.x_max - bbox.x_min) * draw_w
        h = (bbox.y_max - bbox.y_min) * draw_h

        fill = ROOM_COLOURS.get(room.room_spec.room_type, DEFAULT_COLOUR)

        # Room rectangle
        ET.SubElement(svg, "rect", {
            "x": f"{x:.1f}", "y": f"{y:.1f}",
            "width": f"{w:.1f}", "height": f"{h:.1f}",
            "fill": fill,
            "stroke": "#334155",
            "stroke-width": "2",
            "rx": "2",
            "data-room-type": room.room_spec.room_type.value,
            "data-room-id": room.room_spec.room_id,
            "filter": "url(#shadow)",
        })

        # Room label
        if show_room_labels:
            display_name = room.room_spec.room_type.value.replace("_", " ").title()
            room_area_m2 = bbox.area * plot_area
            label_text = f"{display_name}"
            area_text = f"{room_area_m2:.1f} m²"

            cx = x + w / 2
            cy = y + h / 2

            # Truncate label if room is small
            font_size = min(11, max(8, w / 8))

            name_elem = ET.SubElement(svg, "text", {
                "x": f"{cx:.1f}", "y": f"{cy - 6:.1f}",
                "text-anchor": "middle",
                "dominant-baseline": "middle",
                "font-family": "Arial, sans-serif",
                "font-size": f"{font_size}",
                "font-weight": "600",
                "fill": "#1E293B",
            })
            name_elem.text = label_text

            area_elem = ET.SubElement(svg, "text", {
                "x": f"{cx:.1f}", "y": f"{cy + 8:.1f}",
                "text-anchor": "middle",
                "dominant-baseline": "middle",
                "font-family": "Arial, sans-serif",
                "font-size": f"{max(font_size - 2, 7)}",
                "fill": "#64748B",
            })
            area_elem.text = area_text

        # Dimension lines
        if show_dimensions:
            room_w_m = (bbox.x_max - bbox.x_min) * plot_w
            room_h_m = (bbox.y_max - bbox.y_min) * plot_h

            # Bottom dimension
            dim_y = y + h + 12
            ET.SubElement(svg, "line", {
                "x1": f"{x:.1f}", "y1": f"{dim_y:.1f}",
                "x2": f"{x + w:.1f}", "y2": f"{dim_y:.1f}",
                "stroke": "#94A3B8", "stroke-width": "1",
            })
            dim_text = ET.SubElement(svg, "text", {
                "x": f"{x + w/2:.1f}", "y": f"{dim_y + 10:.1f}",
                "text-anchor": "middle",
                "font-size": "8", "fill": "#94A3B8",
                "font-family": "Arial, sans-serif",
            })
            dim_text.text = f"{room_w_m:.1f}m"

            # Right dimension
            dim_x = x + w + 8
            dim_text_r = ET.SubElement(svg, "text", {
                "x": f"{dim_x:.1f}", "y": f"{y + h/2:.1f}",
                "text-anchor": "start",
                "dominant-baseline": "middle",
                "font-size": "8", "fill": "#94A3B8",
                "font-family": "Arial, sans-serif",
                "transform": f"rotate(90 {dim_x:.1f} {y + h/2:.1f})",
            })
            dim_text_r.text = f"{room_h_m:.1f}m"

        # Door markers
        for dx, dy in room.door_midpoints:
            door_x = pad + dx * draw_w
            door_y = pad + dy * draw_h
            # Small arc for door
            ET.SubElement(svg, "circle", {
                "cx": f"{door_x:.1f}", "cy": f"{door_y:.1f}",
                "r": "4",
                "fill": "#2196F3",
                "stroke": "#1565C0",
                "stroke-width": "1.5",
            })

    # Title
    title = ET.SubElement(svg, "text", {
        "x": str(pad), "y": "18",
        "font-size": "12", "font-weight": "bold",
        "fill": "#1E293B", "font-family": "Arial, sans-serif",
    })
    title.text = f"Floor Plan — {plot_area:.0f} m² | {layout.facing.value}-facing"

    # Metrics
    metrics = ET.SubElement(svg, "text", {
        "x": str(pad), "y": str(canvas_height_px - 8),
        "font-size": "9", "fill": "#94A3B8",
        "font-family": "Arial, sans-serif",
    })
    metrics.text = (
        f"Overlap: {layout.overlap_rate:.1%} | "
        f"Adjacency: {layout.adjacency_satisfaction:.1%} | "
        f"Mode: {layout.generation_mode}"
    )

    return ET.tostring(svg, encoding="unicode", xml_declaration=False)


def _add_compass(
    parent: ET.Element, cx: float, cy: float, facing: str
) -> None:
    """Add a small compass indicator to the SVG."""
    g = ET.SubElement(parent, "g", {
        "transform": f"translate({cx},{cy})",
    })

    # Circle
    ET.SubElement(g, "circle", {
        "cx": "0", "cy": "0", "r": "18",
        "fill": "#F8FAFC", "stroke": "#CBD5E1", "stroke-width": "1",
    })

    # N marker
    n = ET.SubElement(g, "text", {
        "x": "0", "y": "-7",
        "text-anchor": "middle", "font-size": "10",
        "font-weight": "bold",
        "fill": "#EF4444" if facing == "NORTH" else "#94A3B8",
        "font-family": "Arial",
    })
    n.text = "N"

    # Arrow pointing up (North)
    ET.SubElement(g, "polygon", {
        "points": "0,-16 -3,-10 3,-10",
        "fill": "#EF4444",
    })


# ─── FUNCTION 2: DXF Export ─────────────────────────────────────────────────


def layout_to_dxf(
    layout: LayoutGraph,
    plot_width_m: float = 10.0,
    plot_height_m: float = 10.0,
    scale_str: str = "1:100",
) -> bytes:
    """Convert a LayoutGraph into a DXF R2018 document.

    Args:
        layout: The layout graph to render
        plot_width_m: Plot width in metres
        plot_height_m: Plot height in metres
        scale_str: Scale string (e.g. "1:100")

    Returns:
        DXF file as bytes
    """
    if not HAS_EZDXF:
        raise RuntimeError("ezdxf is not installed — cannot export DXF")

    # Parse scale
    scale_parts = scale_str.split(":")
    scale_factor = int(scale_parts[1]) if len(scale_parts) == 2 else 100

    doc = ezdxf.new("R2018")
    msp = doc.modelspace()

    # Create layers
    doc.layers.add("WALLS", color=7)
    doc.layers.add("DOORS", color=3)
    doc.layers.add("DIMENSIONS", color=2)
    doc.layers.add("ANNOTATIONS", color=1)
    doc.layers.add("PLOT_BOUNDARY", color=5)

    # Plot boundary
    bx = plot_width_m * scale_factor
    by = plot_height_m * scale_factor
    msp.add_lwpolyline(
        [(0, 0), (bx, 0), (bx, by), (0, by), (0, 0)],
        dxfattribs={"layer": "PLOT_BOUNDARY"},
    )

    # Rooms
    for room in layout.rooms:
        bbox = room.bbox
        x1 = bbox.x_min * plot_width_m * scale_factor
        y1 = bbox.y_min * plot_height_m * scale_factor
        x2 = bbox.x_max * plot_width_m * scale_factor
        y2 = bbox.y_max * plot_height_m * scale_factor

        # Wall lines
        walls = [(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)]
        msp.add_lwpolyline(walls, dxfattribs={"layer": "WALLS"})

        # Annotation at centroid
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        room_area = (bbox.x_max - bbox.x_min) * plot_width_m * \
                    (bbox.y_max - bbox.y_min) * plot_height_m
        label = f"{room.room_spec.room_type.value.replace('_', ' ').title()} — {room_area:.1f}m²"
        msp.add_text(
            label,
            height=0.3 * scale_factor,
            dxfattribs={"layer": "ANNOTATIONS", "insert": (cx, cy)},
        )

        # Door arcs
        for dx, dy in room.door_midpoints:
            door_x = dx * plot_width_m * scale_factor
            door_y = dy * plot_height_m * scale_factor
            msp.add_circle(
                center=(door_x, door_y),
                radius=0.3 * scale_factor,
                dxfattribs={"layer": "DOORS"},
            )

    # Write to bytes
    stream = io.BytesIO()
    doc.save(stream)
    return stream.getvalue()


# ─── FUNCTION 3: Overlap Rate ───────────────────────────────────────────────


def compute_overlap_rate(layout: LayoutGraph) -> float:
    """Compute total overlapping area / total floor area using Shapely.

    Args:
        layout: The layout graph

    Returns:
        Float between 0.0 and 1.0
    """
    if not layout.rooms:
        return 0.0

    if HAS_SHAPELY:
        polys = []
        total_area = 0.0

        for room in layout.rooms:
            b = room.bbox
            poly = shapely_box(b.x_min, b.y_min, b.x_max, b.y_max)
            polys.append(poly)
            total_area += poly.area

        if total_area == 0:
            return 0.0

        overlap_area = 0.0
        for i in range(len(polys)):
            for j in range(i + 1, len(polys)):
                intersection = polys[i].intersection(polys[j])
                overlap_area += intersection.area

        return min(overlap_area / total_area, 1.0)
    else:
        # Fallback: manual calculation
        total_area = 0.0
        total_overlap = 0.0
        rooms = layout.rooms

        for room in rooms:
            w = room.bbox.x_max - room.bbox.x_min
            h = room.bbox.y_max - room.bbox.y_min
            total_area += w * h

        if total_area == 0:
            return 0.0

        for i in range(len(rooms)):
            for j in range(i + 1, len(rooms)):
                ri, rj = rooms[i].bbox, rooms[j].bbox
                ox = max(0, min(ri.x_max, rj.x_max) - max(ri.x_min, rj.x_min))
                oy = max(0, min(ri.y_max, rj.y_max) - max(ri.y_min, rj.y_min))
                total_overlap += ox * oy

        return min(total_overlap / total_area, 1.0)


# ─── FUNCTION 4: Adjacency Satisfaction ─────────────────────────────────────


def compute_adjacency_satisfaction(layout: LayoutGraph) -> float:
    """Check fraction of required adjacency edges that are satisfied.

    Two rooms are "adjacent" if their polygons share a boundary
    of length >= 0.9m. In normalised units for a 10m plot, that's 0.09.

    Args:
        layout: The layout graph

    Returns:
        Fraction between 0.0 and 1.0
    """
    required_edges = [e for e in layout.adjacency_edges if e.required]
    if not required_edges:
        return 1.0

    threshold = 0.09  # 0.9m / 10m = 0.09 normalised

    room_map = {r.room_spec.room_id: r.bbox for r in layout.rooms}
    satisfied = 0

    for edge in required_edges:
        if edge.room_a_id not in room_map or edge.room_b_id not in room_map:
            continue

        ba = room_map[edge.room_a_id]
        bb = room_map[edge.room_b_id]

        if HAS_SHAPELY:
            pa = shapely_box(ba.x_min, ba.y_min, ba.x_max, ba.y_max)
            pb = shapely_box(bb.x_min, bb.y_min, bb.x_max, bb.y_max)
            shared = pa.boundary.intersection(pb.boundary)
            if shared.length >= threshold:
                satisfied += 1
        else:
            # Manual check
            shared_len = 0.0
            # X-axis adjacency
            if abs(ba.x_max - bb.x_min) < 0.01 or abs(bb.x_max - ba.x_min) < 0.01:
                y_overlap = max(0, min(ba.y_max, bb.y_max) - max(ba.y_min, bb.y_min))
                shared_len = max(shared_len, y_overlap)
            # Y-axis adjacency
            if abs(ba.y_max - bb.y_min) < 0.01 or abs(bb.y_max - ba.y_min) < 0.01:
                x_overlap = max(0, min(ba.x_max, bb.x_max) - max(ba.x_min, bb.x_min))
                shared_len = max(shared_len, x_overlap)

            if shared_len >= threshold:
                satisfied += 1

    return satisfied / len(required_edges)
