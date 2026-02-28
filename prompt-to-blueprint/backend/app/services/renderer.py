"""
Renderer — SVG and DXF export service for floor plan layouts.

Converts a LayoutGraph into visual outputs:
- SVG: in-browser rendering with professional blueprint aesthetics,
       furniture symbols, wall thickness, doors, windows, dimensions
- DXF: CAD-compatible format using ezdxf
"""

from __future__ import annotations

import io
import logging
import math
import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING, Optional

from app.models.schemas import (
    BoundingBox,
    LayoutGraph,
    RoomLayout,
    RoomType,
)

logger = logging.getLogger(__name__)

# Try to import ezdxf
try:
    import ezdxf  # type: ignore
    from ezdxf.enums import TextEntityAlignment  # type: ignore

    HAS_EZDXF = True
except ImportError:
    HAS_EZDXF = False
    ezdxf = None  # type: ignore
    TextEntityAlignment = None  # type: ignore
    logger.warning("ezdxf not installed — DXF export disabled")

# Try to import Shapely
try:
    from shapely.geometry import box as shapely_box  # type: ignore

    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False
    shapely_box = None  # type: ignore


# ─── Room Colour Palette (elegant, professional colours) ────────────────────

ROOM_COLOURS: dict[RoomType, str] = {
    RoomType.LIVING_ROOM: "#DCEEFB",
    RoomType.KITCHEN: "#FFF3D6",
    RoomType.MASTER_BEDROOM: "#E0E4F5",
    RoomType.BEDROOM: "#E8E0F5",
    RoomType.BATHROOM: "#D6F5EE",
    RoomType.TOILET: "#D6F5EE",
    RoomType.CORRIDOR: "#EDEDED",
    RoomType.BALCONY: "#D9F2D9",
    RoomType.STUDY: "#FFECD6",
    RoomType.DINING: "#FFE8E0",
    RoomType.UTILITY: "#E8E8E8",
    RoomType.GARAGE: "#E0E0E0",
}

DEFAULT_COLOUR = "#F5F5F5"

# Additional room patterns
ROOM_PATTERNS: dict[RoomType, str] = {}

# Wall rendering constants
WALL_THICKNESS_PX = 8  # main walls
INNER_WALL_PX = 4  # inner partition walls
DOOR_WIDTH_PX = 22  # door swing arc radius
WINDOW_WIDTH_PX = 20  # window marker width


# ─── Formatting Helpers ───


def format_ft_in(meters: float) -> str:
    """Convert meters to architectural feet and inches (e.g., 10' 6\")."""
    total_inches = meters * 39.3701
    feet = int(total_inches // 12)
    inches = int(round(total_inches % 12))
    if inches == 12:
        feet += 1
        inches = 0
    return f"{feet}' {inches}\""


# ─── FUNCTION 1: SVG Rendering ──────────────────────────────────────────────


def layout_to_svg(
    layout: LayoutGraph,
    canvas_width_px: int = 800,
    canvas_height_px: int = 800,
    show_dimensions: bool = True,
    show_room_labels: bool = True,
) -> str:
    """Convert a LayoutGraph into a professional architectural SVG string.

    Renders with:
    - Thick outer walls, thinner inner partition walls
    - Furniture symbols inside rooms
    - Door swing arcs at door positions
    - Window markers on exterior walls
    - Dimension annotations
    - Compass rose and title block
    """
    # Calculate plot dimensions
    plot_area = layout.plot_area_sqm
    aspect = 1.0
    plot_w = math.sqrt(plot_area * aspect)
    plot_h = plot_area / plot_w

    # Padding for dimension annotations
    pad = 50
    draw_w = canvas_width_px - 2 * pad
    draw_h = canvas_height_px - 2 * pad

    # SVG root
    svg = ET.Element(
        "svg",
        {
            "xmlns": "http://www.w3.org/2000/svg",
            "viewBox": f"0 0 {canvas_width_px} {canvas_height_px}",
            "width": str(canvas_width_px),
            "height": str(canvas_height_px),
            "style": "background: #FFFFFF; font-family: 'Segoe UI', Arial, sans-serif;",
        },
    )

    # ─── Defs ───
    defs = ET.SubElement(svg, "defs")

    # Shadow filter
    shadow_filter = ET.SubElement(
        defs,
        "filter",
        {
            "id": "room-shadow",
            "x": "-1%",
            "y": "-1%",
            "width": "102%",
            "height": "102%",
        },
    )
    ET.SubElement(
        shadow_filter,
        "feDropShadow",
        {
            "dx": "0.5",
            "dy": "0.5",
            "stdDeviation": "1.5",
            "flood-opacity": "0.08",
        },
    )

    # Corridor hatching
    hatch = ET.SubElement(
        defs,
        "pattern",
        {
            "id": "corridor-hatch",
            "patternUnits": "userSpaceOnUse",
            "width": "6",
            "height": "6",
            "patternTransform": "rotate(45)",
        },
    )
    ET.SubElement(
        hatch,
        "line",
        {
            "x1": "0",
            "y1": "0",
            "x2": "0",
            "y2": "6",
            "stroke": "#C0C8D4",
            "stroke-width": "1",
        },
    )

    # Bathroom tile pattern
    tile = ET.SubElement(
        defs,
        "pattern",
        {
            "id": "bath-tile",
            "patternUnits": "userSpaceOnUse",
            "width": "10",
            "height": "10",
        },
    )
    ET.SubElement(
        tile,
        "rect",
        {
            "width": "10",
            "height": "10",
            "fill": "#D6F5EE",
        },
    )
    ET.SubElement(
        tile,
        "path",
        {
            "d": "M 0 10 L 10 0 M -1 1 L 1 -1 M 9 11 L 11 9",
            "stroke": "#B0DDD0",
            "stroke-width": "0.5",
        },
    )

    # Kitchen floor tiles pattern
    kitchen_tile = ET.SubElement(
        defs,
        "pattern",
        {
            "id": "kitchen-tile",
            "patternUnits": "userSpaceOnUse",
            "width": "12",
            "height": "12",
        },
    )
    ET.SubElement(
        kitchen_tile,
        "rect",
        {"width": "12", "height": "12", "fill": "#FFF8E7"},
    )
    ET.SubElement(
        kitchen_tile,
        "rect",
        {
            "x": "0",
            "y": "0",
            "width": "12",
            "height": "12",
            "fill": "none",
            "stroke": "#E8D4B8",
            "stroke-width": "0.5",
        },
    )

    # Living room carpet pattern
    carpet = ET.SubElement(
        defs,
        "pattern",
        {
            "id": "living-carpet",
            "patternUnits": "userSpaceOnUse",
            "width": "8",
            "height": "8",
        },
    )
    ET.SubElement(carpet, "rect", {"width": "8", "height": "8", "fill": "#E8F4FC"})
    ET.SubElement(
        carpet,
        "rect",
        {
            "x": "1",
            "y": "1",
            "width": "6",
            "height": "6",
            "fill": "none",
            "stroke": "#C5DBE8",
            "stroke-width": "0.3",
        },
    )

    # Bedroom wood flooring pattern
    wood_floor = ET.SubElement(
        defs,
        "pattern",
        {
            "id": "wood-floor",
            "patternUnits": "userSpaceOnUse",
            "width": "20",
            "height": "6",
        },
    )
    ET.SubElement(wood_floor, "rect", {"width": "20", "height": "6", "fill": "#F5EBE0"})
    for i in range(4):
        ET.SubElement(
            wood_floor,
            "line",
            {
                "x1": f"{i * 5}",
                "y1": "0",
                "x2": f"{i * 5}",
                "y2": "6",
                "stroke": "#D4C4B0",
                "stroke-width": "0.5",
            },
        )

    # Balcony pattern
    balcony_pat = ET.SubElement(
        defs,
        "pattern",
        {
            "id": "balcony-pat",
            "patternUnits": "userSpaceOnUse",
            "width": "8",
            "height": "8",
        },
    )
    ET.SubElement(balcony_pat, "rect", {"width": "8", "height": "8", "fill": "#E8F5E8"})
    ET.SubElement(
        balcony_pat,
        "line",
        {
            "x1": "0",
            "y1": "4",
            "x2": "8",
            "y2": "4",
            "stroke": "#C8E0C8",
            "stroke-width": "0.5",
        },
    )

    # ─── Background ───
    ET.SubElement(
        svg,
        "rect",
        {
            "x": "0",
            "y": "0",
            "width": str(canvas_width_px),
            "height": str(canvas_height_px),
            "fill": "#FDFEFF",
        },
    )

    # ─── Title Block ───
    title = ET.SubElement(
        svg,
        "text",
        {
            "x": str(pad),
            "y": "22",
            "font-size": "13",
            "font-weight": "700",
            "fill": "#1A2740",
            "font-family": "'Segoe UI', Arial, sans-serif",
            "letter-spacing": "0.5",
        },
    )
    title.text = f"FLOOR PLAN — {plot_area:.0f} m² | {layout.facing.value}-FACING"

    # Subtitle with metrics
    subtitle = ET.SubElement(
        svg,
        "text",
        {
            "x": str(pad),
            "y": "38",
            "font-size": "9",
            "fill": "#7B8DA0",
            "font-family": "'Segoe UI', Arial, sans-serif",
        },
    )
    subtitle.text = (
        f"Plot: {plot_w:.1f}m × {plot_h:.1f}m | "
        f"Rooms: {len(layout.rooms)} | "
        f"Mode: {layout.generation_mode}"
    )

    # ─── Plot boundary (thick outer wall) ───
    # Outer boundary line
    ET.SubElement(
        svg,
        "rect",
        {
            "x": str(pad),
            "y": str(pad),
            "width": str(draw_w),
            "height": str(draw_h),
            "fill": "none",
            "stroke": "#1A2740",
            "stroke-width": str(WALL_THICKNESS_PX + 1),
            "rx": "1",
        },
    )
    # Inner boundary line for depth
    ET.SubElement(
        svg,
        "rect",
        {
            "x": str(pad + 3),
            "y": str(pad + 3),
            "width": str(draw_w - 6),
            "height": str(draw_h - 6),
            "fill": "none",
            "stroke": "#2A3750",
            "stroke-width": "1.5",
            "rx": "0.5",
        },
    )

    # Track door midpoints already drawn
    drawn_doors: set[tuple[int, int]] = set()

    # Collect room pixel data for window/furniture placement
    room_pixel_data: list[dict] = []

    # ─── Room Fills ───
    for room in layout.rooms:
        bbox = room.bbox
        x = pad + bbox.x_min * draw_w
        y = pad + bbox.y_min * draw_h
        w = (bbox.x_max - bbox.x_min) * draw_w
        h = (bbox.y_max - bbox.y_min) * draw_h

        fill = ROOM_COLOURS.get(room.room_spec.room_type, DEFAULT_COLOUR)
        is_corridor = room.room_spec.room_type == RoomType.CORRIDOR
        is_bathroom = room.room_spec.room_type in (RoomType.BATHROOM, RoomType.TOILET)
        is_kitchen = room.room_spec.room_type == RoomType.KITCHEN
        is_living = room.room_spec.room_type == RoomType.LIVING_ROOM
        is_bedroom = room.room_spec.room_type in (
            RoomType.BEDROOM,
            RoomType.MASTER_BEDROOM,
        )
        is_balcony = room.room_spec.room_type == RoomType.BALCONY

        rpd = {
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "room": room,
            "fill": fill,
            "room_w_m": (bbox.x_max - bbox.x_min) * plot_w,
            "room_h_m": (bbox.y_max - bbox.y_min) * plot_h,
        }
        room_pixel_data.append(rpd)

        # Room fill with patterns
        if is_bathroom:
            fill_attr = "url(#bath-tile)"
        elif is_kitchen:
            fill_attr = "url(#kitchen-tile)"
        elif is_living:
            fill_attr = "url(#living-carpet)"
        elif is_bedroom:
            fill_attr = "url(#wood-floor)"
        elif is_balcony:
            fill_attr = "url(#balcony-pat)"
        else:
            fill_attr = fill
        ET.SubElement(
            svg,
            "rect",
            {
                "x": f"{x:.1f}",
                "y": f"{y:.1f}",
                "width": f"{w:.1f}",
                "height": f"{h:.1f}",
                "fill": fill_attr,
                "stroke": "none",
                "data-room-type": room.room_spec.room_type.value,
                "data-room-id": room.room_spec.room_id,
                "filter": "url(#room-shadow)",
            },
        )

        # Corridor hatching overlay
        if is_corridor:
            ET.SubElement(
                svg,
                "rect",
                {
                    "x": f"{x:.1f}",
                    "y": f"{y:.1f}",
                    "width": f"{w:.1f}",
                    "height": f"{h:.1f}",
                    "fill": "url(#corridor-hatch)",
                    "stroke": "none",
                    "opacity": "0.3",
                },
            )

    # ─── Inner partition walls ───
    for rpd in room_pixel_data:
        x, y, w, h = rpd["x"], rpd["y"], rpd["w"], rpd["h"]
        # Outer wall line
        ET.SubElement(
            svg,
            "rect",
            {
                "x": f"{x:.1f}",
                "y": f"{y:.1f}",
                "width": f"{w:.1f}",
                "height": f"{h:.1f}",
                "fill": "none",
                "stroke": "#3A4A5C",
                "stroke-width": str(INNER_WALL_PX),
            },
        )
        # Inner wall line for depth effect
        ET.SubElement(
            svg,
            "rect",
            {
                "x": f"{x + 1.5:.1f}",
                "y": f"{y + 1.5:.1f}",
                "width": f"{w - 3:.1f}",
                "height": f"{h - 3:.1f}",
                "fill": "none",
                "stroke": "#5A6A7C",
                "stroke-width": "0.8",
            },
        )

    # ─── Furniture Symbols ───
    for rpd in room_pixel_data:
        _draw_furniture(svg, rpd)

    # ─── Room Labels ───
    if show_room_labels:
        for rpd in room_pixel_data:
            room = rpd["room"]
            x, y, w, h = rpd["x"], rpd["y"], rpd["w"], rpd["h"]
            display_name = room.room_spec.room_type.value.replace("_", " ").title()
            room_area_m2 = room.bbox.area * plot_area

            cx = x + w / 2
            cy = y + h / 2

            font_size = min(11, max(7, min(w, h) / 5.5))

            # Room name
            name_elem = ET.SubElement(
                svg,
                "text",
                {
                    "x": f"{cx:.1f}",
                    "y": f"{cy - 7:.1f}",
                    "text-anchor": "middle",
                    "dominant-baseline": "middle",
                    "font-size": f"{font_size:.1f}",
                    "font-weight": "600",
                    "fill": "#1E293B",
                    "font-family": "'Segoe UI', Arial, sans-serif",
                },
            )
            name_elem.text = display_name

            # Dimensions in Ft / In
            room_bbox = room.bbox
            dim_ft_w = format_ft_in((room_bbox.x_max - room_bbox.x_min) * plot_w)
            dim_ft_h = format_ft_in((room_bbox.y_max - room_bbox.y_min) * plot_h)

            dim_elem = ET.SubElement(
                svg,
                "text",
                {
                    "x": f"{cx:.1f}",
                    "y": f"{cy + 6:.1f}",
                    "text-anchor": "middle",
                    "dominant-baseline": "middle",
                    "font-size": f"{max(font_size - 1.5, 6.0):.1f}",
                    "fill": "#334155",
                    "font-weight": "500",
                    "font-family": "'Segoe UI', Arial, sans-serif",
                },
            )
            dim_elem.text = f"{dim_ft_w} x {dim_ft_h}"

            # Area in Sqm (smaller, lighter)
            area_elem = ET.SubElement(
                svg,
                "text",
                {
                    "x": f"{cx:.1f}",
                    "y": f"{cy + 16:.1f}",
                    "text-anchor": "middle",
                    "dominant-baseline": "middle",
                    "font-size": f"{max(font_size - 3, 5.0):.1f}",
                    "fill": "#64748B",
                    "font-family": "'Segoe UI', Arial, sans-serif",
                },
            )
            area_elem.text = f"{room_area_m2:.1f} m²"

    # ─── Dimension Lines ───
    if show_dimensions:
        for rpd in room_pixel_data:
            _draw_dimensions(svg, rpd)

    # ─── Door Arcs ───
    for rpd in room_pixel_data:
        room = rpd["room"]
        bbox = room.bbox
        for dx, dy in room.door_midpoints:
            door_x = pad + dx * draw_w
            door_y = pad + dy * draw_h

            key = (round(door_x * 10), round(door_y * 10))
            if key in drawn_doors:
                continue
            drawn_doors.add(key)

            _draw_door_arc(svg, door_x, door_y, dx, dy, bbox)

    # ─── Window Markers on Exterior Walls ───
    for rpd in room_pixel_data:
        _draw_exterior_windows(svg, rpd, pad, draw_w, draw_h)

    # ─── Compass Rose ───
    _add_compass(svg, canvas_width_px - 50, 50, layout.facing.value)

    # ─── Footer Metrics ───
    metrics = ET.SubElement(
        svg,
        "text",
        {
            "x": str(pad),
            "y": str(canvas_height_px - 8),
            "font-size": "8",
            "fill": "#94A3B8",
            "font-family": "'Segoe UI', Arial, sans-serif",
        },
    )
    metrics.text = (
        f"Overlap: {layout.overlap_rate:.1%} | "
        f"Adjacency: {layout.adjacency_satisfaction:.1%} | "
        f"Mode: {layout.generation_mode}"
    )

    return ET.tostring(svg, encoding="unicode", xml_declaration=False)


# ─── Furniture Drawing Helpers ──────────────────────────────────────────────


def _draw_furniture(parent: ET.Element, rpd: dict) -> None:
    """Draw room-appropriate furniture symbols."""
    room = rpd["room"]
    x, y, w, h = rpd["x"], rpd["y"], rpd["w"], rpd["h"]
    rt = room.room_spec.room_type
    stroke = "#8B9DB5"
    stroke_w = "0.8"

    if w < 30 or h < 30:
        return  # too small for furniture

    if rt == RoomType.LIVING_ROOM:
        _draw_sofa(parent, x, y, w, h, stroke, stroke_w)
    elif rt in (RoomType.BEDROOM, RoomType.MASTER_BEDROOM):
        _draw_bed(
            parent,
            x,
            y,
            w,
            h,
            stroke,
            stroke_w,
            is_master=(rt == RoomType.MASTER_BEDROOM),
        )
    elif rt == RoomType.KITCHEN:
        _draw_kitchen(parent, x, y, w, h, stroke, stroke_w)
    elif rt == RoomType.BATHROOM:
        _draw_bathroom(parent, x, y, w, h, stroke, stroke_w)
    elif rt == RoomType.TOILET:
        _draw_toilet(parent, x, y, w, h, stroke, stroke_w)
    elif rt == RoomType.DINING:
        _draw_dining(parent, x, y, w, h, stroke, stroke_w)
    elif rt == RoomType.STUDY:
        _draw_study(parent, x, y, w, h, stroke, stroke_w)
    elif rt == RoomType.BALCONY:
        _draw_balcony(parent, x, y, w, h, stroke, stroke_w)

    _draw_electrical(parent, x, y, w, h, rt)


def _draw_electrical(parent, x, y, w, h, rt):
    """Draw electrical symbols (outlets, switches, lights)."""
    elec_color = "#64748B"

    # Electrical outlet (small square with lines)
    if w > 40 and h > 40:
        out_x = x + w * 0.15
        out_y = y + h * 0.15
        ET.SubElement(
            parent,
            "rect",
            {
                "x": f"{out_x:.1f}",
                "y": f"{out_y:.1f}",
                "width": "3",
                "height": "3",
                "fill": "none",
                "stroke": elec_color,
                "stroke-width": "0.4",
            },
        )

    # Light switch (small rectangle)
    sw_x = x + 3
    sw_y = y + h * 0.5
    ET.SubElement(
        parent,
        "rect",
        {
            "x": f"{sw_x:.1f}",
            "y": f"{sw_y:.1f}",
            "width": "2.5",
            "height": "2",
            "fill": "none",
            "stroke": elec_color,
            "stroke-width": "0.3",
        },
    )

    # Ceiling light (circle in center)
    if w > 35 and h > 35:
        ET.SubElement(
            parent,
            "circle",
            {
                "cx": f"{x + w * 0.5:.1f}",
                "cy": f"{y + h * 0.3:.1f}",
                "r": "2.5",
                "fill": "none",
                "stroke": elec_color,
                "stroke-width": "0.4",
            },
        )
        ET.SubElement(
            parent,
            "line",
            {
                "x1": f"{x + w * 0.5:.1f}",
                "y1": f"{y + h * 0.3 - 2.5:.1f}",
                "x2": f"{x + w * 0.5:.1f}",
                "y2": f"{y + h * 0.3 - 5:.1f}",
                "stroke": elec_color,
                "stroke-width": "0.3",
            },
        )


def _draw_sofa(parent, x, y, w, h, stroke, stroke_w):
    """Sofa + coffee table in living room."""
    # Sofa (bottom area)
    sw = min(w * 0.55, 60)
    sh = min(h * 0.18, 14)
    sx = x + w * 0.5 - sw / 2
    sy = y + h * 0.72

    ET.SubElement(
        parent,
        "rect",
        {
            "x": f"{sx:.1f}",
            "y": f"{sy:.1f}",
            "width": f"{sw:.1f}",
            "height": f"{sh:.1f}",
            "fill": "none",
            "stroke": stroke,
            "stroke-width": stroke_w,
            "rx": "2",
        },
    )
    # Backrest
    ET.SubElement(
        parent,
        "rect",
        {
            "x": f"{sx:.1f}",
            "y": f"{sy + sh:.1f}",
            "width": f"{sw:.1f}",
            "height": f"{sh * 0.4:.1f}",
            "fill": "none",
            "stroke": stroke,
            "stroke-width": stroke_w,
            "rx": "1",
        },
    )

    # Coffee table
    tw = sw * 0.45
    th = sh * 0.6
    tx = x + w * 0.5 - tw / 2
    ty = sy - th - 8
    ET.SubElement(
        parent,
        "rect",
        {
            "x": f"{tx:.1f}",
            "y": f"{ty:.1f}",
            "width": f"{tw:.1f}",
            "height": f"{th:.1f}",
            "fill": "none",
            "stroke": stroke,
            "stroke-width": stroke_w,
            "rx": "1",
        },
    )

    # TV unit
    if w > 70:
        tv_w = min(w * 0.25, 30)
        tv_h = min(h * 0.06, 6)
        tv_x = x + w - tv_w - 4
        tv_y = y + h * 0.2
        ET.SubElement(
            parent,
            "rect",
            {
                "x": f"{tv_x:.1f}",
                "y": f"{tv_y:.1f}",
                "width": f"{tv_w:.1f}",
                "height": f"{tv_h:.1f}",
                "fill": "none",
                "stroke": stroke,
                "stroke-width": "0.5",
                "rx": "1",
            },
        )
        ET.SubElement(
            parent,
            "line",
            {
                "x1": f"{tv_x + 2:.1f}",
                "y1": f"{tv_y + tv_h / 2:.1f}",
                "x2": f"{tv_x + tv_w - 2:.1f}",
                "y2": f"{tv_y + tv_h / 2:.1f}",
                "stroke": stroke,
                "stroke-width": "0.3",
            },
        )

    # Side chairs
    if w > 60 and h > 50:
        chair_w = 10
        chair_h = 10
        # Left chair
        ET.SubElement(
            parent,
            "rect",
            {
                "x": f"{x + 6:.1f}",
                "y": f"{y + h * 0.35:.1f}",
                "width": f"{chair_w:.1f}",
                "height": f"{chair_h:.1f}",
                "fill": "none",
                "stroke": stroke,
                "stroke-width": "0.5",
                "rx": "2",
            },
        )
        # Right chair
        ET.SubElement(
            parent,
            "rect",
            {
                "x": f"{x + w - chair_w - 6:.1f}",
                "y": f"{y + h * 0.35:.1f}",
                "width": f"{chair_w:.1f}",
                "height": f"{chair_h:.1f}",
                "fill": "none",
                "stroke": stroke,
                "stroke-width": "0.5",
                "rx": "2",
            },
        )

    # AC Unit symbol
    ac_w = min(w * 0.12, 14)
    ac_h = 4
    ac_x = x + w - ac_w - 2
    ac_y = y + 2
    ET.SubElement(
        parent,
        "rect",
        {
            "x": f"{ac_x:.1f}",
            "y": f"{ac_y:.1f}",
            "width": f"{ac_w:.1f}",
            "height": f"{ac_h:.1f}",
            "fill": "none",
            "stroke": stroke,
            "stroke-width": "0.5",
            "rx": "1",
        },
    )


def _draw_bed(parent, x, y, w, h, stroke, stroke_w, is_master=False):
    """Bed with headboard, pillows, and a wardrobe."""
    # Wardrobe (along the top or bottom wall depending on room shape)
    ww = min(w * 0.4, 40)
    wh = min(h * 0.15, 12)
    wx = x + w - ww - 2
    wy = y + 2

    # Wardrobe outline
    ET.SubElement(
        parent,
        "rect",
        {
            "x": f"{wx:.1f}",
            "y": f"{wy:.1f}",
            "width": f"{ww:.1f}",
            "height": f"{wh:.1f}",
            "fill": "none",
            "stroke": stroke,
            "stroke-width": stroke_w,
        },
    )
    # Wardrobe cross-hatch (standard architectural symbol)
    ET.SubElement(
        parent,
        "line",
        {
            "x1": f"{wx:.1f}",
            "y1": f"{wy:.1f}",
            "x2": f"{wx + ww:.1f}",
            "y2": f"{wy + wh:.1f}",
            "stroke": stroke,
            "stroke-width": "0.4",
        },
    )
    ET.SubElement(
        parent,
        "line",
        {
            "x1": f"{wx + ww:.1f}",
            "y1": f"{wy:.1f}",
            "x2": f"{wx:.1f}",
            "y2": f"{wy + wh:.1f}",
            "stroke": stroke,
            "stroke-width": "0.4",
        },
    )

    # Bed rectangle (centred, occupying ~50% of room)
    bw = min(w * 0.55, 50 if is_master else 40)
    bh = min(h * 0.50, 45 if is_master else 38)
    bx = x + w * 0.5 - bw / 2
    by = y + h * 0.55 - bh / 2  # shifted slightly down to avoid wardrobe

    # Bed frame
    ET.SubElement(
        parent,
        "rect",
        {
            "x": f"{bx:.1f}",
            "y": f"{by:.1f}",
            "width": f"{bw:.1f}",
            "height": f"{bh:.1f}",
            "fill": "none",
            "stroke": stroke,
            "stroke-width": stroke_w,
            "rx": "1",
        },
    )

    # Headboard (top bar)
    ET.SubElement(
        parent,
        "rect",
        {
            "x": f"{bx:.1f}",
            "y": f"{by:.1f}",
            "width": f"{bw:.1f}",
            "height": f"{3:.1f}",
            "fill": stroke,
            "stroke": "none",
            "opacity": "0.3",
        },
    )

    # Pillows
    pw = bw * 0.38
    ph = bh * 0.14
    py = by + 6
    # Left pillow
    ET.SubElement(
        parent,
        "rect",
        {
            "x": f"{bx + 3:.1f}",
            "y": f"{py:.1f}",
            "width": f"{pw:.1f}",
            "height": f"{ph:.1f}",
            "fill": "none",
            "stroke": stroke,
            "stroke-width": "0.6",
            "rx": "2",
        },
    )
    # Right pillow (for master or double)
    if is_master or bw > 35:
        ET.SubElement(
            parent,
            "rect",
            {
                "x": f"{bx + bw - pw - 3:.1f}",
                "y": f"{py:.1f}",
                "width": f"{pw:.1f}",
                "height": f"{ph:.1f}",
                "fill": "none",
                "stroke": stroke,
                "stroke-width": "0.6",
                "rx": "2",
            },
        )

    # Side table
    st_w = bw * 0.18
    st_h = st_w
    st_x = bx - st_w - 4
    st_y = by + bh * 0.3
    if st_x > x + 4:
        ET.SubElement(
            parent,
            "rect",
            {
                "x": f"{st_x:.1f}",
                "y": f"{st_y:.1f}",
                "width": f"{st_w:.1f}",
                "height": f"{st_h:.1f}",
                "fill": "none",
                "stroke": stroke,
                "stroke-width": "0.5",
                "rx": "1",
            },
        )
        # Lamp on nightstand
        ET.SubElement(
            parent,
            "circle",
            {
                "cx": f"{st_x + st_w / 2:.1f}",
                "cy": f"{st_y - 2:.1f}",
                "r": "2.5",
                "fill": "none",
                "stroke": stroke,
                "stroke-width": "0.4",
            },
        )

    # Second nightstand on right side
    st_x2 = bx + bw + 4
    if st_x2 + st_w < x + w - 4:
        ET.SubElement(
            parent,
            "rect",
            {
                "x": f"{st_x2:.1f}",
                "y": f"{st_y:.1f}",
                "width": f"{st_w:.1f}",
                "height": f"{st_h:.1f}",
                "fill": "none",
                "stroke": stroke,
                "stroke-width": "0.5",
                "rx": "1",
            },
        )
        ET.SubElement(
            parent,
            "circle",
            {
                "cx": f"{st_x2 + st_w / 2:.1f}",
                "cy": f"{st_y - 2:.1f}",
                "r": "2.5",
                "fill": "none",
                "stroke": stroke,
                "stroke-width": "0.4",
            },
        )

    # TV (if room is wide enough)
    if w > 80:
        tv_w = min(w * 0.2, 25)
        tv_h = min(h * 0.08, 8)
        tv_x = x + w - tv_w - 4
        tv_y = y + h * 0.15
        ET.SubElement(
            parent,
            "rect",
            {
                "x": f"{tv_x:.1f}",
                "y": f"{tv_y:.1f}",
                "width": f"{tv_w:.1f}",
                "height": f"{tv_h:.1f}",
                "fill": "none",
                "stroke": stroke,
                "stroke-width": "0.6",
                "rx": "1",
            },
        )
        ET.SubElement(
            parent,
            "line",
            {
                "x1": f"{tv_x + 2:.1f}",
                "y1": f"{tv_y + tv_h / 2:.1f}",
                "x2": f"{tv_x + tv_w - 2:.1f}",
                "y2": f"{tv_y + tv_h / 2:.1f}",
                "stroke": stroke,
                "stroke-width": "0.3",
            },
        )

    # Ceiling fan symbol
    if h > 50:
        cf_x = x + w * 0.3
        cf_y = y + h * 0.25
        cf_r = 6
        ET.SubElement(
            parent,
            "circle",
            {
                "cx": f"{cf_x:.1f}",
                "cy": f"{cf_y:.1f}",
                "r": f"{cf_r:.1f}",
                "fill": "none",
                "stroke": stroke,
                "stroke-width": "0.4",
                "stroke-dasharray": "2,1",
            },
        )


def _draw_kitchen(parent, x, y, w, h, stroke, stroke_w):
    """Kitchen counter (L-shaped) + stove symbol."""
    # Counter along top wall
    cw = w * 0.85
    ch = min(h * 0.15, 12)
    cx = x + (w - cw) / 2
    cy = y + 4

    ET.SubElement(
        parent,
        "rect",
        {
            "x": f"{cx:.1f}",
            "y": f"{cy:.1f}",
            "width": f"{cw:.1f}",
            "height": f"{ch:.1f}",
            "fill": "none",
            "stroke": stroke,
            "stroke-width": stroke_w,
            "rx": "1",
        },
    )

    # L-extension (right side)
    lw = min(w * 0.12, 10)
    lh = min(h * 0.35, 30)
    lx = cx + cw - lw
    ly = cy + ch

    ET.SubElement(
        parent,
        "rect",
        {
            "x": f"{lx:.1f}",
            "y": f"{ly:.1f}",
            "width": f"{lw:.1f}",
            "height": f"{lh:.1f}",
            "fill": "none",
            "stroke": stroke,
            "stroke-width": stroke_w,
        },
    )

    # Stove burners (4 circles)
    burner_area_x = cx + cw * 0.6
    burner_area_y = cy + ch / 2
    br = min(ch * 0.2, 3)
    for dx_off, dy_off in [
        (-br * 1.5, -br * 0.6),
        (br * 1.5, -br * 0.6),
        (-br * 1.5, br * 0.6),
        (br * 1.5, br * 0.6),
    ]:
        ET.SubElement(
            parent,
            "circle",
            {
                "cx": f"{burner_area_x + dx_off:.1f}",
                "cy": f"{burner_area_y + dy_off:.1f}",
                "r": f"{br:.1f}",
                "fill": "none",
                "stroke": stroke,
                "stroke-width": "0.5",
            },
        )

    # Sink (double basin + drainboard)
    sk_w = min(cw * 0.2, 18)
    sk_h = ch * 0.7
    sk_x = cx + cw * 0.15
    sk_y = cy + (ch - sk_h) / 2

    # Sink outer edge
    ET.SubElement(
        parent,
        "rect",
        {
            "x": f"{sk_x:.1f}",
            "y": f"{sk_y:.1f}",
            "width": f"{sk_w:.1f}",
            "height": f"{sk_h:.1f}",
            "fill": "none",
            "stroke": stroke,
            "stroke-width": stroke_w,
            "rx": "1",
        },
    )
    # Basin
    ET.SubElement(
        parent,
        "rect",
        {
            "x": f"{sk_x + sk_w * 0.4:.1f}",
            "y": f"{sk_y + 2:.1f}",
            "width": f"{sk_w * 0.5:.1f}",
            "height": f"{sk_h - 4:.1f}",
            "fill": "none",
            "stroke": stroke,
            "stroke-width": "0.5",
            "rx": "1",
        },
    )
    # Drainboard lines
    for i in range(1, 4):
        lx = sk_x + i * (sk_w * 0.3) / 4
        ET.SubElement(
            parent,
            "line",
            {
                "x1": f"{lx:.1f}",
                "y1": f"{sk_y + 2:.1f}",
                "x2": f"{lx:.1f}",
                "y2": f"{sk_y + sk_h - 2:.1f}",
                "stroke": stroke,
                "stroke-width": "0.4",
            },
        )

    # Refrigerator (right side)
    if w > 80:
        ref_w = min(w * 0.12, 18)
        ref_h = min(h * 0.4, 35)
        ref_x = x + w - ref_w - 2
        ref_y = y + h - ref_h - 2
        ET.SubElement(
            parent,
            "rect",
            {
                "x": f"{ref_x:.1f}",
                "y": f"{ref_y:.1f}",
                "width": f"{ref_w:.1f}",
                "height": f"{ref_h:.1f}",
                "fill": "none",
                "stroke": stroke,
                "stroke-width": stroke_w,
            },
        )
        ET.SubElement(
            parent,
            "line",
            {
                "x1": f"{ref_x:.1f}",
                "y1": f"{ref_y + ref_h * 0.4:.1f}",
                "x2": f"{ref_x + ref_w:.1f}",
                "y2": f"{ref_y + ref_h * 0.4:.1f}",
                "stroke": stroke,
                "stroke-width": "0.6",
            },
        )

    # Microwave (above counter)
    if w > 60:
        mw_w = min(w * 0.1, 12)
        mw_h = min(ch * 0.8, 8)
        mw_x = cx + cw * 0.35
        mw_y = cy - mw_h - 1
        ET.SubElement(
            parent,
            "rect",
            {
                "x": f"{mw_x:.1f}",
                "y": f"{mw_y:.1f}",
                "width": f"{mw_w:.1f}",
                "height": f"{mw_h:.1f}",
                "fill": "none",
                "stroke": stroke,
                "stroke-width": "0.5",
                "rx": "1",
            },
        )

    # Dishwasher (below counter, if room allows)
    if h > 60 and w > 50:
        dw_w = min(w * 0.15, 16)
        dw_h = min(h * 0.12, 10)
        dw_x = x + w * 0.55
        dw_y = y + h - dw_h - 2
        ET.SubElement(
            parent,
            "rect",
            {
                "x": f"{dw_x:.1f}",
                "y": f"{dw_y:.1f}",
                "width": f"{dw_w:.1f}",
                "height": f"{dw_h:.1f}",
                "fill": "none",
                "stroke": stroke,
                "stroke-width": "0.5",
                "rx": "1",
            },
        )


def _draw_bathroom(parent, x, y, w, h, stroke, stroke_w):
    """Shower tray + toilet + sink."""
    # Shower tray (top-right corner)
    sw = min(w * 0.4, 25)
    sh = min(h * 0.35, 25)
    sx = x + w - sw - 2
    sy = y + 2
    ET.SubElement(
        parent,
        "rect",
        {
            "x": f"{sx:.1f}",
            "y": f"{sy:.1f}",
            "width": f"{sw:.1f}",
            "height": f"{sh:.1f}",
            "fill": "none",
            "stroke": stroke,
            "stroke-width": stroke_w,
        },
    )
    # Shower geometric X (standard architectural symbol)
    ET.SubElement(
        parent,
        "line",
        {
            "x1": f"{sx:.1f}",
            "y1": f"{sy:.1f}",
            "x2": f"{sx + sw:.1f}",
            "y2": f"{sy + sh:.1f}",
            "stroke": stroke,
            "stroke-width": "0.4",
        },
    )
    ET.SubElement(
        parent,
        "line",
        {
            "x1": f"{sx + sw:.1f}",
            "y1": f"{sy:.1f}",
            "x2": f"{sx:.1f}",
            "y2": f"{sy + sh:.1f}",
            "stroke": stroke,
            "stroke-width": "0.4",
        },
    )
    # Shower drain dot
    ET.SubElement(
        parent,
        "circle",
        {
            "cx": f"{sx + sw / 2:.1f}",
            "cy": f"{sy + sh / 2:.1f}",
            "r": "1.5",
            "fill": stroke,
            "stroke": "none",
        },
    )

    # Toilet (bottom-left)
    _draw_toilet_symbol(parent, x + 8, y + h - 22, min(w * 0.3, 16), 18, stroke)

    # Sink (top-left)
    sk_w = min(w * 0.22, 12)
    sk_h = min(h * 0.12, 8)
    ET.SubElement(
        parent,
        "ellipse",
        {
            "cx": f"{x + 10 + sk_w / 2:.1f}",
            "cy": f"{y + sh + 12:.1f}",
            "rx": f"{sk_w / 2:.1f}",
            "ry": f"{sk_h / 2:.1f}",
            "fill": "none",
            "stroke": stroke,
            "stroke-width": "0.6",
        },
    )

    # Bathtub (if bathroom is large enough)
    if w > 70 and h > 60:
        bt_w = min(w * 0.35, 40)
        bt_h = min(h * 0.15, 18)
        bt_x = x + 2
        bt_y = y + h - bt_h - 2
        ET.SubElement(
            parent,
            "rect",
            {
                "x": f"{bt_x:.1f}",
                "y": f"{bt_y:.1f}",
                "width": f"{bt_w:.1f}",
                "height": f"{bt_h:.1f}",
                "fill": "none",
                "stroke": stroke,
                "stroke-width": stroke_w,
                "rx": "3",
            },
        )
        ET.SubElement(
            parent,
            "line",
            {
                "x1": f"{bt_x + bt_w * 0.2:.1f}",
                "y1": f"{bt_y:.1f}",
                "x2": f"{bt_x + bt_w * 0.2:.1f}",
                "y2": f"{bt_y + bt_h:.1f}",
                "stroke": stroke,
                "stroke-width": "0.4",
            },
        )

    # Medicine cabinet (mirror)
    if h > 50:
        mc_w = min(w * 0.15, 10)
        mc_h = min(h * 0.15, 12)
        mc_x = x + w - mc_w - 2
        mc_y = y + 2
        ET.SubElement(
            parent,
            "rect",
            {
                "x": f"{mc_x:.1f}",
                "y": f"{mc_y:.1f}",
                "width": f"{mc_w:.1f}",
                "height": f"{mc_h:.1f}",
                "fill": "none",
                "stroke": stroke,
                "stroke-width": "0.5",
                "rx": "1",
            },
        )
        ET.SubElement(
            parent,
            "line",
            {
                "x1": f"{mc_x:.1f}",
                "y1": f"{mc_y + mc_h / 2:.1f}",
                "x2": f"{mc_x + mc_w:.1f}",
                "y2": f"{mc_y + mc_h / 2:.1f}",
                "stroke": stroke,
                "stroke-width": "0.3",
            },
        )


def _draw_toilet(parent, x, y, w, h, stroke, stroke_w):
    """Toilet + wash basin."""
    tw = min(w * 0.35, 16)
    th = min(h * 0.4, 20)
    tx = x + w / 2 - tw / 2
    ty = y + h * 0.55
    _draw_toilet_symbol(parent, tx, ty, tw, th, stroke)

    # Wash basin
    sk_w = min(w * 0.3, 14)
    sk_h = min(h * 0.12, 8)
    ET.SubElement(
        parent,
        "ellipse",
        {
            "cx": f"{x + w / 2:.1f}",
            "cy": f"{y + h * 0.25:.1f}",
            "rx": f"{sk_w / 2:.1f}",
            "ry": f"{sk_h / 2:.1f}",
            "fill": "none",
            "stroke": stroke,
            "stroke-width": "0.6",
        },
    )


def _draw_toilet_symbol(parent, x, y, w, h, stroke):
    """Draw a toilet bowl plan-view symbol."""
    # Tank
    ET.SubElement(
        parent,
        "rect",
        {
            "x": f"{x:.1f}",
            "y": f"{y + h * 0.7:.1f}",
            "width": f"{w:.1f}",
            "height": f"{h * 0.3:.1f}",
            "fill": "none",
            "stroke": stroke,
            "stroke-width": "0.6",
            "rx": "1",
        },
    )
    # Bowl
    ET.SubElement(
        parent,
        "ellipse",
        {
            "cx": f"{x + w / 2:.1f}",
            "cy": f"{y + h * 0.35:.1f}",
            "rx": f"{w / 2:.1f}",
            "ry": f"{h * 0.4:.1f}",
            "fill": "none",
            "stroke": stroke,
            "stroke-width": "0.6",
        },
    )


def _draw_dining(parent, x, y, w, h, stroke, stroke_w):
    """Dining table with chairs."""
    # Table
    tw = min(w * 0.45, 40)
    th = min(h * 0.35, 30)
    tx = x + w / 2 - tw / 2
    ty = y + h / 2 - th / 2
    ET.SubElement(
        parent,
        "rect",
        {
            "x": f"{tx:.1f}",
            "y": f"{ty:.1f}",
            "width": f"{tw:.1f}",
            "height": f"{th:.1f}",
            "fill": "none",
            "stroke": stroke,
            "stroke-width": stroke_w,
            "rx": "2",
        },
    )

    # Chairs (small semicircles on each side)
    chair_r = min(tw * 0.12, 4)
    positions = [
        (tx + tw * 0.3, ty - chair_r - 1),  # top
        (tx + tw * 0.7, ty - chair_r - 1),
        (tx + tw * 0.3, ty + th + 1),  # bottom
        (tx + tw * 0.7, ty + th + 1),
        (tx - chair_r - 1, ty + th * 0.5),  # left
        (tx + tw + 1, ty + th * 0.5),  # right
    ]
    for cx, cy in positions:
        if cx > x + 2 and cx < x + w - 2 and cy > y + 2 and cy < y + h - 2:
            ET.SubElement(
                parent,
                "circle",
                {
                    "cx": f"{cx:.1f}",
                    "cy": f"{cy:.1f}",
                    "r": f"{chair_r:.1f}",
                    "fill": "none",
                    "stroke": stroke,
                    "stroke-width": "0.5",
                },
            )

    # Sideboard
    if w > 60:
        sb_w = min(w * 0.25, 25)
        sb_h = min(h * 0.1, 8)
        sb_x = x + 2
        sb_y = y + h - sb_h - 2
        ET.SubElement(
            parent,
            "rect",
            {
                "x": f"{sb_x:.1f}",
                "y": f"{sb_y:.1f}",
                "width": f"{sb_w:.1f}",
                "height": f"{sb_h:.1f}",
                "fill": "none",
                "stroke": stroke,
                "stroke-width": "0.5",
                "rx": "1",
            },
        )
        ET.SubElement(
            parent,
            "line",
            {
                "x1": f"{sb_x + sb_w / 2:.1f}",
                "y1": f"{sb_y:.1f}",
                "x2": f"{sb_x + sb_w / 2:.1f}",
                "y2": f"{sb_y + sb_h:.1f}",
                "stroke": stroke,
                "stroke-width": "0.4",
            },
        )


def _draw_study(parent, x, y, w, h, stroke, stroke_w):
    """Desk with chair."""
    # Desk
    dw = min(w * 0.5, 35)
    dh = min(h * 0.18, 12)
    dx = x + w * 0.5 - dw / 2
    dy = y + h * 0.3
    ET.SubElement(
        parent,
        "rect",
        {
            "x": f"{dx:.1f}",
            "y": f"{dy:.1f}",
            "width": f"{dw:.1f}",
            "height": f"{dh:.1f}",
            "fill": "none",
            "stroke": stroke,
            "stroke-width": stroke_w,
            "rx": "1",
        },
    )

    # Chair (circle below desk)
    ET.SubElement(
        parent,
        "circle",
        {
            "cx": f"{dx + dw / 2:.1f}",
            "cy": f"{dy + dh + 8:.1f}",
            "r": "4",
            "fill": "none",
            "stroke": stroke,
            "stroke-width": "0.5",
        },
    )

    # Bookshelf
    if w > 50:
        bs_w = min(w * 0.15, 15)
        bs_h = min(h * 0.35, 30)
        bs_x = x + 2
        bs_y = y + h - bs_h - 2
        ET.SubElement(
            parent,
            "rect",
            {
                "x": f"{bs_x:.1f}",
                "y": f"{bs_y:.1f}",
                "width": f"{bs_w:.1f}",
                "height": f"{bs_h:.1f}",
                "fill": "none",
                "stroke": stroke,
                "stroke-width": "0.5",
            },
        )
        for i in range(3):
            ET.SubElement(
                parent,
                "line",
                {
                    "x1": f"{bs_x:.1f}",
                    "y1": f"{bs_y + (i + 1) * bs_h / 4:.1f}",
                    "x2": f"{bs_x + bs_w:.1f}",
                    "y2": f"{bs_y + (i + 1) * bs_h / 4:.1f}",
                    "stroke": stroke,
                    "stroke-width": "0.3",
                },
            )

    # Computer on desk
    if dw > 20:
        comp_w = min(dw * 0.25, 10)
        comp_h = min(dh * 0.5, 5)
        comp_x = dx + dw * 0.6
        comp_y = dy + 2
        ET.SubElement(
            parent,
            "rect",
            {
                "x": f"{comp_x:.1f}",
                "y": f"{comp_y:.1f}",
                "width": f"{comp_w:.1f}",
                "height": f"{comp_h:.1f}",
                "fill": "none",
                "stroke": stroke,
                "stroke-width": "0.4",
                "rx": "0.5",
            },
        )


def _draw_balcony(parent, x, y, w, h, stroke, stroke_w):
    """Railing lines on balcony."""
    # Railing (parallel lines near the outside edge)
    rail_y = y + h - 4
    for i in range(2):
        ET.SubElement(
            parent,
            "line",
            {
                "x1": f"{x + 4:.1f}",
                "y1": f"{rail_y - i * 3:.1f}",
                "x2": f"{x + w - 4:.1f}",
                "y2": f"{rail_y - i * 3:.1f}",
                "stroke": stroke,
                "stroke-width": "0.7",
            },
        )

    # Vertical railing supports
    n_supports = max(3, int(w / 15))
    spacing = (w - 8) / max(n_supports - 1, 1)
    for i in range(n_supports):
        sx = x + 4 + i * spacing
        ET.SubElement(
            parent,
            "line",
            {
                "x1": f"{sx:.1f}",
                "y1": f"{rail_y:.1f}",
                "x2": f"{sx:.1f}",
                "y2": f"{rail_y - 3:.1f}",
                "stroke": stroke,
                "stroke-width": "0.5",
            },
        )


# ─── Dimension Lines ────────────────────────────────────────────────────────


def _draw_dimensions(parent: ET.Element, rpd: dict) -> None:
    """Draw dimension annotations with tick marks."""
    x, y, w, h = rpd["x"], rpd["y"], rpd["w"], rpd["h"]
    room_w_m = rpd["room_w_m"]
    room_h_m = rpd["room_h_m"]
    dim_color = "#8B9DB5"

    if w < 25 or h < 25:
        return

    # Bottom dimension line
    dim_y = y + h + 10
    # Tick marks
    for tx in [x, x + w]:
        ET.SubElement(
            parent,
            "line",
            {
                "x1": f"{tx:.1f}",
                "y1": f"{dim_y - 3:.1f}",
                "x2": f"{tx:.1f}",
                "y2": f"{dim_y + 3:.1f}",
                "stroke": dim_color,
                "stroke-width": "0.8",
            },
        )
    # Line
    ET.SubElement(
        parent,
        "line",
        {
            "x1": f"{x:.1f}",
            "y1": f"{dim_y:.1f}",
            "x2": f"{x + w:.1f}",
            "y2": f"{dim_y:.1f}",
            "stroke": dim_color,
            "stroke-width": "0.6",
        },
    )
    # Label
    dim_text = ET.SubElement(
        parent,
        "text",
        {
            "x": f"{x + w / 2:.1f}",
            "y": f"{dim_y + 10:.1f}",
            "text-anchor": "middle",
            "font-size": "7",
            "fill": dim_color,
            "font-family": "'Segoe UI', Arial, sans-serif",
        },
    )
    dim_text.text = f"{room_w_m:.1f}m"

    # Right dimension (rotated)
    dim_x = x + w + 8
    for ty in [y, y + h]:
        ET.SubElement(
            parent,
            "line",
            {
                "x1": f"{dim_x - 3:.1f}",
                "y1": f"{ty:.1f}",
                "x2": f"{dim_x + 3:.1f}",
                "y2": f"{ty:.1f}",
                "stroke": dim_color,
                "stroke-width": "0.8",
            },
        )
    ET.SubElement(
        parent,
        "line",
        {
            "x1": f"{dim_x:.1f}",
            "y1": f"{y:.1f}",
            "x2": f"{dim_x:.1f}",
            "y2": f"{y + h:.1f}",
            "stroke": dim_color,
            "stroke-width": "0.6",
        },
    )
    dim_text_r = ET.SubElement(
        parent,
        "text",
        {
            "x": f"{dim_x:.1f}",
            "y": f"{y + h / 2:.1f}",
            "text-anchor": "start",
            "dominant-baseline": "middle",
            "font-size": "7",
            "fill": dim_color,
            "font-family": "'Segoe UI', Arial, sans-serif",
            "transform": f"rotate(90 {dim_x:.1f} {y + h / 2:.1f})",
        },
    )
    dim_text_r.text = f"{room_h_m:.1f}m"


# ─── Door Arc Drawing ───────────────────────────────────────────────────────


def _draw_door_arc(
    parent: ET.Element,
    door_x: float,
    door_y: float,
    dx_norm: float,
    dy_norm: float,
    bbox: BoundingBox,
) -> None:
    """Draw an architectural door swing arc."""
    arc_r = DOOR_WIDTH_PX / 2
    door_color = "#2563EB"

    on_vertical = abs(dx_norm - bbox.x_min) < 0.015 or abs(dx_norm - bbox.x_max) < 0.015

    if on_vertical:
        # Door on vertical wall (door leaf)
        ET.SubElement(
            parent,
            "line",
            {
                "x1": f"{door_x:.1f}",
                "y1": f"{door_y - arc_r:.1f}",
                "x2": f"{door_x:.1f}",
                "y2": f"{door_y + arc_r:.1f}",
                "stroke": "#1E293B",
                "stroke-width": "2.5",
            },
        )
        arc_path = (
            f"M {door_x:.1f} {door_y - arc_r:.1f} "
            f"A {arc_r} {arc_r} 0 0 1 {door_x + arc_r:.1f} {door_y:.1f}"
        )
    else:
        # Door on horizontal wall (door leaf)
        ET.SubElement(
            parent,
            "line",
            {
                "x1": f"{door_x - arc_r:.1f}",
                "y1": f"{door_y:.1f}",
                "x2": f"{door_x + arc_r:.1f}",
                "y2": f"{door_y:.1f}",
                "stroke": "#1E293B",
                "stroke-width": "2.5",
            },
        )
        arc_path = (
            f"M {door_x - arc_r:.1f} {door_y:.1f} "
            f"A {arc_r} {arc_r} 0 0 1 {door_x:.1f} {door_y - arc_r:.1f}"
        )

    ET.SubElement(
        parent,
        "path",
        {
            "d": arc_path,
            "fill": "none",
            "stroke": door_color,
            "stroke-width": "1.2",
            "stroke-dasharray": "3,3",
        },
    )


# ─── Window Markers on Exterior Walls ───────────────────────────────────────


def _draw_exterior_windows(
    parent: ET.Element,
    rpd: dict,
    pad: float,
    draw_w: float,
    draw_h: float,
) -> None:
    """Draw window markers on walls that touch the plot boundary."""
    room = rpd["room"]
    bbox = room.bbox
    x, y, w, h = rpd["x"], rpd["y"], rpd["w"], rpd["h"]
    win_color = "#3B82F6"

    # Skip corridors, utility, garage — they typically don't have windows
    if room.room_spec.room_type in (
        RoomType.CORRIDOR,
        RoomType.UTILITY,
        RoomType.GARAGE,
        RoomType.TOILET,
    ):
        return

    win_w = min(WINDOW_WIDTH_PX, w * 0.3, h * 0.3)

    # Top edge touches plot top (y_min ≈ 0)
    if bbox.y_min < 0.02 and w > 30:
        wx = x + w / 2 - win_w / 2
        wy = y
        _draw_window_mark(parent, wx, wy, win_w, True, win_color)

    # Bottom edge touches plot bottom (y_max ≈ 1)
    if bbox.y_max > 0.98 and w > 30:
        wx = x + w / 2 - win_w / 2
        wy = y + h
        _draw_window_mark(parent, wx, wy, win_w, True, win_color)

    # Left edge touches plot left (x_min ≈ 0)
    if bbox.x_min < 0.02 and h > 30:
        wx = x
        wy = y + h / 2 - win_w / 2
        _draw_window_mark(parent, wx, wy, win_w, False, win_color)

    # Right edge touches plot right (x_max ≈ 1)
    if bbox.x_max > 0.98 and h > 30:
        wx = x + w
        wy = y + h / 2 - win_w / 2
        _draw_window_mark(parent, wx, wy, win_w, False, win_color)


def _draw_window_mark(
    parent: ET.Element,
    x: float,
    y: float,
    size: float,
    horizontal: bool,
    color: str,
) -> None:
    """Draw a window marker (thick outer frame + blue glass line)."""
    frame_color = "#3A4A5C"

    if horizontal:
        # Outer frame bounds
        ET.SubElement(
            parent,
            "rect",
            {
                "x": f"{x:.1f}",
                "y": f"{y - 2:.1f}",
                "width": f"{size:.1f}",
                "height": "4",
                "fill": "#FFFFFF",
                "stroke": frame_color,
                "stroke-width": "1",
            },
        )
        # Glass line
        ET.SubElement(
            parent,
            "line",
            {
                "x1": f"{x:.1f}",
                "y1": f"{y:.1f}",
                "x2": f"{x + size:.1f}",
                "y2": f"{y:.1f}",
                "stroke": color,
                "stroke-width": "1.5",
            },
        )
    else:
        # Window on vertical wall
        ET.SubElement(
            parent,
            "rect",
            {
                "x": f"{x - 2:.1f}",
                "y": f"{y:.1f}",
                "width": "4",
                "height": f"{size:.1f}",
                "fill": "#FFFFFF",
                "stroke": frame_color,
                "stroke-width": "1",
            },
        )
        # Glass line
        ET.SubElement(
            parent,
            "line",
            {
                "x1": f"{x:.1f}",
                "y1": f"{y:.1f}",
                "x2": f"{x:.1f}",
                "y2": f"{y + size:.1f}",
                "stroke": color,
                "stroke-width": "1.5",
            },
        )


# ─── Compass Rose ───────────────────────────────────────────────────────────


def _add_compass(parent: ET.Element, cx: float, cy: float, facing: str) -> None:
    """Add a compass rose indicator."""
    g = ET.SubElement(
        parent,
        "g",
        {
            "transform": f"translate({cx},{cy})",
        },
    )

    # Outer circle
    ET.SubElement(
        g,
        "circle",
        {
            "cx": "0",
            "cy": "0",
            "r": "20",
            "fill": "#F8FAFC",
            "stroke": "#CBD5E1",
            "stroke-width": "1.5",
        },
    )

    # Inner circle
    ET.SubElement(
        g,
        "circle",
        {
            "cx": "0",
            "cy": "0",
            "r": "3",
            "fill": "#CBD5E1",
            "stroke": "none",
        },
    )

    # Direction markers
    directions = [
        ("N", 0, -9, "#DC2626" if facing == "NORTH" else "#94A3B8"),
        ("S", 0, 13, "#DC2626" if facing == "SOUTH" else "#94A3B8"),
        ("E", 9, 3, "#DC2626" if facing == "EAST" else "#94A3B8"),
        ("W", -9, 3, "#DC2626" if facing == "WEST" else "#94A3B8"),
    ]
    for label, dx, dy, color in directions:
        t = ET.SubElement(
            g,
            "text",
            {
                "x": str(dx),
                "y": str(dy),
                "text-anchor": "middle",
                "font-size": "9",
                "font-weight": "bold" if label[0] == facing[0] else "normal",
                "fill": color,
                "font-family": "'Segoe UI', Arial",
            },
        )
        t.text = label

    # North arrow
    ET.SubElement(
        g,
        "polygon",
        {
            "points": "0,-18 -3,-12 3,-12",
            "fill": "#DC2626",
        },
    )


# ─── FUNCTION 2: DXF Export ─────────────────────────────────────────────────


def layout_to_dxf(
    layout: LayoutGraph,
    plot_width_m: float = 10.0,
    plot_height_m: float = 10.0,
    scale_str: str = "1:100",
) -> bytes:
    """Convert a LayoutGraph into a DXF R2018 document."""
    if not HAS_EZDXF:
        raise RuntimeError("ezdxf is not installed — cannot export DXF")

    scale_parts = scale_str.split(":")
    try:
        scale_factor = int(scale_parts[1]) if len(scale_parts) == 2 else 100
    except (TypeError, ValueError):
        scale_factor = 100

    doc = ezdxf.new("R2018")  # type: ignore
    msp = doc.modelspace()  # type: ignore

    doc.layers.add("WALLS", color=7)
    doc.layers.add("DOORS", color=3)
    doc.layers.add("DIMENSIONS", color=2)
    doc.layers.add("ANNOTATIONS", color=1)
    doc.layers.add("PLOT_BOUNDARY", color=5)

    bx = plot_width_m * scale_factor
    by = plot_height_m * scale_factor
    msp.add_lwpolyline(
        [(0, 0), (bx, 0), (bx, by), (0, by), (0, 0)],
        dxfattribs={"layer": "PLOT_BOUNDARY"},
    )

    for room in layout.rooms:
        bbox = room.bbox
        x1 = bbox.x_min * plot_width_m * scale_factor
        y1 = bbox.y_min * plot_height_m * scale_factor
        x2 = bbox.x_max * plot_width_m * scale_factor
        y2 = bbox.y_max * plot_height_m * scale_factor

        walls = [(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)]
        msp.add_lwpolyline(walls, dxfattribs={"layer": "WALLS"})

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        room_area = (
            (bbox.x_max - bbox.x_min)
            * plot_width_m
            * (bbox.y_max - bbox.y_min)
            * plot_height_m
        )
        label = f"{room.room_spec.room_type.value.replace('_', ' ').title()} — {room_area:.1f}m²"
        msp.add_text(
            label,
            height=0.3 * scale_factor,
            dxfattribs={"layer": "ANNOTATIONS", "insert": (cx, cy)},
        )

        for dx, dy in room.door_midpoints:
            door_x = dx * plot_width_m * scale_factor
            door_y = dy * plot_height_m * scale_factor
            msp.add_circle(
                center=(door_x, door_y),
                radius=0.3 * scale_factor,
                dxfattribs={"layer": "DOORS"},
            )

    text_stream = io.StringIO()
    doc.write(text_stream)
    return text_stream.getvalue().encode("utf-8")


# ─── FUNCTION 3: Overlap Rate ───────────────────────────────────────────────


def compute_overlap_rate(layout: LayoutGraph) -> float:
    """Compute total overlapping area / total floor area using Shapely."""
    if not layout.rooms:
        return 0.0

    if HAS_SHAPELY:
        polys = []
        total_area = 0.0

        for room in layout.rooms:
            b = room.bbox
            poly = shapely_box(b.x_min, b.y_min, b.x_max, b.y_max)  # type: ignore
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
    """Check fraction of required adjacency edges that are satisfied."""
    required_edges = [e for e in layout.adjacency_edges if e.required]
    if not required_edges:
        return 1.0

    threshold = 0.09

    room_map = {r.room_spec.room_id: r.bbox for r in layout.rooms}
    satisfied = 0

    for edge in required_edges:
        if edge.room_a_id not in room_map or edge.room_b_id not in room_map:
            continue

        ba = room_map[edge.room_a_id]
        bb = room_map[edge.room_b_id]

        if HAS_SHAPELY:
            pa = shapely_box(ba.x_min, ba.y_min, ba.x_max, ba.y_max)  # type: ignore
            pb = shapely_box(bb.x_min, bb.y_min, bb.x_max, bb.y_max)  # type: ignore
            shared = pa.boundary.intersection(pb.boundary)
            if shared.length >= threshold:
                satisfied += 1
        else:
            shared_len = 0.0
            if abs(ba.x_max - bb.x_min) < 0.01 or abs(bb.x_max - ba.x_min) < 0.01:
                y_overlap = max(0, min(ba.y_max, bb.y_max) - max(ba.y_min, bb.y_min))
                shared_len = max(shared_len, y_overlap)
            if abs(ba.y_max - bb.y_min) < 0.01 or abs(bb.y_max - ba.y_min) < 0.01:
                x_overlap = max(0, min(ba.x_max, bb.x_max) - max(ba.x_min, bb.x_min))
                shared_len = max(shared_len, x_overlap)

            if shared_len >= threshold:
                satisfied += 1

    return satisfied / len(required_edges)
