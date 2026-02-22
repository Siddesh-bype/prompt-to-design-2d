"""
CubiCasa5K Adapter — Converts CubiCasa5K SVG annotations to our LayoutGraph format.

CubiCasa5K structure (after extraction):
    data/
    ├── train.txt / val.txt / test.txt   (sample paths, one per line)
    └── cubicasa5k/
        └── <sample_id>/
            ├── F1_original.png          (floor plan image)
            └── model.svg                (polygon annotations)

The SVG contains <polygon> elements with class attributes indicating room type
(e.g., "Kitchen", "LivingRoom", "Bedroom", "Bath", "Hallway", etc.)
"""

from __future__ import annotations

import logging
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

from app.models.schemas import (
    AdjacencyEdge,
    BoundingBox,
    CompassFacing,
    ConnectionType,
    LayoutGraph,
    ParsedLayout,
    RoomLayout,
    RoomSpec,
    RoomType,
)

logger = logging.getLogger(__name__)

# ─── Category Mapping ────────────────────────────────────────────────────────

# Map CubiCasa5K SVG class names → our RoomType enum
CUBICASA_ROOM_MAP: dict[str, RoomType] = {
    # Exact class matches
    "Kitchen": RoomType.KITCHEN,
    "LivingRoom": RoomType.LIVING_ROOM,
    "Bedroom": RoomType.BEDROOM,
    "Bath": RoomType.BATHROOM,
    "Bathroom": RoomType.BATHROOM,
    "Toilet": RoomType.TOILET,
    "Hallway": RoomType.CORRIDOR,
    "Corridor": RoomType.CORRIDOR,
    "Storage": RoomType.UTILITY,
    "Closet": RoomType.UTILITY,
    "Garage": RoomType.GARAGE,
    "Balcony": RoomType.BALCONY,
    "Terrace": RoomType.BALCONY,
    "Dining": RoomType.DINING,
    "DiningRoom": RoomType.DINING,
    "Study": RoomType.STUDY,
    "Office": RoomType.STUDY,
    "Railing": RoomType.BALCONY,
    "Outdoor": RoomType.BALCONY,
    # Fallback for unrecognised categories
    "Room": RoomType.BEDROOM,
    "OtherRoom": RoomType.UTILITY,
    "Undefined": RoomType.UTILITY,
}


def _map_room_type(cubicasa_class: str) -> RoomType:
    """Map a CubiCasa class name to our RoomType enum."""
    # Try exact match first
    if cubicasa_class in CUBICASA_ROOM_MAP:
        return CUBICASA_ROOM_MAP[cubicasa_class]

    # Try case-insensitive partial match
    lower = cubicasa_class.lower()
    for key, room_type in CUBICASA_ROOM_MAP.items():
        if key.lower() in lower:
            return room_type

    # Default fallback
    return RoomType.UTILITY


# ─── SVG Polygon Extraction ─────────────────────────────────────────────────

# Namespace for SVG
SVG_NS = {"svg": "http://www.w3.org/2000/svg"}


def _parse_polygon_points(points_str: str) -> list[tuple[float, float]]:
    """Parse SVG polygon points attribute.

    Formats: "x1,y1 x2,y2 ..." or "x1 y1 x2 y2 ..."
    """
    points = []
    # Handle both comma-separated and space-separated
    tokens = re.split(r'[\s,]+', points_str.strip())

    for i in range(0, len(tokens) - 1, 2):
        try:
            x = float(tokens[i])
            y = float(tokens[i + 1])
            points.append((x, y))
        except (ValueError, IndexError):
            continue

    return points


def _polygon_to_bbox(points: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    """Convert polygon points to bounding box (x_min, y_min, x_max, y_max)."""
    if not points:
        return (0, 0, 0, 0)

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs), min(ys), max(xs), max(ys))


def _polygon_area(points: list[tuple[float, float]]) -> float:
    """Calculate polygon area using shoelace formula."""
    n = len(points)
    if n < 3:
        return 0.0

    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += points[i][0] * points[j][1]
        area -= points[j][0] * points[i][1]

    return abs(area) / 2.0


# ─── SVG Parsing ─────────────────────────────────────────────────────────────


def parse_cubicasa_svg(svg_path: str) -> Optional[dict]:
    """Parse a single CubiCasa5K model.svg file.

    Args:
        svg_path: Path to the model.svg annotation file

    Returns:
        Dict with parsed_layout and layout_graph, or None if parsing fails
    """
    try:
        tree = ET.parse(svg_path)
        root = tree.getroot()
    except ET.ParseError as e:
        logger.warning(f"Failed to parse SVG {svg_path}: {e}")
        return None

    # Collect all room polygons
    rooms_data = []

    # Search for polygon/polyline elements with class/id attributes
    for elem in root.iter():
        tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag

        if tag not in ("polygon", "polyline", "path", "rect"):
            continue

        # Get room class from class attribute, id, or parent group
        room_class = (
            elem.get("class", "")
            or elem.get("id", "")
            or _get_parent_class(root, elem)
        )

        if not room_class:
            continue

        # Skip non-room elements (walls, doors, windows, icons)
        skip_classes = {"Wall", "wall", "Window", "window", "Door", "door",
                        "Icon", "icon", "Stair", "stair", "Separator", "separator"}
        if any(s in room_class for s in skip_classes):
            continue

        # Get polygon points
        points = []
        if tag == "polygon" or tag == "polyline":
            points_str = elem.get("points", "")
            if points_str:
                points = _parse_polygon_points(points_str)
        elif tag == "rect":
            x = float(elem.get("x", 0))
            y = float(elem.get("y", 0))
            w = float(elem.get("width", 0))
            h = float(elem.get("height", 0))
            points = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]

        if len(points) < 3:
            continue

        # Get room type
        # Extract first relevant class name token
        class_token = room_class.split()[0].strip()
        room_type = _map_room_type(class_token)

        area = _polygon_area(points)
        bbox = _polygon_to_bbox(points)

        if area < 1.0:  # Skip tiny elements
            continue

        rooms_data.append({
            "room_type": room_type,
            "points": points,
            "bbox": bbox,
            "area": area,
            "class": class_token,
        })

    if not rooms_data:
        return None

    # Compute normalisation bounds (full drawing bounds)
    all_xs = [p[0] for r in rooms_data for p in r["points"]]
    all_ys = [p[1] for r in rooms_data for p in r["points"]]
    global_x_min, global_x_max = min(all_xs), max(all_xs)
    global_y_min, global_y_max = min(all_ys), max(all_ys)
    w_range = max(global_x_max - global_x_min, 1e-6)
    h_range = max(global_y_max - global_y_min, 1e-6)

    # Total plot area estimate (using drawing bounds)
    # Assume 1 SVG unit ≈ 1cm, so convert to m²
    plot_area_sqm = (w_range * h_range) / 10000.0  # cm² → m²
    plot_area_sqm = max(plot_area_sqm, 30.0)  # Minimum 30 m²

    # Build RoomSpecs and RoomLayouts
    room_specs = []
    room_layouts = []

    # Track master bedroom (largest bedroom)
    bedroom_sizes = [(i, r["area"]) for i, r in enumerate(rooms_data)
                     if r["room_type"] in (RoomType.BEDROOM, RoomType.MASTER_BEDROOM)]
    master_idx = max(bedroom_sizes, key=lambda x: x[1])[0] if bedroom_sizes else -1

    for i, room in enumerate(rooms_data):
        room_type = room["room_type"]

        # Promote largest bedroom to master
        if i == master_idx:
            room_type = RoomType.MASTER_BEDROOM

        # Convert area from SVG units to m²
        area_sqm = room["area"] / 10000.0  # cm² → m²
        area_sqm = max(area_sqm, 3.0)  # Minimum 3 m²

        spec = RoomSpec(
            room_id=f"room_{i + 1}",
            room_type=room_type,
            target_area_sqm=round(area_sqm, 1),
        )
        room_specs.append(spec)

        # Normalise bounding box to [0,1]
        x_min, y_min, x_max, y_max = room["bbox"]
        bbox = BoundingBox(
            x_min=max(0, (x_min - global_x_min) / w_range),
            y_min=max(0, (y_min - global_y_min) / h_range),
            x_max=min(1, (x_max - global_x_min) / w_range),
            y_max=min(1, (y_max - global_y_min) / h_range),
        )

        room_layouts.append(RoomLayout(room_spec=spec, bbox=bbox))

    # Infer adjacency edges (rooms with overlapping/touching bounding boxes)
    adjacency_edges = _infer_adjacency(room_layouts)

    parsed = ParsedLayout(
        rooms=room_specs,
        plot_area_sqm=round(plot_area_sqm, 1),
        facing=CompassFacing.NORTH,
        adjacency_constraints=adjacency_edges,
        vastu_enabled=False,
    )

    layout_graph = LayoutGraph(
        rooms=room_layouts,
        adjacency_edges=adjacency_edges,
        plot_area_sqm=round(plot_area_sqm, 1),
        facing=CompassFacing.NORTH,
        generation_mode="heuristic",
    )

    return {
        "parsed_layout": parsed.model_dump(),
        "layout_graph": layout_graph.model_dump(),
    }


def _get_parent_class(root: ET.Element, elem: ET.Element) -> str:
    """Try to get class from parent <g> element."""
    # Build parent map
    parent_map = {child: parent for parent in root.iter() for child in parent}
    parent = parent_map.get(elem)
    if parent is not None:
        return parent.get("class", "") or parent.get("id", "")
    return ""


def _infer_adjacency(rooms: list[RoomLayout], threshold: float = 0.02) -> list[AdjacencyEdge]:
    """Infer adjacency from bounding box proximity."""
    edges = []

    for i in range(len(rooms)):
        for j in range(i + 1, len(rooms)):
            bi = rooms[i].bbox
            bj = rooms[j].bbox

            # Check if bounding boxes overlap or are within threshold
            h_overlap = min(bi.x_max, bj.x_max) - max(bi.x_min, bj.x_min)
            v_overlap = min(bi.y_max, bj.y_max) - max(bi.y_min, bj.y_min)

            # Adjacent if they share a wall (overlap in one dimension, touching in other)
            h_adjacent = h_overlap > threshold and abs(bi.y_max - bj.y_min) < threshold
            v_adjacent = v_overlap > threshold and abs(bi.x_max - bj.x_min) < threshold

            # Or overlapping
            overlapping = h_overlap > threshold and v_overlap > threshold

            if h_adjacent or v_adjacent or overlapping:
                conn = ConnectionType.DOOR
                # Kitchen/dining → OPENING
                types = {rooms[i].room_spec.room_type, rooms[j].room_spec.room_type}
                if types & {RoomType.KITCHEN, RoomType.DINING, RoomType.LIVING_ROOM}:
                    conn = ConnectionType.OPENING

                edges.append(AdjacencyEdge(
                    room_a_id=rooms[i].room_spec.room_id,
                    room_b_id=rooms[j].room_spec.room_id,
                    connection_type=conn,
                    required=True,
                ))

    return edges


# ─── Batch Loading ───────────────────────────────────────────────────────────


def load_cubicasa_dataset(
    data_dir: str,
    split: str = "train",
    max_samples: int | None = None,
) -> list[dict]:
    """Load CubiCasa5K dataset for a given split.

    Args:
        data_dir: Path to extracted CubiCasa5K root (containing cubicasa5k/ and txt files)
        split: One of 'train', 'val', 'test'
        max_samples: Optional max number of samples to load

    Returns:
        List of dicts with parsed_layout and layout_graph keys
    """
    split_file = os.path.join(data_dir, f"{split}.txt")

    if not os.path.exists(split_file):
        logger.error(f"Split file not found: {split_file}")
        return []

    # Read sample paths
    with open(split_file, "r") as f:
        sample_paths = [line.strip() for line in f if line.strip()]

    if max_samples:
        sample_paths = sample_paths[:max_samples]

    samples = []
    for sample_path in sample_paths:
        # CubiCasa format: each line is a relative path to sample directory
        svg_path = os.path.join(data_dir, sample_path, "model.svg")

        if not os.path.exists(svg_path):
            # Try alternative path structure
            svg_path = os.path.join(data_dir, "cubicasa5k", sample_path, "model.svg")

        if not os.path.exists(svg_path):
            logger.debug(f"SVG not found: {svg_path}")
            continue

        result = parse_cubicasa_svg(svg_path)
        if result and len(result["layout_graph"]["rooms"]) >= 2:
            samples.append(result)

    logger.info(f"Loaded {len(samples)} CubiCasa5K samples from {split} split")
    return samples
