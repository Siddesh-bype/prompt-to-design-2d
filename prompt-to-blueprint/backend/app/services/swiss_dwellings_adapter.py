"""
Modified Swiss Dwellings (MSD) Adapter — Converts MSD graph data to our LayoutGraph format.

Actual MSD v2 dataset structure:
    data/modified-swiss-dwellings-v2/
    ├── train/
    │   ├── graph_in/    (input graphs — zoning_type only)
    │   ├── graph_out/   (output graphs — geometry + room_type + centroid)
    │   ├── struct_in/   (numpy adjacency)
    │   └── full_out/
    └── test/
        └── (same sub-structure)

Each graph_out pickle contains a networkx Graph where:
  - Nodes have: geometry (list of polygon coords), room_type (int), centroid (tensor)
  - Edges have: connectivity ('door', 'window', 'wall', etc.)

Room type integers mapping (from MSD documentation):
  0=LivingRoom, 1=Kitchen, 2=Bedroom, 3=Bathroom, 4=Hallway/Corridor,
  5=Balcony, 6=Dining, 7=Storage/Utility, 8=Garage, 9=Study/Office
"""

from __future__ import annotations

import glob
import logging
import math
import os
import pickle
import random
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

# ─── Integer Room Type Mapping ───────────────────────────────────────────────

MSD_INT_ROOM_MAP: dict[int, RoomType] = {
    0: RoomType.LIVING_ROOM,
    1: RoomType.KITCHEN,
    2: RoomType.BEDROOM,
    3: RoomType.BATHROOM,
    4: RoomType.CORRIDOR,
    5: RoomType.BALCONY,
    6: RoomType.DINING,
    7: RoomType.UTILITY,
    8: RoomType.GARAGE,
    9: RoomType.STUDY,
}


def _map_room_type_int(room_type_int: int) -> RoomType:
    """Map an MSD integer room type to our RoomType enum."""
    return MSD_INT_ROOM_MAP.get(room_type_int, RoomType.UTILITY)


def _geometry_to_bbox(geometry: list) -> tuple[float, float, float, float] | None:
    """Convert polygon geometry to (x_min, y_min, x_max, y_max)."""
    try:
        if not geometry:
            return None
        xs = [p[0] for p in geometry]
        ys = [p[1] for p in geometry]
        return (min(xs), min(ys), max(xs), max(ys))
    except Exception:
        return None


def _polygon_area(geometry: list) -> float:
    """Compute polygon area via shoelace formula."""
    try:
        n = len(geometry)
        if n < 3:
            return 0.0
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += geometry[i][0] * geometry[j][1]
            area -= geometry[j][0] * geometry[i][1]
        return abs(area) / 2.0
    except Exception:
        return 0.0


def _connection_type_from_str(conn: str) -> ConnectionType:
    """Map MSD connectivity string to ConnectionType."""
    conn_lower = conn.lower() if conn else ""
    if "door" in conn_lower:
        return ConnectionType.DOOR
    elif "window" in conn_lower or "opening" in conn_lower:
        return ConnectionType.OPENING
    return ConnectionType.WALL


# ─── Graph Parsing ───────────────────────────────────────────────────────────


def parse_msd_graph(graph_path: str) -> dict | None:
    """Parse a single MSD graph_out pickle file.

    Args:
        graph_path: Path to the graph_out pickle file

    Returns:
        Dict with parsed_layout and layout_graph, or None if parsing fails
    """
    try:
        with open(graph_path, "rb") as f:
            graph = pickle.load(f)
    except Exception as e:
        logger.debug(f"Failed to load {graph_path}: {e}")
        return None

    nodes = list(graph.nodes(data=True))
    edges = list(graph.edges(data=True))

    if len(nodes) < 2:
        return None

    # Collect raw bboxes for normalisation
    raw_rooms = []
    for node_id, attrs in nodes:
        geometry = attrs.get("geometry", [])
        room_type_int = attrs.get("room_type", 7)

        bbox_raw = _geometry_to_bbox(geometry)
        if bbox_raw is None:
            continue

        area = _polygon_area(geometry)
        room_type = _map_room_type_int(room_type_int)

        raw_rooms.append({
            "node_id": node_id,
            "room_type": room_type,
            "area": area,
            "bbox": bbox_raw,
        })

    if len(raw_rooms) < 2:
        return None

    # Cap at 30 rooms (keep largest by area to preserve main rooms)
    if len(raw_rooms) > 30:
        raw_rooms.sort(key=lambda r: r["area"], reverse=True)
        raw_rooms = raw_rooms[:30]

    # Compute global bounding box for normalisation
    all_x_min = min(r["bbox"][0] for r in raw_rooms)
    all_y_min = min(r["bbox"][1] for r in raw_rooms)
    all_x_max = max(r["bbox"][2] for r in raw_rooms)
    all_y_max = max(r["bbox"][3] for r in raw_rooms)

    width = max(all_x_max - all_x_min, 1e-6)
    height = max(all_y_max - all_y_min, 1e-6)
    scale = max(width, height)

    # Build rooms
    room_specs = []
    room_layouts = []
    node_to_room_id = {}

    for i, raw in enumerate(raw_rooms):
        room_id = f"room_{i + 1}"
        node_to_room_id[raw["node_id"]] = room_id

        area_sqm = max(4.0, min(raw["area"], 200.0))

        spec = RoomSpec(
            room_id=room_id,
            room_type=raw["room_type"],
            target_area_sqm=round(area_sqm, 1),
        )
        room_specs.append(spec)

        x_min = max(0.0, (raw["bbox"][0] - all_x_min) / scale)
        y_min = max(0.0, (raw["bbox"][1] - all_y_min) / scale)
        x_max = min(1.0, (raw["bbox"][2] - all_x_min) / scale)
        y_max = min(1.0, (raw["bbox"][3] - all_y_min) / scale)

        # Clamp
        x_max = max(x_max, x_min + 0.01)
        y_max = max(y_max, y_min + 0.01)
        x_max = min(x_max, 1.0)
        y_max = min(y_max, 1.0)

        try:
            room_layouts.append(RoomLayout(
                room_spec=spec,
                bbox=BoundingBox(
                    x_min=round(x_min, 6),
                    y_min=round(y_min, 6),
                    x_max=round(x_max, 6),
                    y_max=round(y_max, 6),
                ),
            ))
        except Exception:
            continue

    if len(room_layouts) < 2:
        return None

    # Build adjacency edges
    adjacency_edges = []
    for u, v, edge_attrs in edges:
        room_a = node_to_room_id.get(u)
        room_b = node_to_room_id.get(v)
        if room_a and room_b:
            conn_str = edge_attrs.get("connectivity", "wall")
            adjacency_edges.append(AdjacencyEdge(
                room_a_id=room_a,
                room_b_id=room_b,
                connection_type=_connection_type_from_str(conn_str),
                required=True,
            ))

    # Estimate plot area
    total_area = sum(r["area"] for r in raw_rooms)
    plot_area = max(30.0, min(total_area * 1.2, 2000.0))

    parsed = ParsedLayout(
        rooms=room_specs,
        plot_area_sqm=round(plot_area, 1),
        facing=random.choice(list(CompassFacing)),
        adjacency_constraints=adjacency_edges,
    )

    layout_graph = LayoutGraph(
        rooms=room_layouts,
        adjacency_edges=adjacency_edges,
        plot_area_sqm=round(plot_area, 1),
        facing=parsed.facing,
        generation_mode="heuristic",
    )

    return {
        "parsed_layout": parsed.model_dump(),
        "layout_graph": layout_graph.model_dump(),
    }


# ─── Batch Loading ───────────────────────────────────────────────────────────


def load_swiss_dwellings_dataset(
    data_dir: str,
    max_samples: int | None = None,
) -> list[dict]:
    """Load Modified Swiss Dwellings dataset.

    Searches for graph_out pickle files in the dataset directory.

    Args:
        data_dir: Path to extracted MSD root directory
        max_samples: Optional max number of samples

    Returns:
        List of dicts with parsed_layout and layout_graph keys
    """
    # Find graph_out pickle files
    graph_patterns = [
        os.path.join(data_dir, "train", "graph_out", "*.pickle"),
        os.path.join(data_dir, "test", "graph_out", "*.pickle"),
        os.path.join(data_dir, "graph_out", "*.pickle"),
        os.path.join(data_dir, "*.pickle"),
        os.path.join(data_dir, "**", "graph_out", "*.pickle"),
    ]

    pickle_files = []
    for pattern in graph_patterns:
        found = glob.glob(pattern, recursive=True)
        pickle_files.extend(found)
        if pickle_files:
            break

    if not pickle_files:
        # Try .pkl extension too
        for pattern in graph_patterns:
            found = glob.glob(pattern.replace(".pickle", ".pkl"), recursive=True)
            pickle_files.extend(found)
            if pickle_files:
                break

    pickle_files = sorted(set(pickle_files))

    if not pickle_files:
        logger.warning(f"No MSD pickle files found in {data_dir}")
        return []

    logger.info(f"Found {len(pickle_files)} MSD graph files")

    if max_samples and len(pickle_files) > max_samples:
        random.shuffle(pickle_files)
        pickle_files = pickle_files[:max_samples]

    samples = []
    errors = 0

    for pkl_path in pickle_files:
        result = parse_msd_graph(pkl_path)
        if result:
            samples.append(result)
        else:
            errors += 1

    logger.info(f"MSD: Loaded {len(samples)} samples ({errors} errors)")
    return samples
