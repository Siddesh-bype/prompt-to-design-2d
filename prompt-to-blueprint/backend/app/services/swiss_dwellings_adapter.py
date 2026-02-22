"""
Modified Swiss Dwellings (MSD) Adapter — Converts MSD graph data to our LayoutGraph format.

MSD dataset structure (after extraction):
    data/
    └── modified_swiss_dwellings/
        ├── metadata.csv              (building metadata)
        └── graphs/
            └── <building_id>.pkl     (networkx graph per floor plan)

Each graph node represents a room with attributes:
    - shape: Shapely polygon (room boundary)
    - room_type: str (room category)
    - zoning_type: str (zoning category)

Each graph edge represents adjacency between rooms.
"""

from __future__ import annotations

import glob
import logging
import os
import pickle
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

# Map MSD room_type strings → our RoomType enum
MSD_ROOM_MAP: dict[str, RoomType] = {
    # Common MSD categories (German/English mix)
    "kitchen": RoomType.KITCHEN,
    "küche": RoomType.KITCHEN,
    "living": RoomType.LIVING_ROOM,
    "living_room": RoomType.LIVING_ROOM,
    "livingroom": RoomType.LIVING_ROOM,
    "wohnzimmer": RoomType.LIVING_ROOM,
    "bedroom": RoomType.BEDROOM,
    "schlafzimmer": RoomType.BEDROOM,
    "room": RoomType.BEDROOM,
    "zimmer": RoomType.BEDROOM,
    "bathroom": RoomType.BATHROOM,
    "bath": RoomType.BATHROOM,
    "badezimmer": RoomType.BATHROOM,
    "wc": RoomType.TOILET,
    "toilet": RoomType.TOILET,
    "toilette": RoomType.TOILET,
    "corridor": RoomType.CORRIDOR,
    "hallway": RoomType.CORRIDOR,
    "flur": RoomType.CORRIDOR,
    "gang": RoomType.CORRIDOR,
    "entrance": RoomType.CORRIDOR,
    "eingang": RoomType.CORRIDOR,
    "balcony": RoomType.BALCONY,
    "balkon": RoomType.BALCONY,
    "loggia": RoomType.BALCONY,
    "terrace": RoomType.BALCONY,
    "terrasse": RoomType.BALCONY,
    "storage": RoomType.UTILITY,
    "abstellraum": RoomType.UTILITY,
    "utility": RoomType.UTILITY,
    "hauswirtschaft": RoomType.UTILITY,
    "laundry": RoomType.UTILITY,
    "waschküche": RoomType.UTILITY,
    "garage": RoomType.GARAGE,
    "parking": RoomType.GARAGE,
    "study": RoomType.STUDY,
    "arbeitszimmer": RoomType.STUDY,
    "office": RoomType.STUDY,
    "büro": RoomType.STUDY,
    "dining": RoomType.DINING,
    "dining_room": RoomType.DINING,
    "esszimmer": RoomType.DINING,
}


def _map_room_type(msd_type: str) -> RoomType:
    """Map an MSD room type string to our RoomType enum."""
    lower = msd_type.lower().strip()

    # Exact match
    if lower in MSD_ROOM_MAP:
        return MSD_ROOM_MAP[lower]

    # Partial match
    for key, room_type in MSD_ROOM_MAP.items():
        if key in lower or lower in key:
            return room_type

    # Default fallback
    return RoomType.UTILITY


# ─── Graph Parsing ───────────────────────────────────────────────────────────


def parse_msd_graph(graph_path: str) -> Optional[dict]:
    """Parse a single MSD graph file (.pkl or .pt).

    Args:
        graph_path: Path to the graph file

    Returns:
        Dict with parsed_layout and layout_graph, or None if parsing fails
    """
    try:
        with open(graph_path, "rb") as f:
            graph = pickle.load(f)
    except Exception as e:
        logger.warning(f"Failed to load graph {graph_path}: {e}")
        return None

    # Extract nodes (rooms) and edges
    nodes = []
    edges = []

    try:
        # networkx graph
        if hasattr(graph, "nodes") and hasattr(graph, "edges"):
            for node_id, attrs in graph.nodes(data=True):
                nodes.append({
                    "id": str(node_id),
                    "attrs": attrs,
                })
            for u, v, attrs in graph.edges(data=True):
                edges.append((str(u), str(v), attrs))
        # Dict-based format
        elif isinstance(graph, dict):
            if "nodes" in graph:
                for n in graph["nodes"]:
                    if isinstance(n, dict):
                        nodes.append({"id": str(n.get("id", len(nodes))), "attrs": n})
                    else:
                        nodes.append({"id": str(len(nodes)), "attrs": {}})
            if "edges" in graph:
                for e in graph["edges"]:
                    if isinstance(e, (list, tuple)) and len(e) >= 2:
                        edges.append((str(e[0]), str(e[1]), e[2] if len(e) > 2 else {}))
        else:
            logger.warning(f"Unknown graph format: {type(graph)}")
            return None
    except Exception as e:
        logger.warning(f"Failed to parse graph structure: {e}")
        return None

    if len(nodes) < 2:
        return None

    # Extract room data from node attributes
    rooms_data = []
    global_x_min = global_y_min = float("inf")
    global_x_max = global_y_max = float("-inf")

    for node in nodes:
        attrs = node["attrs"]

        # Get room type
        room_type_str = (
            attrs.get("room_type", "")
            or attrs.get("roomtype", "")
            or attrs.get("type", "")
            or attrs.get("category", "")
            or attrs.get("label", "")
            or "room"
        )
        room_type = _map_room_type(str(room_type_str))

        # Get room shape / polygon / bounding box
        shape = attrs.get("shape", None)
        bbox = None
        area = None

        if shape is not None:
            # Shapely polygon
            try:
                if hasattr(shape, "bounds"):
                    x_min, y_min, x_max, y_max = shape.bounds
                    bbox = (x_min, y_min, x_max, y_max)
                    area = float(shape.area)
                elif hasattr(shape, "__iter__"):
                    # List of coordinates
                    coords = list(shape)
                    if coords:
                        xs = [c[0] for c in coords]
                        ys = [c[1] for c in coords]
                        bbox = (min(xs), min(ys), max(xs), max(ys))
                        area = abs(max(xs) - min(xs)) * abs(max(ys) - min(ys))
            except Exception:
                pass

        # Try other attributes for bounds
        if bbox is None:
            if all(k in attrs for k in ("x_min", "y_min", "x_max", "y_max")):
                bbox = (attrs["x_min"], attrs["y_min"], attrs["x_max"], attrs["y_max"])
            elif all(k in attrs for k in ("x", "y", "width", "height")):
                x, y, w, h = attrs["x"], attrs["y"], attrs["width"], attrs["height"]
                bbox = (x, y, x + w, y + h)
            elif "bounds" in attrs:
                b = attrs["bounds"]
                if len(b) == 4:
                    bbox = tuple(b)
            elif "position" in attrs and "size" in attrs:
                pos = attrs["position"]
                size = attrs["size"]
                bbox = (pos[0], pos[1], pos[0] + size[0], pos[1] + size[1])

        if bbox is None:
            continue

        if area is None:
            area = abs(bbox[2] - bbox[0]) * abs(bbox[3] - bbox[1])

        if area < 0.5:  # Skip tiny elements
            continue

        # Update global bounds
        global_x_min = min(global_x_min, bbox[0])
        global_y_min = min(global_y_min, bbox[1])
        global_x_max = max(global_x_max, bbox[2])
        global_y_max = max(global_y_max, bbox[3])

        rooms_data.append({
            "id": node["id"],
            "room_type": room_type,
            "bbox": bbox,
            "area": area,
        })

    if len(rooms_data) < 2:
        return None

    # Normalise bounding boxes
    w_range = max(global_x_max - global_x_min, 1e-6)
    h_range = max(global_y_max - global_y_min, 1e-6)

    # Plot area in m² (assume coordinates are in metres)
    plot_area_sqm = w_range * h_range
    if plot_area_sqm > 10000:
        # Probably in cm or mm — convert
        plot_area_sqm /= 10000.0
    plot_area_sqm = max(plot_area_sqm, 30.0)

    room_specs = []
    room_layouts = []
    id_to_idx = {}

    # Find master bedroom (largest bedroom)
    bedroom_sizes = [(i, r["area"]) for i, r in enumerate(rooms_data)
                     if r["room_type"] in (RoomType.BEDROOM, RoomType.MASTER_BEDROOM)]
    master_idx = max(bedroom_sizes, key=lambda x: x[1])[0] if bedroom_sizes else -1

    for i, room in enumerate(rooms_data):
        room_type = room["room_type"]
        if i == master_idx:
            room_type = RoomType.MASTER_BEDROOM

        # Convert area
        area_sqm = room["area"]
        if area_sqm > 1000:
            area_sqm /= 10000.0  # cm² → m²
        area_sqm = max(area_sqm, 3.0)

        spec = RoomSpec(
            room_id=f"room_{i + 1}",
            room_type=room_type,
            target_area_sqm=round(area_sqm, 1),
        )
        room_specs.append(spec)

        # Normalise bbox to [0,1]
        x_min, y_min, x_max, y_max = room["bbox"]
        bbox = BoundingBox(
            x_min=max(0, (x_min - global_x_min) / w_range),
            y_min=max(0, (y_min - global_y_min) / h_range),
            x_max=min(1, (x_max - global_x_min) / w_range),
            y_max=min(1, (y_max - global_y_min) / h_range),
        )

        room_layouts.append(RoomLayout(room_spec=spec, bbox=bbox))
        id_to_idx[room["id"]] = i

    # Build adjacency edges from graph edges
    adjacency_edges = []
    for u, v, attrs in edges:
        if u in id_to_idx and v in id_to_idx:
            conn_type = ConnectionType.DOOR
            types = {rooms_data[id_to_idx[u]]["room_type"],
                     rooms_data[id_to_idx[v]]["room_type"]}
            if types & {RoomType.KITCHEN, RoomType.DINING, RoomType.LIVING_ROOM}:
                conn_type = ConnectionType.OPENING

            adjacency_edges.append(AdjacencyEdge(
                room_a_id=room_specs[id_to_idx[u]].room_id,
                room_b_id=room_specs[id_to_idx[v]].room_id,
                connection_type=conn_type,
                required=True,
            ))

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


# ─── Batch Loading ───────────────────────────────────────────────────────────


def load_swiss_dwellings_dataset(
    data_dir: str,
    max_samples: int | None = None,
) -> list[dict]:
    """Load Modified Swiss Dwellings dataset.

    Args:
        data_dir: Path to extracted MSD root directory
        max_samples: Optional max number of samples

    Returns:
        List of dicts with parsed_layout and layout_graph keys
    """
    # Find graph files
    graph_files = sorted(
        glob.glob(os.path.join(data_dir, "**", "*.pkl"), recursive=True)
    )

    if not graph_files:
        # Also try .pt files
        graph_files = sorted(
            glob.glob(os.path.join(data_dir, "**", "*.pt"), recursive=True)
        )

    if not graph_files:
        # Try JSON format
        graph_files = sorted(
            glob.glob(os.path.join(data_dir, "**", "*.json"), recursive=True)
        )
        if graph_files:
            return _load_json_format(graph_files, max_samples)

    if max_samples:
        graph_files = graph_files[:max_samples]

    logger.info(f"Found {len(graph_files)} MSD graph files in {data_dir}")

    samples = []
    for path in graph_files:
        result = parse_msd_graph(path)
        if result and len(result["layout_graph"]["rooms"]) >= 2:
            samples.append(result)

    logger.info(f"Loaded {len(samples)} MSD samples")
    return samples


def _load_json_format(json_files: list[str], max_samples: int | None = None) -> list[dict]:
    """Fallback: load MSD data from JSON files."""
    import json

    if max_samples:
        json_files = json_files[:max_samples]

    samples = []
    for path in json_files:
        try:
            with open(path, "r") as f:
                data = json.load(f)

            # If already in our format
            if "parsed_layout" in data and "layout_graph" in data:
                samples.append(data)
                continue

            # If it's a list of apartments/floor plans
            items = data if isinstance(data, list) else [data]
            for item in items:
                if isinstance(item, dict):
                    result = _convert_json_item(item)
                    if result:
                        samples.append(result)
        except Exception as e:
            logger.warning(f"Failed to load JSON {path}: {e}")

    logger.info(f"Loaded {len(samples)} MSD samples from JSON")
    return samples


def _convert_json_item(item: dict) -> Optional[dict]:
    """Convert a single JSON item from MSD to our format."""
    rooms = item.get("rooms", item.get("areas", []))
    if not rooms or len(rooms) < 2:
        return None

    room_specs = []
    room_layouts = []

    for i, room in enumerate(rooms):
        if isinstance(room, dict):
            rt_str = room.get("type", room.get("room_type", room.get("category", "room")))
            room_type = _map_room_type(str(rt_str))

            area = float(room.get("area", room.get("size", 10.0)))
            if area > 1000:
                area /= 10000.0

            spec = RoomSpec(
                room_id=f"room_{i + 1}",
                room_type=room_type,
                target_area_sqm=round(max(area, 3.0), 1),
            )
            room_specs.append(spec)

            # Get bbox
            bbox_data = room.get("bbox", room.get("bounds", None))
            if bbox_data and len(bbox_data) == 4:
                bbox = BoundingBox(
                    x_min=max(0, min(1, float(bbox_data[0]))),
                    y_min=max(0, min(1, float(bbox_data[1]))),
                    x_max=max(0, min(1, float(bbox_data[2]))),
                    y_max=max(0, min(1, float(bbox_data[3]))),
                )
            else:
                # Generate placeholder
                row = i // 3
                col = i % 3
                bbox = BoundingBox(
                    x_min=col * 0.33,
                    y_min=row * 0.33,
                    x_max=min(1, (col + 1) * 0.33),
                    y_max=min(1, (row + 1) * 0.33),
                )

            room_layouts.append(RoomLayout(room_spec=spec, bbox=bbox))

    plot_area = float(item.get("plot_area", item.get("total_area", 100.0)))

    parsed = ParsedLayout(
        rooms=room_specs,
        plot_area_sqm=round(plot_area, 1),
        facing=CompassFacing.NORTH,
    )

    layout_graph = LayoutGraph(
        rooms=room_layouts,
        adjacency_edges=[],
        plot_area_sqm=round(plot_area, 1),
        facing=CompassFacing.NORTH,
        generation_mode="heuristic",
    )

    return {
        "parsed_layout": parsed.model_dump(),
        "layout_graph": layout_graph.model_dump(),
    }
