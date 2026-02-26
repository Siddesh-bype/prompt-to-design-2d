"""
CubiCasa5K Adapter — Converts CubiCasa5K annotations to our LayoutGraph format.

Supports two data formats:
1. COCO JSON (cubicasa5k_coco/) — fast, pre-processed bounding boxes (preferred)
2. SVG annotations (cubicasa5k/) — slower, parses polygon annotations

COCO structure:
    data/cubicasa5k_coco/
    ├── train_coco_pt.json   (4200 images, 49K room annotations)
    ├── val_coco_pt.json
    └── test_coco_pt.json
"""

from __future__ import annotations

import glob
import json
import logging
import os
import random
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

# ─── Room Types for random assignment ────────────────────────────────────────

MAIN_ROOM_TYPES = [
    RoomType.LIVING_ROOM, RoomType.BEDROOM, RoomType.KITCHEN,
    RoomType.BATHROOM, RoomType.MASTER_BEDROOM, RoomType.DINING,
    RoomType.STUDY, RoomType.CORRIDOR, RoomType.UTILITY, RoomType.BALCONY,
]


def _assign_room_types(n_rooms: int, seed: int = 0) -> list[RoomType]:
    """Assign plausible room types to rooms based on count."""
    rng = random.Random(seed)

    if n_rooms <= 2:
        return [RoomType.LIVING_ROOM, RoomType.BEDROOM][:n_rooms]

    # Always include at least: living, kitchen, bathroom
    types = [RoomType.LIVING_ROOM, RoomType.KITCHEN, RoomType.BATHROOM]

    # Add bedrooms
    n_bedrooms = max(1, n_rooms // 3)
    if n_bedrooms >= 1:
        types.append(RoomType.MASTER_BEDROOM)
    for _ in range(n_bedrooms - 1):
        types.append(RoomType.BEDROOM)

    # Fill remaining with random types
    fill_types = [RoomType.DINING, RoomType.STUDY, RoomType.CORRIDOR,
                  RoomType.UTILITY, RoomType.BALCONY, RoomType.BATHROOM]
    while len(types) < n_rooms:
        types.append(rng.choice(fill_types))

    # Truncate if too many
    types = types[:n_rooms]
    rng.shuffle(types)
    return types


# ─── COCO JSON Loading ──────────────────────────────────────────────────────


def _load_coco_format(
    coco_dir: str,
    split: str = "train",
    max_samples: int | None = None,
) -> list[dict]:
    """Load CubiCasa5K from COCO JSON annotations (fast path).

    Args:
        coco_dir: Path to cubicasa5k_coco directory
        split: One of 'train', 'val', 'test'
        max_samples: Max number of floor plans to load

    Returns:
        List of dicts with parsed_layout and layout_graph keys
    """
    json_file = os.path.join(coco_dir, f"{split}_coco_pt.json")
    if not os.path.exists(json_file):
        logger.warning(f"COCO JSON not found: {json_file}")
        return []

    logger.info(f"Loading CubiCasa5K from COCO JSON: {json_file}")
    with open(json_file, "r") as f:
        coco_data = json.load(f)

    images = coco_data.get("images", [])
    annotations = coco_data.get("annotations", [])

    # Group annotations by image_id
    img_annots: dict[int, list] = {}
    for ann in annotations:
        if ann.get("category_id") == 2:  # room annotations only
            img_id = ann["image_id"]
            img_annots.setdefault(img_id, []).append(ann)

    # Build image dimension lookup
    img_dims = {img["id"]: (img["width"], img["height"]) for img in images}

    if max_samples and len(img_annots) > max_samples:
        selected_ids = random.sample(list(img_annots.keys()), max_samples)
        img_annots = {k: img_annots[k] for k in selected_ids}

    samples = []
    errors = 0

    for img_id, room_anns in img_annots.items():
        try:
            sample = _coco_image_to_sample(img_id, room_anns, img_dims)
            if sample:
                samples.append(sample)
            else:
                errors += 1
        except Exception as e:
            errors += 1
            continue

    logger.info(f"CubiCasa5K COCO: Loaded {len(samples)} floor plans from {split} ({errors} errors)")
    return samples


def _coco_image_to_sample(
    img_id: int,
    room_anns: list[dict],
    img_dims: dict[int, tuple[int, int]],
) -> dict | None:
    """Convert COCO annotations for one image to our format."""
    if len(room_anns) < 2:
        return None

    # Cap rooms at 30
    if len(room_anns) > 30:
        room_anns.sort(key=lambda a: a.get("area", 0), reverse=True)
        room_anns = room_anns[:30]

    w, h = img_dims.get(img_id, (1000, 1000))
    scale = max(w, h, 1)

    # Assign room types
    room_types = _assign_room_types(len(room_anns), seed=img_id)

    room_specs = []
    room_layouts = []

    for i, ann in enumerate(room_anns):
        # COCO bbox: [x, y, width, height]
        bx, by, bw, bh = ann["bbox"]

        # Normalise to [0, 1]
        x_min = max(0.0, bx / w)
        y_min = max(0.0, by / h)
        x_max = min(1.0, (bx + bw) / w)
        y_max = min(1.0, (by + bh) / h)

        # Skip tiny rooms
        if (x_max - x_min) < 0.01 or (y_max - y_min) < 0.01:
            continue

        # Estimate area in m² (assume image ≈ 200m² plot)
        area_frac = (x_max - x_min) * (y_max - y_min)
        area_sqm = max(4.0, min(area_frac * 200.0, 200.0))

        room_type = room_types[i] if i < len(room_types) else RoomType.UTILITY

        spec = RoomSpec(
            room_id=f"room_{i + 1}",
            room_type=room_type,
            target_area_sqm=round(area_sqm, 1),
        )
        room_specs.append(spec)

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

    # Infer adjacency
    adjacency_edges = _infer_adjacency(room_layouts)

    plot_area_sqm = max(30.0, min(sum(s.target_area_sqm for s in room_specs) * 1.2, 2000.0))

    parsed = ParsedLayout(
        rooms=room_specs[:len(room_layouts)],
        plot_area_sqm=round(plot_area_sqm, 1),
        facing=random.choice(list(CompassFacing)),
        adjacency_constraints=adjacency_edges,
    )

    layout_graph = LayoutGraph(
        rooms=room_layouts,
        adjacency_edges=adjacency_edges,
        plot_area_sqm=round(plot_area_sqm, 1),
        facing=parsed.facing,
        generation_mode="heuristic",
    )

    return {
        "parsed_layout": parsed.model_dump(),
        "layout_graph": layout_graph.model_dump(),
    }


# ─── Adjacency Inference ────────────────────────────────────────────────────


def _infer_adjacency(rooms: list[RoomLayout], threshold: float = 0.02) -> list[AdjacencyEdge]:
    """Infer adjacency from bounding box proximity."""
    edges = []

    for i in range(len(rooms)):
        for j in range(i + 1, len(rooms)):
            bi = rooms[i].bbox
            bj = rooms[j].bbox

            h_overlap = min(bi.x_max, bj.x_max) - max(bi.x_min, bj.x_min)
            v_overlap = min(bi.y_max, bj.y_max) - max(bi.y_min, bj.y_min)

            h_adjacent = h_overlap > threshold and abs(bi.y_max - bj.y_min) < threshold
            v_adjacent = v_overlap > threshold and abs(bi.x_max - bj.x_min) < threshold
            overlapping = h_overlap > threshold and v_overlap > threshold

            if h_adjacent or v_adjacent or overlapping:
                conn = ConnectionType.DOOR
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


# ─── Batch Loading (Main Entry Point) ───────────────────────────────────────


def load_cubicasa_dataset(
    data_dir: str,
    split: str = "train",
    max_samples: int | None = None,
) -> list[dict]:
    """Load CubiCasa5K dataset — uses COCO JSON if available, falls back to SVG.

    Args:
        data_dir: Path to CubiCasa5K root directory
        split: One of 'train', 'val', 'test'
        max_samples: Optional max number of samples

    Returns:
        List of dicts with parsed_layout and layout_graph keys
    """
    # Prefer COCO JSON format (much faster)
    coco_dir = os.path.join(data_dir, "..", "cubicasa5k_coco")
    if not os.path.exists(coco_dir):
        coco_dir = os.path.join(data_dir, "cubicasa5k_coco")

    # Also check sibling directory
    parent = os.path.dirname(data_dir)
    for candidate in [
        os.path.join(parent, "cubicasa5k_coco"),
        os.path.join(data_dir, "cubicasa5k_coco"),
        coco_dir,
    ]:
        if os.path.exists(candidate):
            coco_dir = candidate
            break

    coco_json = os.path.join(coco_dir, f"{split}_coco_pt.json")
    if os.path.exists(coco_json):
        return _load_coco_format(coco_dir, split, max_samples)

    # Fallback: SVG parsing (slower)
    logger.info(f"COCO JSON not found, falling back to SVG parsing")
    return _load_svg_format(data_dir, split, max_samples)


def _load_svg_format(
    data_dir: str,
    split: str = "train",
    max_samples: int | None = None,
) -> list[dict]:
    """Fallback: load from SVG model files."""
    svg_paths = []

    split_file = os.path.join(data_dir, f"{split}.txt")
    if os.path.exists(split_file):
        with open(split_file, "r") as f:
            sample_paths = [line.strip() for line in f if line.strip()]
        for sp in sample_paths:
            svg = os.path.join(data_dir, sp, "model.svg")
            if not os.path.exists(svg):
                svg = os.path.join(data_dir, "cubicasa5k", sp, "model.svg")
            if os.path.exists(svg):
                svg_paths.append(svg)
    else:
        # Glob fallback
        for search in [os.path.join(data_dir, "cubicasa5k"), data_dir]:
            found = glob.glob(os.path.join(search, "**", "model.svg"), recursive=True)
            if found:
                svg_paths = sorted(set(found))
                break

        if svg_paths:
            svg_paths.sort()
            n = len(svg_paths)
            if split == "train":
                svg_paths = svg_paths[:int(n * 0.8)]
            elif split == "val":
                svg_paths = svg_paths[int(n * 0.8):int(n * 0.9)]
            elif split == "test":
                svg_paths = svg_paths[int(n * 0.9):]

    if max_samples:
        svg_paths = svg_paths[:max_samples]

    samples = []
    for svg_path in svg_paths:
        result = _parse_svg(svg_path)
        if result and len(result["layout_graph"]["rooms"]) >= 2:
            samples.append(result)

    logger.info(f"CubiCasa5K SVG: Loaded {len(samples)} samples from {split}")
    return samples


def _parse_svg(svg_path: str) -> dict | None:
    """Parse a single CubiCasa SVG file (simplified)."""
    try:
        tree = ET.parse(svg_path)
        root = tree.getroot()
    except ET.ParseError:
        return None

    rooms_data = []
    for elem in root.iter():
        tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
        if tag not in ("polygon", "polyline", "rect"):
            continue

        room_class = elem.get("class", "") or elem.get("id", "")
        if not room_class:
            continue

        skip = {"Wall", "wall", "Window", "window", "Door", "door", "Icon", "icon", "Stair", "stair"}
        if any(s in room_class for s in skip):
            continue

        points = []
        if tag in ("polygon", "polyline"):
            pts = elem.get("points", "")
            tokens = re.split(r'[\s,]+', pts.strip())
            for k in range(0, len(tokens) - 1, 2):
                try:
                    points.append((float(tokens[k]), float(tokens[k+1])))
                except ValueError:
                    continue
        elif tag == "rect":
            x, y = float(elem.get("x", 0)), float(elem.get("y", 0))
            w, h = float(elem.get("width", 0)), float(elem.get("height", 0))
            points = [(x, y), (x+w, y), (x+w, y+h), (x, y+h)]

        if len(points) < 3:
            continue

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        area = abs(sum(points[i][0]*points[(i+1)%len(points)][1] - points[(i+1)%len(points)][0]*points[i][1] for i in range(len(points)))) / 2
        if area < 1.0:
            continue

        rooms_data.append({"bbox": (min(xs), min(ys), max(xs), max(ys)), "area": area})

    if len(rooms_data) < 2:
        return None

    if len(rooms_data) > 30:
        rooms_data.sort(key=lambda r: r["area"], reverse=True)
        rooms_data = rooms_data[:30]

    # Normalise and build
    all_x = [r["bbox"][0] for r in rooms_data] + [r["bbox"][2] for r in rooms_data]
    all_y = [r["bbox"][1] for r in rooms_data] + [r["bbox"][3] for r in rooms_data]
    gx, gy, gX, gY = min(all_x), min(all_y), max(all_x), max(all_y)
    wr, hr = max(gX - gx, 1e-6), max(gY - gy, 1e-6)

    room_types = _assign_room_types(len(rooms_data))
    room_specs, room_layouts = [], []

    for i, room in enumerate(rooms_data):
        area_sqm = max(4.0, min(room["area"] / 10000.0, 200.0))
        spec = RoomSpec(room_id=f"room_{i+1}", room_type=room_types[i], target_area_sqm=round(area_sqm, 1))
        room_specs.append(spec)
        x0, y0, x1, y1 = room["bbox"]
        try:
            room_layouts.append(RoomLayout(room_spec=spec, bbox=BoundingBox(
                x_min=round(max(0, (x0-gx)/wr), 6), y_min=round(max(0, (y0-gy)/hr), 6),
                x_max=round(min(1, (x1-gx)/wr), 6), y_max=round(min(1, (y1-gy)/hr), 6),
            )))
        except Exception:
            continue

    if len(room_layouts) < 2:
        return None

    adj = _infer_adjacency(room_layouts)
    pa = max(30.0, min(sum(s.target_area_sqm for s in room_specs) * 1.2, 2000.0))
    facing = random.choice(list(CompassFacing))

    parsed = ParsedLayout(rooms=room_specs[:len(room_layouts)], plot_area_sqm=round(pa, 1),
                          facing=facing, adjacency_constraints=adj)
    lg = LayoutGraph(rooms=room_layouts, adjacency_edges=adj, plot_area_sqm=round(pa, 1),
                     facing=facing, generation_mode="heuristic")
    return {"parsed_layout": parsed.model_dump(), "layout_graph": lg.model_dump()}
