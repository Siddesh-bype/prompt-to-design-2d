"""
Dataset Generator — Creates synthetic floor plan layouts for GNN training.

Generates random room configurations with valid bounding boxes,
adjacency constraints, and Vastu-compliant placements.
"""

from __future__ import annotations

import json
import math
import os
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
from app.services.gnn_engine import build_node_features, build_edge_features


# ─── BHK Templates ──────────────────────────────────────────────────────────

BHK_TEMPLATES = {
    1: [
        (RoomType.LIVING_ROOM, 18, 25),
        (RoomType.BEDROOM, 10, 16),
        (RoomType.KITCHEN, 6, 10),
        (RoomType.BATHROOM, 4, 6),
    ],
    2: [
        (RoomType.LIVING_ROOM, 20, 30),
        (RoomType.MASTER_BEDROOM, 14, 22),
        (RoomType.BEDROOM, 10, 16),
        (RoomType.KITCHEN, 8, 12),
        (RoomType.BATHROOM, 4, 6),
        (RoomType.BATHROOM, 4, 6),
    ],
    3: [
        (RoomType.LIVING_ROOM, 22, 35),
        (RoomType.MASTER_BEDROOM, 16, 24),
        (RoomType.BEDROOM, 12, 18),
        (RoomType.BEDROOM, 10, 16),
        (RoomType.KITCHEN, 8, 14),
        (RoomType.BATHROOM, 4, 7),
        (RoomType.BATHROOM, 4, 6),
    ],
    4: [
        (RoomType.LIVING_ROOM, 25, 40),
        (RoomType.MASTER_BEDROOM, 18, 26),
        (RoomType.BEDROOM, 14, 20),
        (RoomType.BEDROOM, 12, 18),
        (RoomType.BEDROOM, 10, 16),
        (RoomType.KITCHEN, 10, 16),
        (RoomType.BATHROOM, 4, 7),
        (RoomType.BATHROOM, 4, 6),
        (RoomType.BATHROOM, 4, 6),
    ],
}

# Optional extra rooms
OPTIONAL_ROOMS = [
    (RoomType.BALCONY, 4, 8),
    (RoomType.STUDY, 6, 12),
    (RoomType.DINING, 8, 14),
    (RoomType.CORRIDOR, 4, 6),
    (RoomType.UTILITY, 4, 6),
]


# ─── Generator ───────────────────────────────────────────────────────────────


def generate_sample(bhk: int | None = None, seed: int | None = None) -> dict:
    """Generate a single synthetic floor plan sample.

    Args:
        bhk: Number of bedrooms (1–4). Random if None.
        seed: Random seed for reproducibility.

    Returns:
        Dict with keys: parsed_layout, layout_graph (both as dicts)
    """
    if seed is not None:
        random.seed(seed)

    if bhk is None:
        bhk = random.choice([1, 2, 2, 3, 3, 3, 4])  # Weighted towards 2–3 BHK

    template = BHK_TEMPLATES[bhk]
    facing = random.choice(list(CompassFacing))

    # Create rooms
    rooms = []
    for i, (room_type, area_min, area_max) in enumerate(template):
        area = round(random.uniform(area_min, area_max), 1)
        compass_pref = None
        if random.random() < 0.3:
            compass_pref = random.choice(list(CompassFacing))

        rooms.append(RoomSpec(
            room_id=f"room_{i+1}",
            room_type=room_type,
            target_area_sqm=area,
            compass_preference=compass_pref,
        ))

    # Random optional rooms
    if random.random() < 0.4:
        extra = random.choice(OPTIONAL_ROOMS)
        rt, amin, amax = extra
        rooms.append(RoomSpec(
            room_id=f"room_{len(rooms)+1}",
            room_type=rt,
            target_area_sqm=round(random.uniform(amin, amax), 1),
        ))

    # Calculate total area
    total_room_area = sum(r.target_area_sqm for r in rooms)
    plot_area = total_room_area * random.uniform(1.1, 1.4)  # 10–40% buffer

    # Generate adjacency constraints
    adjacency_constraints = _generate_adjacencies(rooms)

    # Build ParsedLayout
    parsed = ParsedLayout(
        rooms=rooms,
        plot_area_sqm=round(plot_area, 1),
        facing=facing,
        adjacency_constraints=adjacency_constraints,
        vastu_enabled=random.random() < 0.5,
    )

    # Generate ground-truth layout using strip-packing
    layout_graph = _generate_ground_truth(parsed)

    return {
        "parsed_layout": parsed.model_dump(),
        "layout_graph": layout_graph.model_dump(),
    }


def _generate_adjacencies(rooms: list[RoomSpec]) -> list[AdjacencyEdge]:
    """Generate realistic adjacency constraints between rooms."""
    edges = []
    room_map = {r.room_id: r for r in rooms}

    # Standard adjacencies
    master = next((r for r in rooms if r.room_type == RoomType.MASTER_BEDROOM), None)
    bathrooms = [r for r in rooms if r.room_type in (RoomType.BATHROOM, RoomType.TOILET)]
    living = next((r for r in rooms if r.room_type == RoomType.LIVING_ROOM), None)
    kitchen = next((r for r in rooms if r.room_type == RoomType.KITCHEN), None)
    dining = next((r for r in rooms if r.room_type == RoomType.DINING), None)

    # Master bedroom ↔ bathroom (DOOR)
    if master and bathrooms:
        edges.append(AdjacencyEdge(
            room_a_id=master.room_id,
            room_b_id=bathrooms[0].room_id,
            connection_type=ConnectionType.DOOR,
            required=True,
        ))

    # Living room ↔ kitchen (OPENING)
    if living and kitchen:
        edges.append(AdjacencyEdge(
            room_a_id=living.room_id,
            room_b_id=kitchen.room_id,
            connection_type=ConnectionType.OPENING,
            required=True,
        ))

    # Kitchen ↔ dining (OPENING)
    if kitchen and dining:
        edges.append(AdjacencyEdge(
            room_a_id=kitchen.room_id,
            room_b_id=dining.room_id,
            connection_type=ConnectionType.OPENING,
            required=True,
        ))

    return edges


def _generate_ground_truth(parsed: ParsedLayout) -> LayoutGraph:
    """Generate ground-truth bounding boxes using strip-packing.

    Works in absolute metre coords then normalises to [0,1] at the end.
    Falls back to a uniform grid if any room fails validation.
    """
    plot_side = math.sqrt(parsed.plot_area_sqm)
    rooms_raw: list[tuple] = []  # (room_spec, x_min_m, y_min_m, x_max_m, y_max_m)
    cursor_x, cursor_y, row_height = 0.0, 0.0, 0.0

    for room_spec in parsed.rooms:
        area = room_spec.target_area_sqm
        aspect = random.uniform(0.7, 1.8)
        h = math.sqrt(max(area / aspect, 0.01))
        w = max(area / h, 0.01)

        # Clamp dimensions — no jitter that could cause negatives
        w = max(w, 2.4)
        h = max(h, 2.4)

        if cursor_x + w > plot_side:
            cursor_x = 0.0
            cursor_y += row_height
            row_height = 0.0

        if cursor_y >= plot_side:
            cursor_x += w
            row_height = max(row_height, h)
            continue

        x0 = cursor_x
        y0 = cursor_y
        x1 = min(cursor_x + w, plot_side)
        y1 = min(cursor_y + h, plot_side)

        if x1 > x0 and y1 > y0:
            rooms_raw.append((room_spec, x0, y0, x1, y1))

        cursor_x += w
        row_height = max(row_height, h)

    # Fallback grid
    if len(rooms_raw) < 2:
        rooms_raw = []
        n = len(parsed.rooms)
        cols = max(2, math.ceil(math.sqrt(n)))
        rows_count = math.ceil(n / cols)
        cell_w = plot_side / cols
        cell_h = plot_side / rows_count
        for i, room_spec in enumerate(parsed.rooms):
            r = i // cols
            c = i % cols
            rooms_raw.append((room_spec, c * cell_w, r * cell_h,
                              (c + 1) * cell_w, (r + 1) * cell_h))

    # Normalise to [0, 1]
    rooms = []
    s = max(plot_side, 1e-6)
    for room_spec, x0, y0, x1, y1 in rooms_raw:
        nx0 = max(0.0, min(1.0, x0 / s))
        ny0 = max(0.0, min(1.0, y0 / s))
        nx1 = max(0.0, min(1.0, x1 / s))
        ny1 = max(0.0, min(1.0, y1 / s))
        if nx1 <= nx0 or ny1 <= ny0:
            continue
        try:
            rooms.append(RoomLayout(
                room_spec=room_spec,
                bbox=BoundingBox(
                    x_min=round(nx0, 6),
                    y_min=round(ny0, 6),
                    x_max=round(nx1, 6),
                    y_max=round(ny1, 6),
                ),
            ))
        except Exception:
            pass

    return LayoutGraph(
        rooms=rooms,
        adjacency_edges=parsed.adjacency_constraints,
        plot_area_sqm=parsed.plot_area_sqm,
        facing=parsed.facing,
        generation_mode="heuristic",
    )


# ─── Dataset Generation ─────────────────────────────────────────────────────


def generate_dataset(
    n_samples: int = 1000,
    output_dir: str = "data/training",
    seed: int = 42,
) -> str:
    """Generate a full training dataset and save as JSONL.

    Args:
        n_samples: Number of samples to generate
        output_dir: Output directory path
        seed: Random seed

    Returns:
        Path to the generated dataset file
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "floor_plans.jsonl")

    random.seed(seed)

    with open(output_path, "w") as f:
        for i in range(n_samples):
            sample = generate_sample(seed=seed + i)
            f.write(json.dumps(sample) + "\n")

    print(f"Generated {n_samples} samples → {output_path}")
    return output_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate training dataset")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--output", type=str, default="data/training")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_dataset(args.samples, args.output, args.seed)
