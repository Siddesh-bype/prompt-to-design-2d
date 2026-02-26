"""
Constraint Solver — Z3 hard constraints + Shapely/SciPy soft optimisation + heuristic fallback.

Takes a LayoutGraph (which may have overlapping or invalid room placements)
and returns a physically valid LayoutGraph.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

from app.models.schemas import (
    BoundingBox,
    CompassFacing,
    LayoutGraph,
    RoomLayout,
    RoomType,
)

logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import z3
    HAS_Z3 = True
except ImportError:
    HAS_Z3 = False
    logger.warning("z3-solver not installed — Z3 constraint solving disabled")

try:
    from shapely.geometry import box as shapely_box
    from shapely.geometry import Polygon
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False
    logger.warning("shapely not installed — geometric optimisation limited")

try:
    from scipy.optimize import minimize as scipy_minimize
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    logger.warning("scipy not installed — soft optimisation disabled")


# ─── STAGE 1: Z3 Hard Constraint Solver ─────────────────────────────────────


def solve_with_z3(
    layout: LayoutGraph,
    plot_width: float = 10.0,
    plot_height: float = 10.0,
    timeout_ms: int = 4000,
) -> tuple[LayoutGraph, bool]:
    """Solve layout constraints using Z3 SMT solver.

    Encodes hard constraints:
    1. All coords in [0, plot_width/height]
    2. Minimum room dimension >= 2.4m (BIS SP 7:2016)
    3. Room area within 20% of target
    4. No pairwise overlaps
    5. Required adjacencies share wall >= 0.9m
    6. All rooms within plot boundary

    Returns:
        (solved_layout, success_bool)
    """
    if not HAS_Z3:
        logger.warning("Z3 not available — skipping constraint solving")
        return layout, False

    n_rooms = len(layout.rooms)
    if n_rooms == 0:
        return layout, True

    solver = z3.Solver()
    z3.set_option("timeout", timeout_ms)

    # Create Z3 Real variables for each room's bbox
    vars_dict = {}
    for i, room in enumerate(layout.rooms):
        rid = room.room_spec.room_id
        vars_dict[rid] = {
            "x_min": z3.Real(f"{rid}_x_min"),
            "y_min": z3.Real(f"{rid}_y_min"),
            "x_max": z3.Real(f"{rid}_x_max"),
            "y_max": z3.Real(f"{rid}_y_max"),
        }

    # Constraint 1 & 6: All coords in [0, plot_width/height]
    for rid, v in vars_dict.items():
        solver.add(v["x_min"] >= 0)
        solver.add(v["y_min"] >= 0)
        solver.add(v["x_max"] <= plot_width)
        solver.add(v["y_max"] <= plot_height)
        solver.add(v["x_min"] < v["x_max"])
        solver.add(v["y_min"] < v["y_max"])

    # Constraint 2: Minimum room dimension >= 2.4m
    min_dim = 2.4
    for rid, v in vars_dict.items():
        solver.add(v["x_max"] - v["x_min"] >= min_dim)
        solver.add(v["y_max"] - v["y_min"] >= min_dim)

    # Constraint 3: Room area within 20% of target
    for room in layout.rooms:
        rid = room.room_spec.room_id
        v = vars_dict[rid]
        target = room.room_spec.target_area_sqm
        area_expr = (v["x_max"] - v["x_min"]) * (v["y_max"] - v["y_min"])
        solver.add(area_expr >= target * 0.8)
        solver.add(area_expr <= target * 1.2)

    # Constraint 4: No pairwise overlaps
    room_ids = [r.room_spec.room_id for r in layout.rooms]
    for i in range(n_rooms):
        for j in range(i + 1, n_rooms):
            v_i = vars_dict[room_ids[i]]
            v_j = vars_dict[room_ids[j]]
            solver.add(z3.Or(
                v_i["x_max"] <= v_j["x_min"],  # i is left of j
                v_j["x_max"] <= v_i["x_min"],  # j is left of i
                v_i["y_max"] <= v_j["y_min"],  # i is above j
                v_j["y_max"] <= v_i["y_min"],  # j is above i
            ))

    # Constraint 5: Required adjacencies share wall >= 0.9m
    shared_wall_min = 0.9
    for edge in layout.adjacency_edges:
        if not edge.required:
            continue
        if edge.room_a_id not in vars_dict or edge.room_b_id not in vars_dict:
            continue

        v_a = vars_dict[edge.room_a_id]
        v_b = vars_dict[edge.room_b_id]

        # Rooms share a wall if they touch on one axis with overlap >= 0.9m on the other
        # Horizontal adjacency: a.x_max == b.x_min (or vice versa) AND vertical overlap >= 0.9
        # Vertical adjacency: a.y_max == b.y_min (or vice versa) AND horizontal overlap >= 0.9

        h_overlap = z3.If(
            z3.And(v_a["y_min"] < v_b["y_max"], v_b["y_min"] < v_a["y_max"]),
            z3.If(v_a["y_max"] < v_b["y_max"], v_a["y_max"], v_b["y_max"]) -
            z3.If(v_a["y_min"] > v_b["y_min"], v_a["y_min"], v_b["y_min"]),
            0,
        )
        v_overlap = z3.If(
            z3.And(v_a["x_min"] < v_b["x_max"], v_b["x_min"] < v_a["x_max"]),
            z3.If(v_a["x_max"] < v_b["x_max"], v_a["x_max"], v_b["x_max"]) -
            z3.If(v_a["x_min"] > v_b["x_min"], v_a["x_min"], v_b["x_min"]),
            0,
        )

        solver.add(z3.Or(
            z3.And(v_a["x_max"] == v_b["x_min"], h_overlap >= shared_wall_min),
            z3.And(v_b["x_max"] == v_a["x_min"], h_overlap >= shared_wall_min),
            z3.And(v_a["y_max"] == v_b["y_min"], v_overlap >= shared_wall_min),
            z3.And(v_b["y_max"] == v_a["y_min"], v_overlap >= shared_wall_min),
        ))

    # Solve
    result = solver.check()

    if result == z3.sat:
        model = solver.model()
        new_rooms = []

        for room in layout.rooms:
            v = vars_dict[room.room_spec.room_id]
            x_min = _z3_to_float(model.evaluate(v["x_min"]))
            y_min = _z3_to_float(model.evaluate(v["y_min"]))
            x_max = _z3_to_float(model.evaluate(v["x_max"]))
            y_max = _z3_to_float(model.evaluate(v["y_max"]))

            # Normalise to [0, 1]
            bbox = BoundingBox(
                x_min=x_min / plot_width,
                y_min=y_min / plot_height,
                x_max=x_max / plot_width,
                y_max=y_max / plot_height,
            )
            new_rooms.append(RoomLayout(
                room_spec=room.room_spec,
                bbox=bbox,
                door_midpoints=room.door_midpoints,
            ))

        solved = layout.model_copy(update={"rooms": new_rooms})
        logger.info("Z3 solver: SAT — constraints satisfied")
        return solved, True
    else:
        status = "UNSAT" if result == z3.unsat else "TIMEOUT/UNKNOWN"
        logger.warning(f"Z3 solver: {status} — falling back")
        return layout, False


def _z3_to_float(val) -> float:
    """Convert a Z3 value to a Python float."""
    if hasattr(val, "as_fraction"):
        frac = val.as_fraction()
        return float(frac.numerator) / float(frac.denominator)
    return float(str(val))


# ─── STAGE 2: Soft Optimisation ──────────────────────────────────────────────


def optimise_layout(
    layout: LayoutGraph,
    plot_width: float = 10.0,
    plot_height: float = 10.0,
) -> LayoutGraph:
    """Optimise room positions using SciPy L-BFGS-B.

    Objectives:
    - Living room/master bedroom proximity to east/south edges (natural light)
    - Minimise corridor area (target < 8% of gross floor area)
    - Kitchen-to-bedroom centroid distance >= 3.0m
    - Overlap penalty
    """
    if not HAS_SCIPY or not HAS_SHAPELY:
        logger.warning("SciPy/Shapely not available — skipping optimisation")
        return layout

    n = len(layout.rooms)
    if n == 0:
        return layout

    # Pack current positions into flat vector [x_min, y_min, x_max, y_max] * n
    x0 = []
    for room in layout.rooms:
        b = room.bbox
        x0.extend([
            b.x_min * plot_width, b.y_min * plot_height,
            b.x_max * plot_width, b.y_max * plot_height,
        ])

    bounds = []
    for _ in range(n):
        bounds.extend([
            (0, plot_width), (0, plot_height),
            (0, plot_width), (0, plot_height),
        ])

    def objective(x):
        cost = 0.0
        rooms_data = []
        for i in range(n):
            x_min, y_min = x[4*i], x[4*i+1]
            x_max, y_max = x[4*i+2], x[4*i+3]
            cx = (x_min + x_max) / 2
            cy = (y_min + y_max) / 2
            rooms_data.append({
                "type": layout.rooms[i].room_spec.room_type,
                "x_min": x_min, "y_min": y_min,
                "x_max": x_max, "y_max": y_max,
                "cx": cx, "cy": cy,
                "area": max((x_max - x_min) * (y_max - y_min), 0.01),
            })

        # Natural light: living room and master bedroom near south/east
        for rd in rooms_data:
            if rd["type"] in (RoomType.LIVING_ROOM, RoomType.MASTER_BEDROOM):
                dist_to_east = abs(rd["x_max"] - plot_width)
                dist_to_south = abs(rd["y_max"] - plot_height)
                cost += 0.5 * (dist_to_east + dist_to_south)

        # Corridor area penalty: target < 8%
        gross_area = plot_width * plot_height
        for rd in rooms_data:
            if rd["type"] == RoomType.CORRIDOR:
                ratio = rd["area"] / gross_area
                if ratio > 0.08:
                    cost += 10.0 * (ratio - 0.08)

        # Kitchen-to-bedroom distance >= 3.0m
        kitchen_centroids = [(rd["cx"], rd["cy"]) for rd in rooms_data
                            if rd["type"] == RoomType.KITCHEN]
        bedroom_centroids = [(rd["cx"], rd["cy"]) for rd in rooms_data
                            if rd["type"] in (RoomType.BEDROOM, RoomType.MASTER_BEDROOM)]

        for kcx, kcy in kitchen_centroids:
            for bcx, bcy in bedroom_centroids:
                dist = math.sqrt((kcx - bcx)**2 + (kcy - bcy)**2)
                if dist < 3.0:
                    cost += 5.0 * (3.0 - dist)

        # Overlap penalty
        for i in range(n):
            for j in range(i + 1, n):
                ri, rj = rooms_data[i], rooms_data[j]
                ox = max(0, min(ri["x_max"], rj["x_max"]) - max(ri["x_min"], rj["x_min"]))
                oy = max(0, min(ri["y_max"], rj["y_max"]) - max(ri["y_min"], rj["y_min"]))
                cost += 20.0 * ox * oy

        return cost

    try:
        result = scipy_minimize(
            objective, x0, method="L-BFGS-B", bounds=bounds,
            options={"maxiter": 100, "maxfun": 500},
        )
        final = result.x
    except Exception as e:
        logger.warning(f"SciPy optimisation failed: {e}")
        return layout

    # Rebuild layout
    new_rooms = []
    for i, room in enumerate(layout.rooms):
        x_min = max(0, final[4*i]) / plot_width
        y_min = max(0, final[4*i+1]) / plot_height
        x_max = min(plot_width, final[4*i+2]) / plot_width
        y_max = min(plot_height, final[4*i+3]) / plot_height

        # Ensure valid bbox
        if x_min >= x_max:
            x_max = min(x_min + 0.1, 1.0)
        if y_min >= y_max:
            y_max = min(y_min + 0.1, 1.0)

        bbox = BoundingBox(
            x_min=max(0.0, min(x_min, 1.0)),
            y_min=max(0.0, min(y_min, 1.0)),
            x_max=max(0.0, min(x_max, 1.0)),
            y_max=max(0.0, min(y_max, 1.0)),
        )
        new_rooms.append(RoomLayout(
            room_spec=room.room_spec,
            bbox=bbox,
            door_midpoints=room.door_midpoints,
        ))

    return layout.model_copy(update={"rooms": new_rooms})


# ─── STAGE 3: Heuristic Fallback Placer ─────────────────────────────────────

# Priority order for strip-packing
ROOM_PRIORITY = [
    RoomType.LIVING_ROOM,
    RoomType.MASTER_BEDROOM,
    RoomType.KITCHEN,
    RoomType.BATHROOM,
    RoomType.TOILET,
    RoomType.BEDROOM,
    RoomType.STUDY,
    RoomType.DINING,
    RoomType.CORRIDOR,
    RoomType.BALCONY,
    RoomType.UTILITY,
    RoomType.GARAGE,
]


def heuristic_place(
    layout: LayoutGraph,
    plot_width: float = 10.0,
    plot_height: float = 10.0,
) -> LayoutGraph:
    """Place rooms using a squarified treemap algorithm.

    Recursively subdivides the plot rectangle so rooms fill the
    entire area with zero gaps. Each room's slice is proportional
    to its target_area_sqm.
    """
    n = len(layout.rooms)
    if n == 0:
        return layout

    # Sort rooms by priority (largest/most important first)
    def priority_key(room: RoomLayout) -> tuple[int, float]:
        try:
            pri = ROOM_PRIORITY.index(room.room_spec.room_type)
        except ValueError:
            pri = len(ROOM_PRIORITY)
        return (pri, -room.room_spec.target_area_sqm)

    sorted_rooms = sorted(layout.rooms, key=priority_key)

    # Compute area weights (normalised so they sum to 1.0)
    total_area = sum(max(r.room_spec.target_area_sqm, 1.0) for r in sorted_rooms)
    weights = [max(r.room_spec.target_area_sqm, 1.0) / total_area for r in sorted_rooms]

    # Recursively subdivide the rectangle
    bboxes = _treemap_subdivide(weights, 0.0, 0.0, 1.0, 1.0)

    placed = []
    for i, room in enumerate(sorted_rooms):
        x_min, y_min, x_max, y_max = bboxes[i]

        # Clamp to [0, 1]
        x_min = max(0.0, min(x_min, 1.0))
        y_min = max(0.0, min(y_min, 1.0))
        x_max = max(0.0, min(x_max, 1.0))
        y_max = max(0.0, min(y_max, 1.0))

        if x_min >= x_max:
            x_max = min(x_min + 0.05, 1.0)
        if y_min >= y_max:
            y_max = min(y_min + 0.05, 1.0)

        bbox = BoundingBox(
            x_min=x_min, y_min=y_min,
            x_max=x_max, y_max=y_max,
        )
        placed.append(RoomLayout(
            room_spec=room.room_spec,
            bbox=bbox,
            door_midpoints=room.door_midpoints,
        ))

    result = layout.model_copy(update={
        "rooms": placed,
        "generation_mode": "heuristic",
    })
    logger.info(f"Heuristic placer: placed {len(placed)} rooms via treemap")
    return result


def _treemap_subdivide(
    weights: list[float],
    x0: float, y0: float, x1: float, y1: float,
) -> list[tuple[float, float, float, float]]:
    """Recursively subdivide a rectangle into slices proportional to weights.

    Uses alternating horizontal/vertical cuts based on rectangle aspect ratio.
    Returns a list of (x_min, y_min, x_max, y_max) for each weight.
    """
    n = len(weights)
    if n == 0:
        return []
    if n == 1:
        return [(x0, y0, x1, y1)]

    w = x1 - x0
    h = y1 - y0

    total = sum(weights)
    if total <= 0:
        total = 1.0

    # Split into two groups aiming for ~50% area each
    cumsum = 0.0
    split_idx = 0
    half = total / 2.0
    for i, wt in enumerate(weights):
        cumsum += wt
        if cumsum >= half:
            split_idx = i + 1
            break

    # Ensure at least one item in each group
    split_idx = max(1, min(split_idx, n - 1))

    left_weights = weights[:split_idx]
    right_weights = weights[split_idx:]
    left_frac = sum(left_weights) / total

    # Choose split direction based on rectangle aspect
    if w >= h:
        # Vertical split (left | right)
        mid_x = x0 + left_frac * w
        left_bboxes = _treemap_subdivide(left_weights, x0, y0, mid_x, y1)
        right_bboxes = _treemap_subdivide(right_weights, mid_x, y0, x1, y1)
    else:
        # Horizontal split (top / bottom)
        mid_y = y0 + left_frac * h
        left_bboxes = _treemap_subdivide(left_weights, x0, y0, x1, mid_y)
        right_bboxes = _treemap_subdivide(right_weights, x0, mid_y, x1, y1)

    return left_bboxes + right_bboxes


# ─── Main Entrypoint ────────────────────────────────────────────────────────


def run_constraint_pipeline(
    layout: LayoutGraph,
    plot_width: float = 10.0,
    plot_height: float = 10.0,
) -> LayoutGraph:
    """Run the full constraint-solving pipeline.

    Pipeline:
    1. Try Z3 hard constraints → if SAT, soft optimise → check quality
    2. Always fall back to treemap (gap-free) if Z3 produces poor results
    3. Calculate overlap_rate and adjacency_satisfaction

    Args:
        layout: Input LayoutGraph (may have overlaps)
        plot_width: Plot width in metres
        plot_height: Plot height in metres

    Returns:
        Physically valid LayoutGraph with rooms filling the full plot
    """
    result = None

    # Stage 1: Try Z3
    solved, z3_success = solve_with_z3(layout, plot_width, plot_height)

    if z3_success:
        logger.info("Pipeline path: Z3 → Optimise")
        optimised = optimise_layout(solved, plot_width, plot_height)
        overlap = _compute_overlap(optimised, plot_width, plot_height)

        if overlap < 0.05:
            result = optimised
            logger.info(f"Z3 + Optimise accepted (overlap={overlap:.3f})")
        else:
            logger.warning(f"Z3 result rejected (overlap={overlap:.3f}) → treemap")

    # Stage 2: Treemap fallback (gap-free, always works)
    if result is None:
        logger.info("Pipeline path: Treemap (gap-free)")
        result = heuristic_place(layout, plot_width, plot_height)
        # NOTE: Do NOT run optimise_layout after treemap —
        # it would move rooms apart and create gaps.

    # Calculate metrics
    result = result.model_copy(update={
        "overlap_rate": _compute_overlap(result, plot_width, plot_height),
        "adjacency_satisfaction": _compute_adjacency(result, plot_width, plot_height),
    })

    logger.info(
        f"Constraint pipeline complete: "
        f"mode={result.generation_mode}, "
        f"overlap={result.overlap_rate:.3f}, "
        f"adj_sat={result.adjacency_satisfaction:.3f}"
    )

    return result


def _compute_overlap(layout: LayoutGraph, pw: float, ph: float) -> float:
    """Compute total overlap ratio using simple rectangle intersection."""
    total_overlap = 0.0
    total_area = 0.0
    rooms = layout.rooms

    for room in rooms:
        w = (room.bbox.x_max - room.bbox.x_min) * pw
        h = (room.bbox.y_max - room.bbox.y_min) * ph
        total_area += w * h

    if total_area == 0:
        return 0.0

    for i in range(len(rooms)):
        for j in range(i + 1, len(rooms)):
            ri, rj = rooms[i].bbox, rooms[j].bbox
            ox = max(0, min(ri.x_max, rj.x_max) - max(ri.x_min, rj.x_min)) * pw
            oy = max(0, min(ri.y_max, rj.y_max) - max(ri.y_min, rj.y_min)) * ph
            total_overlap += ox * oy

    return min(total_overlap / total_area, 1.0)


def _compute_adjacency(layout: LayoutGraph, pw: float, ph: float) -> float:
    """Compute fraction of required adjacency edges that are satisfied."""
    required_edges = [e for e in layout.adjacency_edges if e.required]
    if not required_edges:
        return 1.0

    room_map = {r.room_spec.room_id: r.bbox for r in layout.rooms}
    satisfied = 0

    for edge in required_edges:
        if edge.room_a_id not in room_map or edge.room_b_id not in room_map:
            continue

        ba = room_map[edge.room_a_id]
        bb = room_map[edge.room_b_id]

        # Check if rooms share a wall of length >= 0.9m
        # Horizontal adjacency
        shared_wall = 0.0

        # Check if touching on x-axis
        if abs(ba.x_max - bb.x_min) < 0.01 or abs(bb.x_max - ba.x_min) < 0.01:
            y_overlap = max(0, min(ba.y_max, bb.y_max) - max(ba.y_min, bb.y_min)) * ph
            shared_wall = max(shared_wall, y_overlap)

        # Check if touching on y-axis
        if abs(ba.y_max - bb.y_min) < 0.01 or abs(bb.y_max - ba.y_min) < 0.01:
            x_overlap = max(0, min(ba.x_max, bb.x_max) - max(ba.x_min, bb.x_min)) * pw
            shared_wall = max(shared_wall, x_overlap)

        if shared_wall >= 0.9:
            satisfied += 1

    return satisfied / len(required_edges)
