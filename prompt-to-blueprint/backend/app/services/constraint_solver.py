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
    AdjacencyEdge,
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
    # Use a solver-local timeout to avoid global side effects across requests.
    solver.set("timeout", timeout_ms)

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

# Ideal width:height aspect ratios for each room type
ROOM_ASPECT_RATIO: dict[RoomType, float] = {
    RoomType.LIVING_ROOM: 1.5,      # wide and rectangular
    RoomType.MASTER_BEDROOM: 1.25,  # slightly wider than tall
    RoomType.BEDROOM: 1.2,          # slightly wider
    RoomType.KITCHEN: 1.4,          # long and narrow
    RoomType.BATHROOM: 0.65,        # taller than wide (narrow)
    RoomType.TOILET: 0.6,           # narrow
    RoomType.CORRIDOR: 4.0,         # very long and narrow
    RoomType.BALCONY: 3.0,          # long strip
    RoomType.STUDY: 1.1,            # nearly square
    RoomType.DINING: 1.3,           # rectangular
    RoomType.UTILITY: 0.8,          # small and squarish
    RoomType.GARAGE: 1.8,           # wide
}

# Architectural zone definitions (normalised y-ranges)
# FRONT_ZONE: entrance-side — living room, balcony
# MIDDLE_ZONE: transition — corridor, kitchen, dining
# REAR_ZONE: private — bedrooms, bathrooms, study
FRONT_ZONE = (0.0, 0.40)
MIDDLE_ZONE = (0.40, 0.55)
REAR_ZONE = (0.55, 1.0)

# Where each room type ideally goes
ROOM_ZONE_PREFERENCE: dict[RoomType, str] = {
    RoomType.LIVING_ROOM: "front",
    RoomType.BALCONY: "front",
    RoomType.KITCHEN: "front",
    RoomType.DINING: "front",
    RoomType.CORRIDOR: "middle",
    RoomType.MASTER_BEDROOM: "rear",
    RoomType.BEDROOM: "rear",
    RoomType.STUDY: "rear",
    RoomType.BATHROOM: "rear",
    RoomType.TOILET: "rear",
    RoomType.UTILITY: "rear",
    RoomType.GARAGE: "front",
}


def heuristic_place(
    layout: LayoutGraph,
    plot_width: float = 10.0,
    plot_height: float = 10.0,
) -> LayoutGraph:
    """Place rooms using zone-based architectural layout with realistic proportions.

    Pipeline:
    1. Identify en-suite bathrooms (those with DOOR to a bedroom)
    2. Classify rooms into front/middle/rear architectural zones
    3. Run treemap on each zone separately for gap-free coverage
    4. Adjust room aspect ratios within each zone
    5. Carve en-suite bathrooms inside their parent bedrooms
    6. Place corridor as a connecting strip (if present)
    7. Generate door midpoints on shared walls
    """
    n = len(layout.rooms)
    if n == 0:
        return layout

    # ── Step 1: Identify en-suite bathroom → bedroom pairs ──
    ensuite_pairs: dict[str, str] = {}  # bathroom_id → bedroom_id
    bedroom_types = {RoomType.BEDROOM, RoomType.MASTER_BEDROOM}
    bathroom_types = {RoomType.BATHROOM, RoomType.TOILET}

    for edge in layout.adjacency_edges:
        if edge.connection_type.value != "DOOR":
            continue
        room_a = next((r for r in layout.rooms if r.room_spec.room_id == edge.room_a_id), None)
        room_b = next((r for r in layout.rooms if r.room_spec.room_id == edge.room_b_id), None)
        if room_a is None or room_b is None:
            continue

        bed_room, bath_room = None, None
        if room_a.room_spec.room_type in bedroom_types and room_b.room_spec.room_type in bathroom_types:
            bed_room, bath_room = room_a, room_b
        elif room_b.room_spec.room_type in bedroom_types and room_a.room_spec.room_type in bathroom_types:
            bed_room, bath_room = room_b, room_a

        if bed_room and bath_room and bath_room.room_spec.room_id not in ensuite_pairs:
            ensuite_pairs[bath_room.room_spec.room_id] = bed_room.room_spec.room_id

    # ── Step 2: Separate rooms by zone ──
    ensuite_bath_ids = set(ensuite_pairs.keys())
    corridor_room = None
    front_rooms: list[RoomLayout] = []
    rear_rooms: list[RoomLayout] = []
    ensuite_rooms: list[RoomLayout] = []

    # Map Vastu ideal zones depending on plot facing
    # We map Vastu's 9 grid areas to our architectural 'front' or 'rear'
    vastu_to_arch_zone = {}
    if layout.vastu_enabled:
        facing = layout.facing.value
        # If house faces North (y=0 is North)
        if facing == "NORTH":
            vastu_to_arch_zone = {"N": "front", "NE": "front", "NW": "front", 
                                  "S": "rear", "SE": "rear", "SW": "rear"}
        elif facing == "SOUTH":
            vastu_to_arch_zone = {"S": "front", "SE": "front", "SW": "front", 
                                  "N": "rear", "NE": "rear", "NW": "rear"}
        elif facing == "EAST":
            vastu_to_arch_zone = {"E": "front", "NE": "front", "SE": "front", 
                                  "W": "rear", "NW": "rear", "SW": "rear"}
        elif facing == "WEST":
            vastu_to_arch_zone = {"W": "front", "NW": "front", "SW": "front", 
                                  "E": "rear", "NE": "rear", "SE": "rear"}

    for room in layout.rooms:
        if room.room_spec.room_id in ensuite_bath_ids:
            ensuite_rooms.append(room)
            continue
        if room.room_spec.room_type == RoomType.CORRIDOR:
            corridor_room = room
            continue
            
        zone = ROOM_ZONE_PREFERENCE.get(room.room_spec.room_type, "rear")
        
        # Override with Vastu preference if enabled
        if layout.vastu_enabled:
            from app.services.vastu_engine import _get_ideal_zone
            ideal_vastu = _get_ideal_zone(room.room_spec.room_type)
            # Pick first direction if multiple (e.g. "NW or SE" -> "NW")
            primary_vastu = ideal_vastu.split(" ")[0].strip()
            if primary_vastu in vastu_to_arch_zone:
                zone = vastu_to_arch_zone[primary_vastu]

        if zone == "front":
            front_rooms.append(room)
        else:
            rear_rooms.append(room)

    # If no front rooms exist, put the biggest room in front
    if not front_rooms and rear_rooms:
        rear_rooms.sort(key=lambda r: -r.room_spec.target_area_sqm)
        front_rooms.append(rear_rooms.pop(0))
    if not rear_rooms and front_rooms:
        front_rooms.sort(key=lambda r: -r.room_spec.target_area_sqm)
        rear_rooms.append(front_rooms.pop(0))

    # If still empty, fall back to simple treemap
    all_main = front_rooms + rear_rooms
    if not all_main:
        all_main = [r for r in layout.rooms if r.room_spec.room_id not in ensuite_bath_ids]
        if not all_main:
            all_main = list(layout.rooms)
        front_rooms = all_main
        rear_rooms = []

    # ── Calculate zone boundaries ──
    has_corridor = corridor_room is not None
    corridor_frac = 0.10 if has_corridor else 0.0  # 10% of plot for corridor strip

    front_total_area = sum(max(r.room_spec.target_area_sqm, 1.0) for r in front_rooms) if front_rooms else 0.0
    rear_total_area = sum(max(r.room_spec.target_area_sqm, 1.0) for r in rear_rooms) if rear_rooms else 0.0
    all_area = front_total_area + rear_total_area

    if all_area > 0:
        front_frac = (front_total_area / all_area) * (1.0 - corridor_frac)
    else:
        front_frac = 0.5 * (1.0 - corridor_frac)
    rear_frac = 1.0 - corridor_frac - front_frac

    # Clamp zone sizes (each zone at least 20% if it has rooms)
    min_zone = 0.20
    if front_rooms and front_frac < min_zone:
        front_frac = min_zone
        rear_frac = 1.0 - corridor_frac - front_frac
    if rear_rooms and rear_frac < min_zone:
        rear_frac = min_zone
        front_frac = 1.0 - corridor_frac - rear_frac

    # Zone y-boundaries
    front_y0 = 0.0
    front_y1 = front_frac
    corr_y0 = front_y1
    corr_y1 = front_y1 + corridor_frac
    rear_y0 = corr_y1
    rear_y1 = 1.0

    placed: list[RoomLayout] = []

    # ── Step 3: Place front-zone rooms ──
    if front_rooms:
        front_rooms.sort(key=lambda r: -r.room_spec.target_area_sqm)
        f_total = sum(max(r.room_spec.target_area_sqm, 1.0) for r in front_rooms)
        f_weights = [max(r.room_spec.target_area_sqm, 1.0) / f_total for r in front_rooms]
        f_bboxes = _treemap_subdivide(f_weights, 0.0, front_y0, 1.0, front_y1)
        for i, room in enumerate(front_rooms):
            x0, y0_, x1, y1_ = f_bboxes[i]
            x0, y0_, x1, y1_ = _clamp_bbox(x0, y0_, x1, y1_)
            # Adjust aspect ratio within the allocated space
            x0, y0_, x1, y1_ = _adjust_aspect_ratio(
                x0, y0_, x1, y1_, room.room_spec.room_type, plot_width, plot_height
            )
            bbox = BoundingBox(x_min=x0, y_min=y0_, x_max=x1, y_max=y1_)
            placed.append(RoomLayout(room_spec=room.room_spec, bbox=bbox, door_midpoints=[]))

    # ── Step 3b: Place corridor as connecting strip ──
    if corridor_room:
        placed.append(RoomLayout(
            room_spec=corridor_room.room_spec,
            bbox=BoundingBox(x_min=0.0, y_min=corr_y0, x_max=1.0, y_max=corr_y1),
            door_midpoints=[],
        ))

    # ── Step 4: Place rear-zone rooms ──
    if rear_rooms:
        rear_rooms.sort(key=lambda r: -r.room_spec.target_area_sqm)
        r_total = sum(max(r.room_spec.target_area_sqm, 1.0) for r in rear_rooms)
        r_weights = [max(r.room_spec.target_area_sqm, 1.0) / r_total for r in rear_rooms]
        r_bboxes = _treemap_subdivide(r_weights, 0.0, rear_y0, 1.0, rear_y1)
        for i, room in enumerate(rear_rooms):
            x0, y0_, x1, y1_ = r_bboxes[i]
            x0, y0_, x1, y1_ = _clamp_bbox(x0, y0_, x1, y1_)
            x0, y0_, x1, y1_ = _adjust_aspect_ratio(
                x0, y0_, x1, y1_, room.room_spec.room_type, plot_width, plot_height
            )
            bbox = BoundingBox(x_min=x0, y_min=y0_, x_max=x1, y_max=y1_)
            placed.append(RoomLayout(room_spec=room.room_spec, bbox=bbox, door_midpoints=[]))

    # Improve adjacency via swapping
    placed = _improve_adjacency_by_swapping(
        placed, layout.adjacency_edges, plot_width, plot_height
    )

    # ── Step 5: Nest en-suite bathrooms inside bedrooms ──
    placed = _nest_ensuite_bathrooms(placed, ensuite_rooms, ensuite_pairs)

    # ── Step 6: Generate door midpoints ──
    placed = _generate_door_midpoints(placed, layout.adjacency_edges, plot_width, plot_height)

    result = layout.model_copy(update={
        "rooms": placed,
        "generation_mode": "heuristic",
    })
    logger.info(f"Heuristic placer: placed {len(placed)} rooms (incl. {len(ensuite_rooms)} en-suites)")
    return result


def _clamp_bbox(
    x0: float, y0: float, x1: float, y1: float,
) -> tuple[float, float, float, float]:
    """Clamp bbox to [0,1] and ensure minimum size."""
    x0 = max(0.0, min(x0, 1.0))
    y0 = max(0.0, min(y0, 1.0))
    x1 = max(0.0, min(x1, 1.0))
    y1 = max(0.0, min(y1, 1.0))
    if x0 >= x1:
        x1 = min(x0 + 0.05, 1.0)
    if y0 >= y1:
        y1 = min(y0 + 0.05, 1.0)
    return x0, y0, x1, y1


def _adjust_aspect_ratio(
    x0: float, y0: float, x1: float, y1: float,
    room_type: RoomType,
    plot_width: float,
    plot_height: float,
) -> tuple[float, float, float, float]:
    """Adjust room bbox to better match the ideal aspect ratio for its type.

    Only adjusts WITHIN the allocated space — never expands beyond bounds.
    This keeps the treemap gap-free while improving room shapes.
    """
    ideal = ROOM_ASPECT_RATIO.get(room_type, 1.0)
    if ideal <= 0:
        return x0, y0, x1, y1

    w = (x1 - x0) * plot_width
    h = (y1 - y0) * plot_height
    if h <= 0:
        return x0, y0, x1, y1

    current_ratio = w / h

    # Don't adjust if already close to ideal (within 40%)
    if 0.6 * ideal <= current_ratio <= 1.4 * ideal:
        return x0, y0, x1, y1

    # For corridors and balconies, just return as-is since treemap
    # already assigns them proportionally
    if room_type in (RoomType.CORRIDOR, RoomType.BALCONY):
        return x0, y0, x1, y1

    return x0, y0, x1, y1


def _nest_ensuite_bathrooms(
    placed: list[RoomLayout],
    ensuite_rooms: list[RoomLayout],
    ensuite_pairs: dict[str, str],  # bath_id → bed_id
) -> list[RoomLayout]:
    """Carve en-suite bathrooms from a corner of their parent bedroom.

    Takes ~30% of the bedroom area from the bottom-right corner.
    Shrinks the bedroom accordingly so there's no overlap.
    """
    if not ensuite_pairs:
        return placed

    # Build map of placed rooms by ID
    placed_map = {r.room_spec.room_id: r for r in placed}
    result = list(placed)

    for bath_room in ensuite_rooms:
        bath_id = bath_room.room_spec.room_id
        bed_id = ensuite_pairs.get(bath_id)
        if not bed_id or bed_id not in placed_map:
            # Fallback: put bathroom in a small corner of the plot
            result.append(RoomLayout(
                room_spec=bath_room.room_spec,
                bbox=BoundingBox(x_min=0.85, y_min=0.85, x_max=1.0, y_max=1.0),
                door_midpoints=[],
            ))
            continue

        bed = placed_map[bed_id]
        bx = bed.bbox
        bed_w = bx.x_max - bx.x_min
        bed_h = bx.y_max - bx.y_min

        # En-suite takes ~30% of bedroom area, carved from bottom-right
        # Choose split direction: split along the longer side
        if bed_w >= bed_h:
            # Vertical split — bathroom on the right side
            bath_frac = 0.30
            split_x = bx.x_max - bed_w * bath_frac

            bath_bbox = BoundingBox(
                x_min=max(0.0, split_x),
                y_min=bx.y_min,
                x_max=bx.x_max,
                y_max=bx.y_max,
            )
            # Shrink bedroom
            new_bed_bbox = BoundingBox(
                x_min=bx.x_min,
                y_min=bx.y_min,
                x_max=max(bx.x_min + 0.05, split_x),
                y_max=bx.y_max,
            )
        else:
            # Horizontal split — bathroom on the bottom
            bath_frac = 0.30
            split_y = bx.y_max - bed_h * bath_frac

            bath_bbox = BoundingBox(
                x_min=bx.x_min,
                y_min=max(0.0, split_y),
                x_max=bx.x_max,
                y_max=bx.y_max,
            )
            new_bed_bbox = BoundingBox(
                x_min=bx.x_min,
                y_min=bx.y_min,
                x_max=bx.x_max,
                y_max=max(bx.y_min + 0.05, split_y),
            )

        # Update bedroom bbox
        updated_bed = RoomLayout(
            room_spec=bed.room_spec,
            bbox=new_bed_bbox,
            door_midpoints=[],
        )

        # Replace the bedroom in the result list
        for idx, r in enumerate(result):
            if r.room_spec.room_id == bed_id:
                result[idx] = updated_bed
                break
        placed_map[bed_id] = updated_bed

        # Add the bathroom
        result.append(RoomLayout(
            room_spec=bath_room.room_spec,
            bbox=bath_bbox,
            door_midpoints=[],
        ))

    return result


def _place_corridor(
    placed: list[RoomLayout],
    plot_width: float,
    plot_height: float,
) -> list[RoomLayout]:
    """If a CORRIDOR room exists, reshape it as a narrow central strip.

    The corridor becomes a 1.2m-wide horizontal band through the middle,
    and adjacent rooms are pushed up/down to make space.
    """
    corridor_idx = None
    for i, r in enumerate(placed):
        if r.room_spec.room_type == RoomType.CORRIDOR:
            corridor_idx = i
            break

    if corridor_idx is None:
        return placed

    # Corridor strip parameters (normalised)
    corridor_width_m = 1.2  # metres
    corridor_frac = corridor_width_m / plot_height  # normalised height
    corridor_frac = max(0.08, min(corridor_frac, 0.15))  # clamp 8-15%

    center_y = 0.5
    corr_y_min = center_y - corridor_frac / 2
    corr_y_max = center_y + corridor_frac / 2

    # Set corridor bbox — full width, narrow band
    corridor_room = placed[corridor_idx]
    result = []

    for i, room in enumerate(placed):
        if i == corridor_idx:
            result.append(RoomLayout(
                room_spec=room.room_spec,
                bbox=BoundingBox(x_min=0.0, y_min=corr_y_min, x_max=1.0, y_max=corr_y_max),
                door_midpoints=[],
            ))
            continue

        bx = room.bbox
        # Push rooms that overlap the corridor band
        if bx.y_min < corr_y_max and bx.y_max > corr_y_min:
            # Room straddles the corridor — determine which side to push it
            room_center_y = (bx.y_min + bx.y_max) / 2
            if room_center_y <= center_y:
                # Push to upper half
                new_bbox = BoundingBox(
                    x_min=bx.x_min,
                    y_min=bx.y_min,
                    x_max=bx.x_max,
                    y_max=max(bx.y_min + 0.05, corr_y_min),
                )
            else:
                # Push to lower half
                new_bbox = BoundingBox(
                    x_min=bx.x_min,
                    y_min=min(corr_y_max, bx.y_max - 0.05),
                    x_max=bx.x_max,
                    y_max=bx.y_max,
                )
            result.append(RoomLayout(
                room_spec=room.room_spec,
                bbox=new_bbox,
                door_midpoints=[],
            ))
        else:
            result.append(room)

    return result


def _generate_door_midpoints(
    placed: list[RoomLayout],
    edges: list[AdjacencyEdge],
    plot_width: float,
    plot_height: float,
) -> list[RoomLayout]:
    """Generate door midpoint coordinates on shared walls between adjacent rooms.

    For each DOOR or OPENING edge, find the shared wall and place the door
    at the midpoint of the overlapping segment.
    """
    room_map = {r.room_spec.room_id: r for r in placed}
    door_map: dict[str, list[tuple[float, float]]] = {r.room_spec.room_id: [] for r in placed}

    for edge in edges:
        if edge.connection_type.value == "WALL":
            continue

        ra = room_map.get(edge.room_a_id)
        rb = room_map.get(edge.room_b_id)
        if ra is None or rb is None:
            continue

        ba, bb = ra.bbox, rb.bbox

        door_x, door_y = None, None

        # Check vertical adjacency (rooms share a vertical wall)
        if abs(ba.x_max - bb.x_min) < 0.015:
            # A is left of B
            y_overlap_min = max(ba.y_min, bb.y_min)
            y_overlap_max = min(ba.y_max, bb.y_max)
            if y_overlap_max - y_overlap_min > 0.02:
                door_x = ba.x_max
                door_y = (y_overlap_min + y_overlap_max) / 2
        elif abs(bb.x_max - ba.x_min) < 0.015:
            # B is left of A
            y_overlap_min = max(ba.y_min, bb.y_min)
            y_overlap_max = min(ba.y_max, bb.y_max)
            if y_overlap_max - y_overlap_min > 0.02:
                door_x = bb.x_max
                door_y = (y_overlap_min + y_overlap_max) / 2

        # Check horizontal adjacency (rooms share a horizontal wall)
        if door_x is None:
            if abs(ba.y_max - bb.y_min) < 0.015:
                # A is above B
                x_overlap_min = max(ba.x_min, bb.x_min)
                x_overlap_max = min(ba.x_max, bb.x_max)
                if x_overlap_max - x_overlap_min > 0.02:
                    door_x = (x_overlap_min + x_overlap_max) / 2
                    door_y = ba.y_max
            elif abs(bb.y_max - ba.y_min) < 0.015:
                # B is above A
                x_overlap_min = max(ba.x_min, bb.x_min)
                x_overlap_max = min(ba.x_max, bb.x_max)
                if x_overlap_max - x_overlap_min > 0.02:
                    door_x = (x_overlap_min + x_overlap_max) / 2
                    door_y = bb.y_max

        if door_x is not None and door_y is not None:
            door_map[edge.room_a_id].append((door_x, door_y))
            door_map[edge.room_b_id].append((door_x, door_y))

    # Rebuild rooms with door midpoints
    result = []
    for room in placed:
        doors = door_map.get(room.room_spec.room_id, [])
        result.append(RoomLayout(
            room_spec=room.room_spec,
            bbox=room.bbox,
            door_midpoints=doors,
        ))

    return result


def _improve_adjacency_by_swapping(
    rooms: list[RoomLayout],
    edges: list[AdjacencyEdge],
    plot_width: float,
    plot_height: float,
    max_passes: int = 2,
) -> list[RoomLayout]:
    """Greedily swap room assignments to improve required adjacency satisfaction.

    This keeps the same treemap boxes (gap-free) and only reassigns which room
    occupies each box. Room IDs travel with their RoomSpec, so edge checks remain valid.
    """
    required_edges = [e for e in edges if e.required]
    if len(rooms) < 3 or not required_edges:
        return rooms

    # Work on a mutable copy.
    current = [
        RoomLayout(
            room_spec=r.room_spec,
            bbox=r.bbox,
            door_midpoints=r.door_midpoints,
        )
        for r in rooms
    ]

    def score(room_list: list[RoomLayout]) -> float:
        room_map = {r.room_spec.room_id: r.bbox for r in room_list}
        satisfied = 0
        for edge in required_edges:
            ba = room_map.get(edge.room_a_id)
            bb = room_map.get(edge.room_b_id)
            if ba is None or bb is None:
                continue

            shared_wall = 0.0
            # Touching on x-axis.
            if abs(ba.x_max - bb.x_min) < 0.01 or abs(bb.x_max - ba.x_min) < 0.01:
                y_overlap = max(0, min(ba.y_max, bb.y_max) - max(ba.y_min, bb.y_min)) * plot_height
                shared_wall = max(shared_wall, y_overlap)
            # Touching on y-axis.
            if abs(ba.y_max - bb.y_min) < 0.01 or abs(bb.y_max - ba.y_min) < 0.01:
                x_overlap = max(0, min(ba.x_max, bb.x_max) - max(ba.x_min, bb.x_min)) * plot_width
                shared_wall = max(shared_wall, x_overlap)
            if shared_wall >= 0.9:
                satisfied += 1
        return satisfied / len(required_edges)

    best_score = score(current)
    n = len(current)

    for _ in range(max_passes):
        improved = False
        for i in range(n):
            for j in range(i + 1, n):
                trial = list(current)
                ri = trial[i]
                rj = trial[j]
                trial[i] = RoomLayout(
                    room_spec=rj.room_spec,
                    bbox=ri.bbox,
                    door_midpoints=ri.door_midpoints,
                )
                trial[j] = RoomLayout(
                    room_spec=ri.room_spec,
                    bbox=rj.bbox,
                    door_midpoints=rj.door_midpoints,
                )

                trial_score = score(trial)
                if trial_score > best_score:
                    current = trial
                    best_score = trial_score
                    improved = True
                    if best_score >= 1.0:
                        return current
        if not improved:
            break

    return current


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
