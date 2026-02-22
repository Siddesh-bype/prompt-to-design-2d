"""
Vastu Shastra Engine — 9-zone grid checker for architectural compliance.

Divides the plot into a 3×3 grid and scores room placements based on
traditional Vastu Shastra principles.
"""

from __future__ import annotations

import logging
from typing import Optional

from app.models.schemas import (
    BoundingBox,
    LayoutGraph,
    RoomLayout,
    RoomType,
    VastuResponse,
)

logger = logging.getLogger(__name__)


# ─── Zone Mapping ────────────────────────────────────────────────────────────

# 3×3 grid zones at normalised [0,1] coords:
#  NW  |  N  |  NE
#  ----+-----+----
#   W  | CTR |  E
#  ----+-----+----
#  SW  |  S  |  SE


def get_vastu_zone(x_norm: float, y_norm: float) -> str:
    """Determine Vastu zone from normalised [0,1] coordinates.

    Grid layout (y increases downward, matching screen coords):
        Row 0 (y: 0.0–0.33): NW, N, NE  (top = North)
        Row 1 (y: 0.33–0.67): W, CENTER, E
        Row 2 (y: 0.67–1.0): SW, S, SE  (bottom = South)

    Args:
        x_norm: Normalised x coordinate [0, 1]
        y_norm: Normalised y coordinate [0, 1]

    Returns:
        Zone string: NW, N, NE, W, CENTER, E, SW, S, SE
    """
    # Column
    if x_norm < 1/3:
        col = 0  # West
    elif x_norm < 2/3:
        col = 1  # Center
    else:
        col = 2  # East

    # Row (y=0 is North/top)
    if y_norm < 1/3:
        row = 0  # North
    elif y_norm < 2/3:
        row = 1  # Center
    else:
        row = 2  # South

    zone_grid = [
        ["NW", "N", "NE"],
        ["W", "CENTER", "E"],
        ["SW", "S", "SE"],
    ]

    return zone_grid[row][col]


# ─── Vastu Rules ─────────────────────────────────────────────────────────────

# Scoring: (zone, RoomType) → score (0–100)
# Higher = more auspicious placement

VASTU_RULES: dict[tuple[str, RoomType], int] = {
    # Kitchen — SE is ideal
    ("SE", RoomType.KITCHEN): 100,
    ("S", RoomType.KITCHEN): 80,
    ("E", RoomType.KITCHEN): 70,
    ("NE", RoomType.KITCHEN): 20,  # Bad: kitchen in NE
    ("NW", RoomType.KITCHEN): 40,
    ("SW", RoomType.KITCHEN): 40,
    ("N", RoomType.KITCHEN): 50,
    ("W", RoomType.KITCHEN): 50,

    # Master Bedroom — SW is ideal
    ("SW", RoomType.MASTER_BEDROOM): 100,
    ("S", RoomType.MASTER_BEDROOM): 85,
    ("W", RoomType.MASTER_BEDROOM): 80,
    ("NE", RoomType.MASTER_BEDROOM): 20,  # Bad
    ("SE", RoomType.MASTER_BEDROOM): 20,  # Bad
    ("NW", RoomType.MASTER_BEDROOM): 60,
    ("N", RoomType.MASTER_BEDROOM): 50,
    ("E", RoomType.MASTER_BEDROOM): 50,

    # Living Room — NE or N is ideal
    ("NE", RoomType.LIVING_ROOM): 100,
    ("N", RoomType.LIVING_ROOM): 95,
    ("E", RoomType.LIVING_ROOM): 85,
    ("NW", RoomType.LIVING_ROOM): 70,
    ("SE", RoomType.LIVING_ROOM): 50,
    ("SW", RoomType.LIVING_ROOM): 40,
    ("S", RoomType.LIVING_ROOM): 50,
    ("W", RoomType.LIVING_ROOM): 60,

    # Bathroom — NW or W
    ("NW", RoomType.BATHROOM): 100,
    ("W", RoomType.BATHROOM): 90,
    ("N", RoomType.BATHROOM): 70,
    ("SW", RoomType.BATHROOM): 60,
    ("NE", RoomType.BATHROOM): 40,
    ("SE", RoomType.BATHROOM): 50,
    ("E", RoomType.BATHROOM): 50,
    ("S", RoomType.BATHROOM): 50,

    # Toilet — same as Bathroom
    ("NW", RoomType.TOILET): 100,
    ("W", RoomType.TOILET): 90,
    ("N", RoomType.TOILET): 70,

    # Study / Pooja — NE
    ("NE", RoomType.STUDY): 100,
    ("N", RoomType.STUDY): 85,
    ("E", RoomType.STUDY): 80,
    ("NW", RoomType.STUDY): 60,
    ("SW", RoomType.STUDY): 30,

    # Regular Bedroom — S or W
    ("S", RoomType.BEDROOM): 80,
    ("W", RoomType.BEDROOM): 80,
    ("SW", RoomType.BEDROOM): 85,
    ("NW", RoomType.BEDROOM): 70,
    ("NE", RoomType.BEDROOM): 50,
    ("SE", RoomType.BEDROOM): 60,

    # Dining — W
    ("W", RoomType.DINING): 90,
    ("NW", RoomType.DINING): 80,
    ("S", RoomType.DINING): 70,

    # Balcony — N or E
    ("N", RoomType.BALCONY): 90,
    ("E", RoomType.BALCONY): 90,
    ("NE", RoomType.BALCONY): 95,

    # Garage — NW or SE
    ("NW", RoomType.GARAGE): 85,
    ("SE", RoomType.GARAGE): 80,

    # Utility — NW
    ("NW", RoomType.UTILITY): 80,
    ("W", RoomType.UTILITY): 70,
}

# Default score for any room in CENTER zone
CENTER_DEFAULT_SCORE = 60

# Default score when no rule found
DEFAULT_SCORE = 60


# ─── Main Checker ────────────────────────────────────────────────────────────


def check_vastu(layout: LayoutGraph) -> VastuResponse:
    """Check Vastu Shastra compliance for a layout.

    For each room:
    1. Find its zone using centroid of BoundingBox
    2. Look up score from VASTU_RULES (default 60)
    3. Compute weighted average by room area
    4. Generate suggestions for rooms scoring below 50

    Args:
        layout: The LayoutGraph to check

    Returns:
        VastuResponse with score, suggestions, and zone_map
    """
    if not layout.rooms:
        return VastuResponse(score=0, suggestions=[], zone_map={})

    scores = []
    weights = []
    zone_map: dict[str, str] = {}
    suggestions: list[str] = []

    for room in layout.rooms:
        # Compute centroid
        cx = (room.bbox.x_min + room.bbox.x_max) / 2
        cy = (room.bbox.y_min + room.bbox.y_max) / 2

        # Get zone
        zone = get_vastu_zone(cx, cy)
        room_name = room.room_spec.label or room.room_spec.room_type.value.replace("_", " ").title()
        zone_map[zone] = room_name

        # Look up score
        key = (zone, room.room_spec.room_type)
        if zone == "CENTER":
            score = VASTU_RULES.get(key, CENTER_DEFAULT_SCORE)
        else:
            score = VASTU_RULES.get(key, DEFAULT_SCORE)

        # Weight by room area
        area = room.bbox.area
        scores.append(score)
        weights.append(max(area, 0.01))

        # Generate suggestion if score < 50
        if score < 50:
            ideal = _get_ideal_zone(room.room_spec.room_type)
            suggestions.append(
                f"Move {room_name} from {zone} to {ideal} zone for better Vastu compliance "
                f"(current score: {score}/100)"
            )

    # Weighted average
    total_weight = sum(weights)
    composite_score = sum(s * w for s, w in zip(scores, weights)) / total_weight if total_weight > 0 else 0
    composite_score = int(round(composite_score))
    composite_score = max(0, min(100, composite_score))

    logger.info(f"Vastu check: composite score = {composite_score}/100, {len(suggestions)} suggestions")

    return VastuResponse(
        score=composite_score,
        suggestions=suggestions,
        zone_map=zone_map,
    )


def _get_ideal_zone(room_type: RoomType) -> str:
    """Get the ideal Vastu zone for a given room type."""
    ideal_zones = {
        RoomType.KITCHEN: "SE",
        RoomType.MASTER_BEDROOM: "SW",
        RoomType.LIVING_ROOM: "NE",
        RoomType.BATHROOM: "NW",
        RoomType.TOILET: "NW",
        RoomType.STUDY: "NE",
        RoomType.BEDROOM: "S or W",
        RoomType.DINING: "W",
        RoomType.BALCONY: "N or E",
        RoomType.CORRIDOR: "CENTER",
        RoomType.UTILITY: "NW",
        RoomType.GARAGE: "NW or SE",
    }
    return ideal_zones.get(room_type, "CENTER")
