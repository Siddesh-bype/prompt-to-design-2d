"""
Pydantic v2 schemas — single source of truth for all data contracts.

These schemas are shared between:
- NLP parser output
- GNN input/output
- API request/response
- Frontend JSON contract
"""

from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


# ─── Enums ───────────────────────────────────────────────────────────────────


class RoomType(str, Enum):
    """Types of rooms in a floor plan."""
    LIVING_ROOM = "LIVING_ROOM"
    MASTER_BEDROOM = "MASTER_BEDROOM"
    BEDROOM = "BEDROOM"
    KITCHEN = "KITCHEN"
    BATHROOM = "BATHROOM"
    TOILET = "TOILET"
    CORRIDOR = "CORRIDOR"
    BALCONY = "BALCONY"
    STUDY = "STUDY"
    DINING = "DINING"
    UTILITY = "UTILITY"
    GARAGE = "GARAGE"


class CompassFacing(str, Enum):
    """Cardinal compass directions."""
    NORTH = "NORTH"
    SOUTH = "SOUTH"
    EAST = "EAST"
    WEST = "WEST"


class ConnectionType(str, Enum):
    """Types of connections between rooms."""
    DOOR = "DOOR"
    OPENING = "OPENING"
    WALL = "WALL"


# ─── Core Models ─────────────────────────────────────────────────────────────


class RoomSpec(BaseModel):
    """Specification for a single room."""
    room_id: str = Field(..., description="Unique identifier for the room")
    room_type: RoomType = Field(..., description="Type of room")
    target_area_sqm: float = Field(
        ..., ge=4.0, le=200.0,
        description="Target area in square metres (4.0–200.0)"
    )
    compass_preference: Optional[CompassFacing] = Field(
        None, description="Preferred compass direction for this room"
    )
    label: Optional[str] = Field(
        None, description="Custom display label for the room"
    )


class AdjacencyEdge(BaseModel):
    """Edge representing a connection between two rooms."""
    room_a_id: str = Field(..., description="ID of the first room")
    room_b_id: str = Field(..., description="ID of the second room")
    connection_type: ConnectionType = Field(
        ..., description="Type of connection"
    )
    required: bool = Field(
        True, description="Whether this adjacency is required"
    )


class ParsedLayout(BaseModel):
    """Output of the NLP parser — structured floor plan specification."""
    rooms: list[RoomSpec] = Field(
        ..., min_length=2, max_length=30,
        description="List of room specifications (2–30 rooms)"
    )
    plot_area_sqm: float = Field(
        ..., ge=30.0, le=2000.0,
        description="Total plot area in square metres (30.0–2000.0)"
    )
    facing: CompassFacing = Field(
        ..., description="Primary compass facing of the plot"
    )
    style_hints: list[str] = Field(
        default=[], description="Style hints from the prompt"
    )
    adjacency_constraints: list[AdjacencyEdge] = Field(
        default=[], description="Required adjacency relationships"
    )
    vastu_enabled: bool = Field(
        False, description="Whether Vastu Shastra rules should be applied"
    )


# ─── Geometry Models ─────────────────────────────────────────────────────────


class BoundingBox(BaseModel):
    """Normalised bounding box in [0.0, 1.0] coordinate space."""
    x_min: float = Field(..., ge=0.0, le=1.0)
    y_min: float = Field(..., ge=0.0, le=1.0)
    x_max: float = Field(..., ge=0.0, le=1.0)
    y_max: float = Field(..., ge=0.0, le=1.0)

    @property
    def area(self) -> float:
        """Compute area of the bounding box in normalised units."""
        return (self.x_max - self.x_min) * (self.y_max - self.y_min)

    def to_pixel_coords(
        self, plot_width_px: int, plot_height_px: int
    ) -> tuple[int, int, int, int]:
        """Convert normalised coords to pixel coordinates.

        Returns (x_min_px, y_min_px, x_max_px, y_max_px).
        """
        return (
            int(self.x_min * plot_width_px),
            int(self.y_min * plot_height_px),
            int(self.x_max * plot_width_px),
            int(self.y_max * plot_height_px),
        )


class RoomLayout(BaseModel):
    """A room with its computed bounding box and door positions."""
    room_spec: RoomSpec
    bbox: BoundingBox
    door_midpoints: list[tuple[float, float]] = Field(
        default=[], description="Door midpoint coordinates [(x, y), ...]"
    )


class LayoutGraph(BaseModel):
    """Complete layout graph — the primary output of the generation pipeline."""
    rooms: list[RoomLayout] = Field(
        ..., description="All rooms with computed positions"
    )
    adjacency_edges: list[AdjacencyEdge] = Field(
        default=[], description="Adjacency relationships between rooms"
    )
    plot_area_sqm: float = Field(
        ..., description="Total plot area in square metres"
    )
    facing: CompassFacing = Field(
        ..., description="Primary compass facing"
    )
    generation_mode: Literal["gnn", "heuristic"] = Field(
        "gnn", description="Which generation method was used"
    )
    overlap_rate: float = Field(
        0.0, description="Fraction of overlapping area (0.0–1.0)"
    )
    adjacency_satisfaction: float = Field(
        0.0, description="Fraction of required adjacencies satisfied (0.0–1.0)"
    )


# ─── API Request / Response Models ───────────────────────────────────────────


class ErrorDetail(BaseModel):
    """Structured error information."""
    error_code: str = Field(..., description="Machine-readable error code")
    message: str = Field(..., description="Human-readable error message")
    fallback_used: bool = Field(
        False, description="Whether a fallback strategy was used"
    )
    details: dict = Field(
        default={}, description="Additional error context"
    )


class GenerateRequest(BaseModel):
    """Request body for POST /api/v1/generate."""
    prompt: str = Field(
        ..., min_length=10, max_length=800,
        description="Natural language description of the floor plan"
    )
    plot_sqm: float = Field(
        100.0, ge=30.0, le=500.0,
        description="Plot area in square metres"
    )
    facing: CompassFacing = Field(
        CompassFacing.NORTH, description="Primary compass facing"
    )
    vastu_enabled: bool = Field(
        False, description="Enable Vastu Shastra compliance checking"
    )


class GenerateResponse(BaseModel):
    """Response body for POST /api/v1/generate."""
    job_id: str = Field(..., description="Unique job identifier")
    ws_url: str = Field(..., description="WebSocket URL for real-time updates")
    estimated_seconds: int = Field(
        ..., description="Estimated processing time"
    )


class JobStatusResponse(BaseModel):
    """Response body for GET /api/v1/jobs/{job_id}."""
    status: Literal[
        "queued", "parsing", "layout_generating",
        "solving_constraints", "rendering", "complete", "error"
    ] = Field(..., description="Current job status")
    progress_pct: int = Field(
        ..., ge=0, le=100, description="Progress percentage"
    )
    result: Optional[LayoutGraph] = Field(
        None, description="Completed layout graph"
    )
    svg_string: Optional[str] = Field(
        None, description="Generated SVG string"
    )
    error: Optional[ErrorDetail] = Field(
        None, description="Error details if status is 'error'"
    )


class VastuRequest(BaseModel):
    """Request body for POST /api/v1/vastu."""
    layout_graph: LayoutGraph = Field(
        ..., description="Layout graph to check"
    )


class VastuResponse(BaseModel):
    """Response body for POST /api/v1/vastu."""
    score: int = Field(
        ..., ge=0, le=100, description="Vastu compliance score"
    )
    suggestions: list[str] = Field(
        default=[], description="Improvement suggestions"
    )
    zone_map: dict[str, str] = Field(
        default={}, description="Room-to-zone mapping"
    )


class ExportRequest(BaseModel):
    """Request body for POST /api/v1/export/dxf and /svg."""
    layout_graph: LayoutGraph = Field(
        ..., description="Layout graph to export"
    )
    scale: Literal["1:50", "1:100", "1:200"] = Field(
        "1:100", description="Export scale"
    )
