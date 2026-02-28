"""
GNN Layout Engine — Graph Neural Network for room layout generation.

Uses PyTorch Geometric GATv2Conv layers to predict bounding boxes
for rooms based on their features and adjacency relationships.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Try to import PyG — graceful fallback if not installed
try:
    import torch_geometric
    from torch_geometric.data import Data
    from torch_geometric.nn import GATv2Conv

    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    GATv2Conv = None  # type: ignore
    Data = None  # type: ignore
    logger.warning("torch_geometric not installed — GNN features will be limited")

# Import schemas
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


# ─── Constants ───────────────────────────────────────────────────────────────

NUM_ROOM_TYPES = 12
NODE_FEAT_DIM = 18
EDGE_FEAT_DIM = 3
HIDDEN_DIM = 128
NUM_HEADS = 4
OUTPUT_DIM = 4  # [x_min, y_min, x_max, y_max]

# Room type-specific minimum sizes (as fraction of plot for a standard 10m x 10m plot)
# These ensure realistic room dimensions based on typical architectural standards
ROOM_MIN_SIZES: dict[RoomType, tuple[float, float]] = {
    RoomType.LIVING_ROOM: (0.20, 0.15),  # min 20% width, 15% height
    RoomType.KITCHEN: (0.12, 0.10),  # min 12% width, 10% height
    RoomType.MASTER_BEDROOM: (0.18, 0.15),  # min 18% width, 15% height
    RoomType.BEDROOM: (0.12, 0.12),  # min 12% width, 12% height
    RoomType.BATHROOM: (0.08, 0.08),  # min 8% width, 8% height
    RoomType.TOILET: (0.06, 0.06),  # min 6% width, 6% height
    RoomType.CORRIDOR: (0.04, 0.20),  # min 4% width, 20% height (long narrow)
    RoomType.BALCONY: (0.10, 0.08),  # min 10% width, 8% height
    RoomType.STUDY: (0.10, 0.10),  # min 10% width, 10% height
    RoomType.DINING: (0.12, 0.10),  # min 12% width, 10% height
    RoomType.UTILITY: (0.08, 0.08),  # min 8% width, 8% height
    RoomType.GARAGE: (0.15, 0.15),  # min 15% width, 15% height
}

# Room type aspect ratio preferences (width/height)
# Helps generate more realistic room shapes
ROOM_ASPECT_RATIOS: dict[RoomType, tuple[float, float]] = {
    RoomType.LIVING_ROOM: (1.0, 1.5),  # prefers wider rooms
    RoomType.KITCHEN: (0.8, 1.3),  # moderately proportioned
    RoomType.MASTER_BEDROOM: (1.0, 1.4),  # prefers wider rooms
    RoomType.BEDROOM: (0.9, 1.3),  # slightly rectangular
    RoomType.BATHROOM: (0.8, 1.2),  # can be square or rectangular
    RoomType.TOILET: (0.7, 1.0),  # usually small rectangular
    RoomType.CORRIDOR: (0.2, 0.5),  # very narrow
    RoomType.BALCONY: (1.5, 4.0),  # long and narrow
    RoomType.STUDY: (0.9, 1.2),  # roughly square
    RoomType.DINING: (1.0, 1.5),  # prefers wider
    RoomType.UTILITY: (0.7, 1.2),  # small rectangular
    RoomType.GARAGE: (1.0, 1.5),  # rectangular
}


# ─── Model Definition ───────────────────────────────────────────────────────


class FloorPlanGNN(nn.Module):
    """GATv2-based Graph Neural Network for floor plan layout generation.

    Architecture:
        - 3 GATv2Conv layers (18→128→128→128) with 4 attention heads
        - Edge feature integration via linear projection
        - Bbox output head: Linear(128, 4) → Sigmoid
        - Door head: Linear(128, 2) per edge → door midpoint
    """

    def __init__(
        self,
        in_channels: int = NODE_FEAT_DIM,
        hidden_channels: int = HIDDEN_DIM,
        num_heads: int = NUM_HEADS,
        edge_dim: int = EDGE_FEAT_DIM,
    ):
        super().__init__()

        # GATv2 layers
        self.conv1 = (
            GATv2Conv(
                in_channels,
                hidden_channels,
                heads=num_heads,
                concat=True,
                edge_dim=edge_dim,
            )
            if HAS_PYG
            else None
        )

        self.conv2 = (
            GATv2Conv(
                hidden_channels * num_heads,
                hidden_channels,
                heads=num_heads,
                concat=True,
                edge_dim=edge_dim,
            )
            if HAS_PYG
            else None
        )

        self.conv3 = (
            GATv2Conv(
                hidden_channels * num_heads,
                hidden_channels,
                heads=num_heads,
                concat=False,
                edge_dim=edge_dim,
            )
            if HAS_PYG
            else None
        )

        # Edge feature projections
        self.edge_proj1 = nn.Linear(edge_dim, edge_dim)
        self.edge_proj2 = nn.Linear(edge_dim, edge_dim)
        self.edge_proj3 = nn.Linear(edge_dim, edge_dim)

        # Batch normalization
        self.bn1 = nn.BatchNorm1d(hidden_channels * num_heads)
        self.bn2 = nn.BatchNorm1d(hidden_channels * num_heads)
        self.bn3 = nn.BatchNorm1d(hidden_channels)

        # Output heads
        self.bbox_head = nn.Linear(hidden_channels, OUTPUT_DIM)
        self.door_head = nn.Linear(
            hidden_channels * 2, 2
        )  # Concat of two node embeddings

        # Dropout
        self.dropout = nn.Dropout(0.1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass.

        Args:
            x: Node features [N, 18]
            edge_index: Edge indices [2, E]
            edge_attr: Edge features [E, 3]

        Returns:
            bbox_pred: Predicted bounding boxes [N, 4] in [0, 1]
            door_pred: Predicted door midpoints [E, 2] in [0, 1] (or None)
        """
        if not HAS_PYG:
            # Fallback: random predictions
            N = x.size(0)
            E = edge_index.size(1) if edge_index is not None else 0
            bbox = torch.sigmoid(torch.randn(N, 4))
            doors = torch.sigmoid(torch.randn(E, 2)) if E > 0 else None
            return bbox, doors

        # Layer 1
        e1 = self.edge_proj1(edge_attr) if edge_attr is not None else None
        x = self.conv1(x, edge_index, edge_attr=e1)
        x = self.bn1(x)
        x = F.elu(x)
        x = self.dropout(x)

        # Layer 2
        e2 = self.edge_proj2(edge_attr) if edge_attr is not None else None
        x = self.conv2(x, edge_index, edge_attr=e2)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.dropout(x)

        # Layer 3
        e3 = self.edge_proj3(edge_attr) if edge_attr is not None else None
        x = self.conv3(x, edge_index, edge_attr=e3)
        x = self.bn3(x)
        x = F.elu(x)

        # Bbox output
        bbox_pred = torch.sigmoid(self.bbox_head(x))

        # Door output (for each edge, concat src and dst embeddings)
        door_pred = None
        if edge_index is not None and edge_index.size(1) > 0:
            src_emb = x[edge_index[0]]
            dst_emb = x[edge_index[1]]
            edge_emb = torch.cat([src_emb, dst_emb], dim=-1)
            door_pred = torch.sigmoid(self.door_head(edge_emb))

        return bbox_pred, door_pred


# ─── Feature Builders ───────────────────────────────────────────────────────


def build_node_features(room_spec: RoomSpec, adjacency_count: int = 0) -> torch.Tensor:
    """Build an 18-dimensional node feature vector for a room.

    Features:
        [0:12]  room_type one-hot (12 classes)
        [12]    target_area_sqm normalised (÷ 80.0)
        [13]    adjacency_rank (adjacency_count / 6.0)
        [14:18] compass_preference one-hot (N/S/E/W)

    Args:
        room_spec: The room specification
        adjacency_count: Number of required edges for this room

    Returns:
        Feature tensor of shape (18,)
    """
    features = torch.zeros(NODE_FEAT_DIM)

    # Room type one-hot [0:12]
    room_type_list = list(RoomType)
    type_idx = room_type_list.index(room_spec.room_type)
    features[type_idx] = 1.0

    # Normalised area [12]
    features[12] = room_spec.target_area_sqm / 80.0

    # Adjacency rank [13]
    features[13] = min(adjacency_count / 6.0, 1.0)

    # Compass preference one-hot [14:18]
    if room_spec.compass_preference is not None:
        compass_map = {
            CompassFacing.NORTH: 14,
            CompassFacing.SOUTH: 15,
            CompassFacing.EAST: 16,
            CompassFacing.WEST: 17,
        }
        features[compass_map[room_spec.compass_preference]] = 1.0

    return features


def build_edge_features(edge: AdjacencyEdge) -> torch.Tensor:
    """Build a 3-dimensional edge feature vector.

    Features:
        [0:2]  connection_type one-hot (DOOR, OPENING — WALL maps to [0,0])
        [2]    required flag (1.0 if required else 0.5)

    Args:
        edge: The adjacency edge

    Returns:
        Feature tensor of shape (3,)
    """
    features = torch.zeros(EDGE_FEAT_DIM)

    if edge.connection_type == ConnectionType.DOOR:
        features[0] = 1.0
    elif edge.connection_type == ConnectionType.OPENING:
        features[1] = 1.0
    # WALL maps to [0, 0]

    features[2] = 1.0 if edge.required else 0.5

    return features


# ─── PyG Data Conversion ────────────────────────────────────────────────────


def parsed_layout_to_pyg(layout: ParsedLayout) -> "Data":
    """Convert a ParsedLayout into a PyTorch Geometric Data object.

    Nodes = rooms, edges = adjacency_constraints (made bidirectional).

    Args:
        layout: Parsed layout from NLP parser

    Returns:
        PyG Data(x, edge_index, edge_attr)
    """
    if not HAS_PYG:
        raise ImportError("torch_geometric is required for parsed_layout_to_pyg")

    # Build room_id to index mapping
    room_id_to_idx = {room.room_id: i for i, room in enumerate(layout.rooms)}

    # Count adjacencies per room
    adj_counts = {room.room_id: 0 for room in layout.rooms}
    for edge in layout.adjacency_constraints:
        if edge.room_a_id in adj_counts:
            adj_counts[edge.room_a_id] += 1
        if edge.room_b_id in adj_counts:
            adj_counts[edge.room_b_id] += 1

    # Build node features
    node_features = torch.stack(
        [
            build_node_features(room, adj_counts.get(room.room_id, 0))
            for room in layout.rooms
        ]
    )

    # Build edges (bidirectional)
    edge_src, edge_dst = [], []
    edge_attrs = []

    for edge in layout.adjacency_constraints:
        if edge.room_a_id in room_id_to_idx and edge.room_b_id in room_id_to_idx:
            a_idx = room_id_to_idx[edge.room_a_id]
            b_idx = room_id_to_idx[edge.room_b_id]

            # Forward
            edge_src.append(a_idx)
            edge_dst.append(b_idx)
            edge_attrs.append(build_edge_features(edge))

            # Backward
            edge_src.append(b_idx)
            edge_dst.append(a_idx)
            edge_attrs.append(build_edge_features(edge))

    if edge_src:
        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
        edge_attr = torch.stack(edge_attrs)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, EDGE_FEAT_DIM))

    return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)


# ─── Inference Wrapper ───────────────────────────────────────────────────────


# Room type preferred positions on the plot
# (x_range, y_range) - normalized 0-1 coordinates
# x: 0=left/front, 1=right/back; y: 0=bottom, 1=top
ROOM_PREFERRED_ZONES: dict[
    RoomType, tuple[tuple[float, float], tuple[float, float]]
] = {
    RoomType.LIVING_ROOM: ((0.0, 0.7), (0.3, 0.7)),  # Front-center
    RoomType.KITCHEN: ((0.3, 1.0), (0.0, 0.4)),  # Back-left
    RoomType.MASTER_BEDROOM: ((0.5, 1.0), (0.5, 1.0)),  # Back-right
    RoomType.BEDROOM: ((0.0, 1.0), (0.5, 1.0)),  # Back area
    RoomType.BATHROOM: ((0.2, 0.8), (0.2, 0.8)),  # Middle area
    RoomType.TOILET: ((0.0, 1.0), (0.1, 0.5)),  # Near entrance area
    RoomType.CORRIDOR: ((0.3, 0.7), (0.0, 1.0)),  # Center corridor
    RoomType.BALCONY: ((0.0, 1.0), (0.8, 1.0)),  # Edge/back
    RoomType.STUDY: ((0.0, 1.0), (0.3, 0.7)),  # Side area
    RoomType.DINING: ((0.2, 0.8), (0.2, 0.6)),  # Near living/kitchen
    RoomType.UTILITY: ((0.7, 1.0), (0.0, 0.3)),  # Corner utility
    RoomType.GARAGE: ((0.0, 0.3), (0.0, 0.3)),  # Front corner
}


def run_gnn_inference(
    parsed_layout: ParsedLayout,
    model_path: str = "models/floorplan_gnn.pt",
) -> LayoutGraph:
    """Run GNN inference on a ParsedLayout to generate room bounding boxes.

    Args:
        parsed_layout: Parsed layout from NLP parser
        model_path: Path to model weights file

    Returns:
        LayoutGraph with predicted room positions
    """
    from app.core.model_registry import ModelRegistry
    device = ModelRegistry().device

    # Load model
    model = FloorPlanGNN()

    if os.path.exists(model_path):
        try:
            state_dict = torch.load(model_path, map_location=device, weights_only=True)
        except TypeError:
            # Compatibility with older torch versions that do not support weights_only.
            state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        logger.info(f"Loaded GNN weights from {model_path}")
    else:
        logger.warning(
            f"Model weights not found at {model_path} — using random initialisation"
        )

    model = model.to(device)
    model.eval()

    # Convert to PyG data
    if HAS_PYG:
        data = parsed_layout_to_pyg(parsed_layout)
        data = data.to(device)

        with torch.inference_mode():
            bbox_pred, door_pred = model(data.x, data.edge_index, data.edge_attr)
    else:
        # Fallback without PyG
        N = len(parsed_layout.rooms)
        x = torch.stack([build_node_features(room) for room in parsed_layout.rooms]).to(
            device
        )
        edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)

        with torch.inference_mode():
            bbox_pred, door_pred = model(x, edge_index, None)

    # Post-process bounding boxes
    bbox_pred = bbox_pred.cpu()
    bbox_pred = torch.clamp(bbox_pred, 0.0, 1.0)

    # Apply room type position preferences (bias towards typical floor plan locations)
    position_bias = 0.15  # How much to bias towards preferred positions
    for i in range(bbox_pred.size(0)):
        room_type = parsed_layout.rooms[i].room_type
        preferred = ROOM_PREFERRED_ZONES.get(room_type)

        if preferred:
            (pref_x_min, pref_x_max), (pref_y_min, pref_y_max) = preferred
            pref_cx = (pref_x_min + pref_x_max) / 2
            pref_cy = (pref_y_min + pref_y_max) / 2

            current_cx = (bbox_pred[i, 0] + bbox_pred[i, 2]) / 2
            current_cy = (bbox_pred[i, 1] + bbox_pred[i, 3]) / 2

            # Blend current position with preferred position
            new_cx = current_cx * (1 - position_bias) + pref_cx * position_bias
            new_cy = current_cy * (1 - position_bias) + pref_cy * position_bias

            w = bbox_pred[i, 2] - bbox_pred[i, 0]
            h = bbox_pred[i, 3] - bbox_pred[i, 1]

            bbox_pred[i, 0] = max(0.0, new_cx - w / 2)
            bbox_pred[i, 1] = max(0.0, new_cy - h / 2)
            bbox_pred[i, 2] = min(1.0, new_cx + w / 2)
            bbox_pred[i, 3] = min(1.0, new_cy + h / 2)

    # Ensure x_min < x_max, y_min < y_max
    for i in range(bbox_pred.size(0)):
        if bbox_pred[i, 0] > bbox_pred[i, 2]:
            bbox_pred[i, 0], bbox_pred[i, 2] = (
                bbox_pred[i, 2].item(),
                bbox_pred[i, 0].item(),
            )
        if bbox_pred[i, 1] > bbox_pred[i, 3]:
            bbox_pred[i, 1], bbox_pred[i, 3] = (
                bbox_pred[i, 3].item(),
                bbox_pred[i, 1].item(),
            )

    # Ensure minimum size (at least 0.05 in normalised coords for a 10m plot = 0.5m)
    min_size = 0.05
    for i in range(bbox_pred.size(0)):
        room_type = parsed_layout.rooms[i].room_type
        room_min = ROOM_MIN_SIZES.get(room_type, (min_size, min_size))

        x_min = float(bbox_pred[i, 0])
        y_min = float(bbox_pred[i, 1])
        x_max = float(bbox_pred[i, 2])
        y_max = float(bbox_pred[i, 3])

        room_min_w, room_min_h = room_min
        current_w = x_max - x_min
        current_h = y_max - y_min

        # Apply room-type specific minimum sizes
        if current_w < room_min_w:
            cx = (x_min + x_max) / 2.0
            x_min = max(0.0, cx - room_min_w / 2.0)
            x_max = min(1.0, cx + room_min_w / 2.0)
            if x_max - x_min < room_min_w:
                x_min = max(0.0, x_max - room_min_w)
                x_max = min(1.0, x_min + room_min_w)

        if current_h < room_min_h:
            cy = (y_min + y_max) / 2.0
            y_min = max(0.0, cy - room_min_h / 2.0)
            y_max = min(1.0, cy + room_min_h / 2.0)
            if y_max - y_min < room_min_h:
                y_min = max(0.0, y_max - room_min_h)
                y_max = min(1.0, y_min + room_min_h)

        bbox_pred[i, 0] = x_min
        bbox_pred[i, 1] = y_min
        bbox_pred[i, 2] = x_max
        bbox_pred[i, 3] = y_max

    # Apply aspect ratio constraints for more realistic room shapes
    for i in range(bbox_pred.size(0)):
        room_type = parsed_layout.rooms[i].room_type
        aspect_range = ROOM_ASPECT_RATIOS.get(room_type, (0.7, 1.5))

        x_min = float(bbox_pred[i, 0])
        y_min = float(bbox_pred[i, 1])
        x_max = float(bbox_pred[i, 2])
        y_max = float(bbox_pred[i, 3])

        current_w = x_max - x_min
        current_h = y_max - y_min
        current_aspect = current_w / current_h if current_h > 0 else 1.0

        min_aspect, max_aspect = aspect_range

        # Adjust aspect ratio to be within acceptable range
        if current_aspect < min_aspect:
            # Room is too tall - make it wider
            new_w = current_h * min_aspect
            cx = (x_min + x_max) / 2.0
            x_min = max(0.0, cx - new_w / 2.0)
            x_max = min(1.0, cx + new_w / 2.0)
        elif current_aspect > max_aspect:
            # Room is too wide - make it taller
            new_h = current_w / max_aspect
            cy = (y_min + y_max) / 2.0
            y_min = max(0.0, cy - new_h / 2.0)
            y_max = min(1.0, cy + new_h / 2.0)

        bbox_pred[i, 0] = x_min
        bbox_pred[i, 1] = y_min
        bbox_pred[i, 2] = x_max
        bbox_pred[i, 3] = y_max

    # Apply room spacing for realistic wall thickness (shrink rooms slightly)
    # This creates gaps between rooms representing wall thickness
    wall_gap = 0.015  # 1.5% of plot size for walls
    for i in range(bbox_pred.size(0)):
        x_min = float(bbox_pred[i, 0])
        y_min = float(bbox_pred[i, 1])
        x_max = float(bbox_pred[i, 2])
        y_max = float(bbox_pred[i, 3])

        x_min = min(x_min + wall_gap, x_max - wall_gap)
        y_min = min(y_min + wall_gap, y_max - wall_gap)
        x_max = max(x_max - wall_gap, x_min + wall_gap)
        y_max = max(y_max - wall_gap, y_min + wall_gap)

        # Clamp to [0, 1] to ensure valid bounding boxes
        x_min = max(0.0, min(1.0, x_min))
        y_min = max(0.0, min(1.0, y_min))
        x_max = max(0.0, min(1.0, x_max))
        y_max = max(0.0, min(1.0, y_max))

        # Ensure min size after clamping
        if x_max - x_min < 0.03:
            cx = (x_min + x_max) / 2
            x_min = max(0.0, cx - 0.015)
            x_max = min(1.0, cx + 0.015)
        if y_max - y_min < 0.03:
            cy = (y_min + y_max) / 2
            y_min = max(0.0, cy - 0.015)
            y_max = min(1.0, cy + 0.015)

        bbox_pred[i, 0] = x_min
        bbox_pred[i, 1] = y_min
        bbox_pred[i, 2] = x_max
        bbox_pred[i, 3] = y_max

    # Build LayoutGraph
    rooms = []
    for i, room_spec in enumerate(parsed_layout.rooms):
        bbox = BoundingBox(
            x_min=float(bbox_pred[i, 0]),
            y_min=float(bbox_pred[i, 1]),
            x_max=float(bbox_pred[i, 2]),
            y_max=float(bbox_pred[i, 3]),
        )

        # Door midpoints from edge predictions
        door_midpoints = []
        if door_pred is not None and HAS_PYG:
            door_cpu = door_pred.cpu()
            edge_index = data.edge_index.cpu()
            for j in range(edge_index.size(1)):
                if edge_index[0, j] == i:
                    door_midpoints.append(
                        (float(door_cpu[j, 0]), float(door_cpu[j, 1]))
                    )

        rooms.append(
            RoomLayout(
                room_spec=room_spec,
                bbox=bbox,
                door_midpoints=door_midpoints,
            )
        )

    return LayoutGraph(
        rooms=rooms,
        adjacency_edges=parsed_layout.adjacency_constraints,
        plot_area_sqm=parsed_layout.plot_area_sqm,
        facing=parsed_layout.facing,
        vastu_enabled=parsed_layout.vastu_enabled,
        generation_mode="gnn",
    )


# ─── Training Loss Functions ────────────────────────────────────────────────


def compute_loss(
    pred_bboxes: torch.Tensor,
    gt_bboxes: torch.Tensor,
    pred_edges: torch.Tensor | None = None,
    gt_adjacency: torch.Tensor | None = None,
    batch_index: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Compute training loss for the GNN model.

    Loss = 1.0×GIoU + 2.0×Overlap + 0.5×AdjBCE

    Args:
        pred_bboxes: Predicted bounding boxes [N, 4]
        gt_bboxes: Ground truth bounding boxes [N, 4]
        pred_edges: Predicted edge features [E, 2] (optional)
        gt_adjacency: Ground truth adjacency [E] (optional)
        batch_index: Graph assignment for each node [N] (optional)

    Returns:
        Dict with keys: total, giou, overlap, adj_bce
    """
    # Generalised IoU Loss
    giou_loss = _giou_loss(pred_bboxes, gt_bboxes)

    # Overlap Loss: penalise pairwise overlaps
    overlap_loss = _overlap_loss(pred_bboxes, batch_index)

    # Adjacency BCE Loss
    if pred_edges is not None and gt_adjacency is not None:
        adj_bce = F.binary_cross_entropy(pred_edges[:, 0], gt_adjacency.float())
    else:
        adj_bce = torch.tensor(0.0, device=pred_bboxes.device)

    total = 1.0 * giou_loss + 2.0 * overlap_loss + 0.5 * adj_bce

    return {
        "total": total,
        "giou": giou_loss,
        "overlap": overlap_loss,
        "adj_bce": adj_bce,
    }


def _giou_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Compute Generalised IoU loss between predicted and ground truth bboxes."""
    pred_x1, pred_y1 = pred[:, 0], pred[:, 1]
    pred_x2, pred_y2 = pred[:, 2], pred[:, 3]
    gt_x1, gt_y1 = gt[:, 0], gt[:, 1]
    gt_x2, gt_y2 = gt[:, 2], gt[:, 3]

    # Intersection
    inter_x1 = torch.max(pred_x1, gt_x1)
    inter_y1 = torch.max(pred_y1, gt_y1)
    inter_x2 = torch.min(pred_x2, gt_x2)
    inter_y2 = torch.min(pred_y2, gt_y2)
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(
        inter_y2 - inter_y1, min=0
    )

    # Union
    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
    union_area = pred_area + gt_area - inter_area + 1e-8

    # IoU
    iou = inter_area / union_area

    # Enclosing box
    enc_x1 = torch.min(pred_x1, gt_x1)
    enc_y1 = torch.min(pred_y1, gt_y1)
    enc_x2 = torch.max(pred_x2, gt_x2)
    enc_y2 = torch.max(pred_y2, gt_y2)
    enc_area = (enc_x2 - enc_x1) * (enc_y2 - enc_y1) + 1e-8

    # GIoU
    giou = iou - (enc_area - union_area) / enc_area

    return (1 - giou).mean()


def _overlap_loss(
    bboxes: torch.Tensor, batch_index: torch.Tensor | None = None
) -> torch.Tensor:
    """Compute pairwise overlap loss between all room pairs within the same graph."""
    N = bboxes.size(0)
    if N < 2:
        return torch.tensor(0.0, device=bboxes.device)

    # Broadcast: [N, 1, 4] vs [1, N, 4]
    b1 = bboxes.unsqueeze(1)  # [N, 1, 4]
    b2 = bboxes.unsqueeze(0)  # [1, N, 4]

    # Intersection
    inter_x1 = torch.max(b1[:, :, 0], b2[:, :, 0])
    inter_y1 = torch.max(b1[:, :, 1], b2[:, :, 1])
    inter_x2 = torch.min(b1[:, :, 2], b2[:, :, 2])
    inter_y2 = torch.min(b1[:, :, 3], b2[:, :, 3])
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(
        inter_y2 - inter_y1, min=0
    )

    # Sum upper triangle only (avoid double counting + self overlap)
    mask = torch.triu(
        torch.ones(N, N, device=bboxes.device, dtype=torch.bool), diagonal=1
    )

    if batch_index is not None:
        # Only penalize overlap for rooms in the SAME graph batch
        batch_mask = batch_index.unsqueeze(1) == batch_index.unsqueeze(0)
        mask = mask & batch_mask

    return inter_area[mask].sum()
