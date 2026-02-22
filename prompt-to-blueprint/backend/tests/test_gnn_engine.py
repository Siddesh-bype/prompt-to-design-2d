"""Unit tests for the GNN layout engine."""

import pytest
import torch

from app.models.schemas import (
    RoomSpec, RoomType, CompassFacing, ConnectionType,
    AdjacencyEdge, ParsedLayout,
)
from app.services.gnn_engine import (
    FloorPlanGNN,
    build_node_features,
    build_edge_features,
    compute_loss,
    run_gnn_inference,
    NODE_FEAT_DIM,
    EDGE_FEAT_DIM,
    HAS_PYG,
)


def _make_test_layout() -> ParsedLayout:
    """Create a simple 2-room test layout."""
    return ParsedLayout(
        rooms=[
            RoomSpec(room_id="room_1", room_type=RoomType.LIVING_ROOM, target_area_sqm=25.0),
            RoomSpec(room_id="room_2", room_type=RoomType.KITCHEN, target_area_sqm=10.0),
            RoomSpec(room_id="room_3", room_type=RoomType.BEDROOM, target_area_sqm=15.0),
        ],
        plot_area_sqm=100.0,
        facing=CompassFacing.NORTH,
        adjacency_constraints=[
            AdjacencyEdge(
                room_a_id="room_1", room_b_id="room_2",
                connection_type=ConnectionType.OPENING, required=True,
            ),
        ],
    )


def test_node_feature_shape():
    """Assert output feature vector shape == (18,)."""
    room = RoomSpec(
        room_id="r1",
        room_type=RoomType.LIVING_ROOM,
        target_area_sqm=25.0,
        compass_preference=CompassFacing.EAST,
    )
    features = build_node_features(room, adjacency_count=3)
    assert features.shape == (NODE_FEAT_DIM,)
    assert features.shape == (18,)

    # Check one-hot encoding
    assert features[0] == 1.0  # LIVING_ROOM is index 0
    assert features[12] == 25.0 / 80.0  # normalised area
    assert features[13] == 3.0 / 6.0  # adjacency rank
    assert features[16] == 1.0  # EAST compass preference


def test_edge_feature_shape():
    """Test edge feature vector shape and values."""
    edge = AdjacencyEdge(
        room_a_id="r1", room_b_id="r2",
        connection_type=ConnectionType.DOOR, required=True,
    )
    features = build_edge_features(edge)
    assert features.shape == (EDGE_FEAT_DIM,)
    assert features[0] == 1.0  # DOOR
    assert features[1] == 0.0  # Not OPENING
    assert features[2] == 1.0  # required


def test_model_forward_pass():
    """Test forward pass with random data produces correct output shapes."""
    model = FloorPlanGNN()
    N, E = 5, 8
    x = torch.randn(N, NODE_FEAT_DIM)
    edge_index = torch.randint(0, N, (2, E))
    edge_attr = torch.randn(E, EDGE_FEAT_DIM)

    model.eval()
    with torch.no_grad():
        bbox, doors = model(x, edge_index, edge_attr)

    assert bbox.shape == (N, 4)
    # All values should be in [0, 1] (sigmoid)
    assert (bbox >= 0).all() and (bbox <= 1).all()


def test_bbox_clamping():
    """Verify no coordinate outside [0,1] after inference post-processing."""
    layout = _make_test_layout()
    result = run_gnn_inference(layout, model_path="nonexistent_model.pt")

    for room in result.rooms:
        assert 0.0 <= room.bbox.x_min <= 1.0
        assert 0.0 <= room.bbox.y_min <= 1.0
        assert 0.0 <= room.bbox.x_max <= 1.0
        assert 0.0 <= room.bbox.y_max <= 1.0
        assert room.bbox.x_min < room.bbox.x_max
        assert room.bbox.y_min < room.bbox.y_max


def test_loss_computation():
    """Assert all loss terms are positive scalars."""
    N = 4
    pred_bboxes = torch.rand(N, 4)
    # Ensure valid bboxes (min < max)
    pred_bboxes[:, 2] = pred_bboxes[:, 0] + torch.rand(N) * 0.3 + 0.1
    pred_bboxes[:, 3] = pred_bboxes[:, 1] + torch.rand(N) * 0.3 + 0.1
    pred_bboxes = torch.clamp(pred_bboxes, 0, 1)

    gt_bboxes = torch.rand(N, 4)
    gt_bboxes[:, 2] = gt_bboxes[:, 0] + torch.rand(N) * 0.3 + 0.1
    gt_bboxes[:, 3] = gt_bboxes[:, 1] + torch.rand(N) * 0.3 + 0.1
    gt_bboxes = torch.clamp(gt_bboxes, 0, 1)

    losses = compute_loss(pred_bboxes, gt_bboxes)

    assert losses["total"].dim() == 0  # scalar
    assert losses["giou"].dim() == 0
    assert losses["overlap"].dim() == 0
    assert losses["adj_bce"].dim() == 0
    assert losses["total"].item() >= 0
