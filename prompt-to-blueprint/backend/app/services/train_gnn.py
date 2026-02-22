"""
GNN Training Script — Train the FloorPlanGNN model on real + synthetic data.

Usage:
    # Synthetic data only
    python -m app.services.train_gnn --dataset synthetic --samples 2000 --epochs 50

    # CubiCasa5K
    python -m app.services.train_gnn --dataset cubicasa --data-dir data/cubicasa5k --epochs 100

    # Modified Swiss Dwellings
    python -m app.services.train_gnn --dataset swiss --data-dir data/modified_swiss_dwellings --epochs 100

    # Combined (all sources)
    python -m app.services.train_gnn --dataset combined --data-dir data --epochs 150

    # Resume from checkpoint
    python -m app.services.train_gnn --dataset cubicasa --resume models/checkpoint_epoch_50.pt
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from app.models.schemas import (
    BoundingBox,
    LayoutGraph,
    ParsedLayout,
    RoomLayout,
)
from app.services.gnn_engine import (
    FloorPlanGNN,
    build_node_features,
    build_edge_features,
    compute_loss,
    parsed_layout_to_pyg,
    HAS_PYG,
)

logger = logging.getLogger(__name__)


# ─── Dataset Loading ─────────────────────────────────────────────────────────


def load_jsonl_dataset(data_path: str) -> list[dict]:
    """Load a JSONL dataset file."""
    samples = []
    with open(data_path, "r") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    logger.info(f"Loaded {len(samples)} samples from {data_path}")
    return samples


def load_dataset_by_source(
    dataset: str,
    data_dir: str = "data",
    synthetic_samples: int = 1000,
    max_samples: int | None = None,
) -> list[dict]:
    """Load training data from the specified source(s).

    Args:
        dataset: One of 'synthetic', 'cubicasa', 'swiss', 'combined'
        data_dir: Root data directory
        synthetic_samples: Number of synthetic samples (for 'synthetic'/'combined')
        max_samples: Max per-source limit

    Returns:
        List of sample dicts with parsed_layout and layout_graph keys
    """
    all_samples = []

    # ── Synthetic ────────────────────────────────────────────
    if dataset in ("synthetic", "combined"):
        jsonl_path = os.path.join(data_dir, "training", "floor_plans.jsonl")

        if not os.path.exists(jsonl_path):
            logger.info(f"Generating {synthetic_samples} synthetic samples...")
            from app.services.dataset_generator import generate_dataset
            generate_dataset(
                n_samples=synthetic_samples,
                output_dir=os.path.join(data_dir, "training"),
            )

        synthetic_data = load_jsonl_dataset(jsonl_path)
        if max_samples:
            synthetic_data = synthetic_data[:max_samples]
        logger.info(f"Synthetic: {len(synthetic_data)} samples")
        all_samples.extend(synthetic_data)

    # ── CubiCasa5K ───────────────────────────────────────────
    if dataset in ("cubicasa", "combined"):
        try:
            from app.services.cubicasa_adapter import load_cubicasa_dataset

            cubicasa_dir = data_dir
            if dataset == "combined":
                cubicasa_dir = os.path.join(data_dir, "cubicasa5k")

            # Check multiple possible paths
            for possible_dir in [cubicasa_dir, os.path.join(data_dir, "cubicasa5k"),
                                  os.path.join(data_dir, "CubiCasa5k")]:
                if os.path.exists(possible_dir):
                    cubicasa_dir = possible_dir
                    break

            if os.path.exists(cubicasa_dir):
                # Load train + val splits
                train_data = load_cubicasa_dataset(cubicasa_dir, "train", max_samples)
                val_data = load_cubicasa_dataset(cubicasa_dir, "val", max_samples)
                cubicasa_data = train_data + val_data

                logger.info(f"CubiCasa5K: {len(cubicasa_data)} samples (train={len(train_data)}, val={len(val_data)})")
                all_samples.extend(cubicasa_data)
            else:
                logger.warning(f"CubiCasa5K data not found at {cubicasa_dir}")
                logger.info("Download from: https://www.kaggle.com/datasets/qmarva/cubicasa5k")
        except ImportError as e:
            logger.warning(f"CubiCasa adapter not available: {e}")

    # ── Modified Swiss Dwellings ─────────────────────────────
    if dataset in ("swiss", "combined"):
        try:
            from app.services.swiss_dwellings_adapter import load_swiss_dwellings_dataset

            swiss_dir = data_dir
            if dataset == "combined":
                swiss_dir = os.path.join(data_dir, "modified_swiss_dwellings")

            # Check multiple possible paths
            for possible_dir in [swiss_dir, os.path.join(data_dir, "modified_swiss_dwellings"),
                                  os.path.join(data_dir, "modified-swiss-dwellings")]:
                if os.path.exists(possible_dir):
                    swiss_dir = possible_dir
                    break

            if os.path.exists(swiss_dir):
                swiss_data = load_swiss_dwellings_dataset(swiss_dir, max_samples)
                logger.info(f"Swiss Dwellings: {len(swiss_data)} samples")
                all_samples.extend(swiss_data)
            else:
                logger.warning(f"Swiss Dwellings data not found at {swiss_dir}")
                logger.info("Download from: https://www.kaggle.com/datasets/caspervanengelenburg/modified-swiss-dwellings")
        except ImportError as e:
            logger.warning(f"Swiss Dwellings adapter not available: {e}")

    if not all_samples:
        raise ValueError(
            f"No training data found for dataset='{dataset}'. "
            f"Check that data exists at: {data_dir}"
        )

    # Shuffle combined data
    random.shuffle(all_samples)

    logger.info(f"Total dataset: {len(all_samples)} samples")
    return all_samples


# ─── Training Loop ───────────────────────────────────────────────────────────


def train(
    dataset: str = "synthetic",
    data_dir: str = "data",
    output_dir: str = "models",
    epochs: int = 50,
    lr: float = 0.001,
    batch_size: int = 1,
    synthetic_samples: int = 1000,
    max_samples: int | None = None,
    resume_from: str | None = None,
) -> None:
    """Train the FloorPlanGNN model.

    Args:
        dataset: Data source — 'synthetic', 'cubicasa', 'swiss', or 'combined'
        data_dir: Root data directory
        output_dir: Directory to save model weights
        epochs: Number of training epochs
        lr: Learning rate
        batch_size: Batch size (1 for graph-level training)
        synthetic_samples: Number of synthetic samples if using synthetic
        max_samples: Optional per-source limit
        resume_from: Path to checkpoint to resume from
    """
    if not HAS_PYG:
        raise ImportError("torch_geometric is required for training")

    from torch_geometric.data import Data

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on device: {device}")
    logger.info(f"Dataset source: {dataset}")

    # Load data
    samples = load_dataset_by_source(dataset, data_dir, synthetic_samples, max_samples)

    # Split 80/20
    split_idx = int(len(samples) * 0.8)
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]
    logger.info(f"Train: {len(train_samples)}, Val: {len(val_samples)}")

    # Convert to PyG data list
    train_data = _samples_to_pyg(train_samples, device)
    val_data = _samples_to_pyg(val_samples, device)

    if not train_data:
        raise ValueError("No valid training samples after conversion to PyG format")

    logger.info(f"PyG conversion: train={len(train_data)}, val={len(val_data)}")

    # Model
    model = FloorPlanGNN().to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    start_epoch = 0

    if resume_from and os.path.exists(resume_from):
        checkpoint = torch.load(resume_from, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        logger.info(f"Resumed from epoch {start_epoch}")

    os.makedirs(output_dir, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(start_epoch, epochs):
        t0 = time.time()

        # Training
        model.train()
        train_losses = []

        for data, gt_bboxes in train_data:
            optimizer.zero_grad()
            bbox_pred, door_pred = model(data.x, data.edge_index, data.edge_attr)

            losses = compute_loss(bbox_pred, gt_bboxes)
            losses["total"].backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(losses["total"].item())

        avg_train = sum(train_losses) / max(len(train_losses), 1)

        # Validation
        model.eval()
        val_losses = []

        with torch.no_grad():
            for data, gt_bboxes in val_data:
                bbox_pred, _ = model(data.x, data.edge_index, data.edge_attr)
                losses = compute_loss(bbox_pred, gt_bboxes)
                val_losses.append(losses["total"].item())

        avg_val = sum(val_losses) / max(len(val_losses), 1)

        scheduler.step()
        elapsed = time.time() - t0

        # Log
        logger.info(
            f"Epoch {epoch+1}/{epochs} — "
            f"train: {avg_train:.4f}, val: {avg_val:.4f}, "
            f"lr: {scheduler.get_last_lr()[0]:.6f}, "
            f"time: {elapsed:.1f}s"
        )

        # Save best
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, "floorplan_gnn.pt"),
            )
            logger.info(f"  → Best model saved (val_loss: {avg_val:.4f})")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": avg_val,
            }, os.path.join(output_dir, f"checkpoint_epoch_{epoch+1}.pt"))

    logger.info(f"Training complete. Best val_loss: {best_val_loss:.4f}")


def _samples_to_pyg(
    samples: list[dict], device: torch.device
) -> list[tuple]:
    """Convert samples to PyG Data + ground truth bboxes."""
    from torch_geometric.data import Data

    result = []

    for sample in samples:
        try:
            parsed = ParsedLayout.model_validate(sample["parsed_layout"])
            gt_layout = LayoutGraph.model_validate(sample["layout_graph"])

            data = parsed_layout_to_pyg(parsed).to(device)

            # Extract GT bounding boxes
            gt_bboxes = torch.tensor([
                [r.bbox.x_min, r.bbox.y_min, r.bbox.x_max, r.bbox.y_max]
                for r in gt_layout.rooms
            ], dtype=torch.float32, device=device)

            if data.x.size(0) == gt_bboxes.size(0):
                result.append((data, gt_bboxes))
        except Exception as e:
            logger.warning(f"Skipping invalid sample: {e}")
            continue

    return result


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Train FloorPlanGNN")
    parser.add_argument(
        "--dataset", type=str, default="synthetic",
        choices=["synthetic", "cubicasa", "swiss", "combined"],
        help="Data source: synthetic, cubicasa, swiss, or combined",
    )
    parser.add_argument("--data-dir", type=str, default="data", help="Root data directory")
    parser.add_argument("--output", type=str, default="models", help="Model output directory")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--samples", type=int, default=1000, help="Synthetic sample count")
    parser.add_argument("--max-per-source", type=int, default=None, help="Max samples per source")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    args = parser.parse_args()

    train(
        dataset=args.dataset,
        data_dir=args.data_dir,
        output_dir=args.output,
        epochs=args.epochs,
        lr=args.lr,
        synthetic_samples=args.samples,
        max_samples=args.max_per_source,
        resume_from=args.resume,
    )
