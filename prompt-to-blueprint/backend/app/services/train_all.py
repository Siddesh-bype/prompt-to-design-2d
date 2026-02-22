"""
Master Training Script — Train ALL models from a single entry point.

Orchestrates training across all model components:
1. GNN Layout Model      — FloorPlanGNN graph neural network
2. NLP Parser            — Ollama SLM fine-tuning data + Modelfile
3. Constraint Weights    — Soft constraint weight optimisation
4. Vastu Scoring Rules   — Vastu rule calibration from real data

Usage:
    # Train everything on CubiCasa5K
    python -m app.services.train_all --dataset cubicasa --data-dir data/cubicasa5k

    # Train everything on combined datasets
    python -m app.services.train_all --dataset combined --data-dir data

    # Train specific models
    python -m app.services.train_all --dataset combined --models gnn nlp constraints
"""

from __future__ import annotations

import logging
import os
import time

logger = logging.getLogger(__name__)


def train_all(
    dataset: str = "combined",
    data_dir: str = "data",
    output_dir: str = "models",
    epochs: int = 50,
    lr: float = 0.001,
    synthetic_samples: int = 1000,
    max_samples: int | None = None,
    models: list[str] | None = None,
) -> dict:
    """Train all models using the specified dataset(s).

    Args:
        dataset: Data source — 'synthetic', 'cubicasa', 'swiss', 'combined'
        data_dir: Root data directory
        output_dir: Model output directory
        epochs: GNN training epochs
        lr: GNN learning rate
        synthetic_samples: Synthetic sample count
        max_samples: Max samples per source
        models: List of models to train (default: all)
                Options: 'gnn', 'nlp', 'constraints', 'vastu'

    Returns:
        Dict of training results per model
    """
    if models is None:
        models = ["gnn", "nlp", "constraints", "vastu"]

    results = {}
    total_start = time.time()

    logger.info("=" * 60)
    logger.info(f"  TRAINING ALL MODELS")
    logger.info(f"  Dataset: {dataset}")
    logger.info(f"  Data dir: {data_dir}")
    logger.info(f"  Models: {', '.join(models)}")
    logger.info("=" * 60)

    # Load dataset once (shared across all models)
    logger.info("\n[1/5] Loading dataset...")
    from app.services.train_gnn import load_dataset_by_source
    try:
        samples = load_dataset_by_source(dataset, data_dir, synthetic_samples, max_samples)
        logger.info(f"  Loaded {len(samples)} samples")
    except Exception as e:
        logger.error(f"  Failed to load dataset: {e}")
        return {"error": str(e)}

    os.makedirs(output_dir, exist_ok=True)

    # ── Model 1: GNN Layout Engine ───────────────────────────
    if "gnn" in models:
        logger.info("\n[2/5] Training GNN Layout Model...")
        t0 = time.time()
        try:
            from app.services.train_gnn import train as train_gnn
            train_gnn(
                dataset=dataset,
                data_dir=data_dir,
                output_dir=output_dir,
                epochs=epochs,
                lr=lr,
                synthetic_samples=synthetic_samples,
                max_samples=max_samples,
            )
            elapsed = time.time() - t0
            results["gnn"] = {
                "status": "success",
                "time_seconds": round(elapsed, 1),
                "output": os.path.join(output_dir, "floorplan_gnn.pt"),
            }
            logger.info(f"  ✓ GNN training complete ({elapsed:.1f}s)")
        except Exception as e:
            results["gnn"] = {"status": "error", "error": str(e)}
            logger.error(f"  ✗ GNN training failed: {e}")
    else:
        logger.info("\n[2/5] Skipping GNN (not selected)")

    # ── Model 2: NLP Parser Fine-tuning ──────────────────────
    if "nlp" in models:
        logger.info("\n[3/5] Generating NLP Fine-tuning Data...")
        t0 = time.time()
        try:
            from app.services.train_nlp import (
                generate_nlp_training_data,
                generate_ollama_modelfile,
            )
            nlp_output = os.path.join(data_dir, "training", "nlp_finetune.jsonl")
            generate_nlp_training_data(samples, nlp_output, augmentation_factor=3)
            modelfile = generate_ollama_modelfile(training_data_path=nlp_output,
                                                   output_path=os.path.join(data_dir, "training", "Modelfile"))
            elapsed = time.time() - t0
            results["nlp"] = {
                "status": "success",
                "time_seconds": round(elapsed, 1),
                "output_data": nlp_output,
                "output_modelfile": modelfile,
            }
            logger.info(f"  ✓ NLP data generated ({elapsed:.1f}s)")
        except Exception as e:
            results["nlp"] = {"status": "error", "error": str(e)}
            logger.error(f"  ✗ NLP data generation failed: {e}")
    else:
        logger.info("\n[3/5] Skipping NLP (not selected)")

    # ── Model 3: Constraint Weight Optimisation ──────────────
    if "constraints" in models:
        logger.info("\n[4/5] Optimising Constraint Weights...")
        t0 = time.time()
        try:
            from app.services.train_constraints import optimise_constraint_weights
            weights_path = os.path.join(output_dir, "constraint_weights.json")
            weights = optimise_constraint_weights(
                samples, n_iterations=100, output_path=weights_path
            )
            elapsed = time.time() - t0
            results["constraints"] = {
                "status": "success",
                "time_seconds": round(elapsed, 1),
                "output": weights_path,
                "weights": {k: round(v, 3) for k, v in weights.items()},
            }
            logger.info(f"  ✓ Constraint weights optimised ({elapsed:.1f}s)")
        except Exception as e:
            results["constraints"] = {"status": "error", "error": str(e)}
            logger.error(f"  ✗ Constraint optimisation failed: {e}")
    else:
        logger.info("\n[4/5] Skipping Constraints (not selected)")

    # ── Model 4: Vastu Rule Calibration ──────────────────────
    if "vastu" in models:
        logger.info("\n[5/5] Calibrating Vastu Scoring Rules...")
        t0 = time.time()
        try:
            vastu_stats = _calibrate_vastu(samples, output_dir)
            elapsed = time.time() - t0
            results["vastu"] = {
                "status": "success",
                "time_seconds": round(elapsed, 1),
                "stats": vastu_stats,
            }
            logger.info(f"  ✓ Vastu calibration complete ({elapsed:.1f}s)")
        except Exception as e:
            results["vastu"] = {"status": "error", "error": str(e)}
            logger.error(f"  ✗ Vastu calibration failed: {e}")
    else:
        logger.info("\n[5/5] Skipping Vastu (not selected)")

    # ── Summary ──────────────────────────────────────────────
    total_elapsed = time.time() - total_start
    logger.info("\n" + "=" * 60)
    logger.info(f"  TRAINING COMPLETE — {total_elapsed:.1f}s total")
    for model_name, result in results.items():
        status = result.get("status", "unknown")
        time_s = result.get("time_seconds", 0)
        icon = "✓" if status == "success" else "✗"
        logger.info(f"  {icon} {model_name}: {status} ({time_s}s)")
    logger.info("=" * 60)

    return results


def _calibrate_vastu(samples: list[dict], output_dir: str) -> dict:
    """Calibrate Vastu scoring by analysing room placement patterns in real data."""
    import json
    from collections import Counter
    from app.models.schemas import LayoutGraph, RoomType

    zone_room_counts = Counter()
    total = 0

    for sample in samples:
        try:
            layout = LayoutGraph.model_validate(sample["layout_graph"])
            for room in layout.rooms:
                if not room.room_spec:
                    continue

                # Determine zone from room centre
                cx = (room.bbox.x_min + room.bbox.x_max) / 2
                cy = (room.bbox.y_min + room.bbox.y_max) / 2

                if cy < 0.33:
                    row = "N"
                elif cy < 0.67:
                    row = ""
                else:
                    row = "S"

                if cx < 0.33:
                    col = "W"
                elif cx < 0.67:
                    col = "CENTER" if not row else ""
                else:
                    col = "E"

                zone = (row + col) or "CENTER"
                rt = room.room_spec.room_type.value
                zone_room_counts[(zone, rt)] += 1
                total += 1
        except Exception:
            continue

    # Save zone frequency analysis
    stats = {
        "total_rooms_analysed": total,
        "zone_frequencies": {
            f"{zone}_{rt}": count
            for (zone, rt), count in zone_room_counts.most_common(50)
        },
    }

    stats_path = os.path.join(output_dir, "vastu_calibration.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"  Analysed {total} rooms across {len(samples)} layouts")
    return stats


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Train ALL Prompt-to-Blueprint models")
    parser.add_argument(
        "--dataset", type=str, default="combined",
        choices=["synthetic", "cubicasa", "swiss", "combined"],
    )
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output", type=str, default="models")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--max-per-source", type=int, default=None)
    parser.add_argument(
        "--models", nargs="+", default=None,
        choices=["gnn", "nlp", "constraints", "vastu"],
        help="Specific models to train (default: all)",
    )
    args = parser.parse_args()

    train_all(
        dataset=args.dataset,
        data_dir=args.data_dir,
        output_dir=args.output,
        epochs=args.epochs,
        lr=args.lr,
        synthetic_samples=args.samples,
        max_samples=args.max_per_source,
        models=args.models,
    )
