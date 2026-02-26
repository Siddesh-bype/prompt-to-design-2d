"""
Constraint Model Trainer — Learns optimal constraint weights from real layouts.

Uses real floor plan data to optimise the soft constraint weights
(natural light priority, corridor area penalty, kitchen-bedroom distance, etc.)
in the constraint solver's SciPy L-BFGS-B objective function.

This is essentially a hyperparameter optimisation pipeline using
real floor plan layouts as ground truth.
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
from typing import Optional

import numpy as np

from app.models.schemas import LayoutGraph, ParsedLayout, RoomType

logger = logging.getLogger(__name__)


# Default constraint weights (initial values)
DEFAULT_WEIGHTS = {
    "natural_light_priority": 1.0,
    "corridor_area_penalty": 0.5,
    "kitchen_bedroom_distance": 0.8,
    "overlap_penalty": 10.0,
    "adjacency_reward": 2.0,
    "aspect_ratio_penalty": 0.3,
    "area_fidelity": 1.5,
}


def evaluate_layout_quality(layout: LayoutGraph) -> dict[str, float]:
    """Evaluate a ground-truth layout's quality metrics.

    Args:
        layout: Ground-truth LayoutGraph

    Returns:
        Dict of metric scores (higher = better)
    """
    rooms = layout.rooms
    if not rooms:
        return {}

    metrics = {}

    # 1. Overlap rate
    from app.services.renderer import calculate_overlap_rate
    metrics["overlap_rate"] = calculate_overlap_rate(rooms)

    # 2. Adjacency satisfaction
    from app.services.renderer import calculate_adjacency_satisfaction
    metrics["adjacency_satisfaction"] = calculate_adjacency_satisfaction(
        rooms, layout.adjacency_edges
    )

    # 3. Area fidelity — how close actual areas are to targets
    area_errors = []
    for room in rooms:
        if room.room_spec and room.room_spec.target_area_sqm > 0:
            actual = room.bbox.area_sqm(layout.plot_area_sqm)
            target = room.room_spec.target_area_sqm
            area_errors.append(abs(actual - target) / target)
    metrics["area_fidelity"] = 1.0 - (sum(area_errors) / max(len(area_errors), 1))

    # 4. Aspect ratio — penalise very elongated rooms
    aspect_ratios = []
    for room in rooms:
        w = room.bbox.x_max - room.bbox.x_min
        h = room.bbox.y_max - room.bbox.y_min
        if w > 0 and h > 0:
            ratio = max(w, h) / min(w, h)
            aspect_ratios.append(min(ratio, 4.0) / 4.0)
    metrics["aspect_ratio"] = 1.0 - (sum(aspect_ratios) / max(len(aspect_ratios), 1))

    # 5. Kitchen-bedroom distance
    kitchens = [r for r in rooms if r.room_spec and r.room_spec.room_type == RoomType.KITCHEN]
    bedrooms = [r for r in rooms if r.room_spec and r.room_spec.room_type in
                (RoomType.BEDROOM, RoomType.MASTER_BEDROOM)]

    if kitchens and bedrooms:
        distances = []
        for k in kitchens:
            kx = (k.bbox.x_min + k.bbox.x_max) / 2
            ky = (k.bbox.y_min + k.bbox.y_max) / 2
            for b in bedrooms:
                bx = (b.bbox.x_min + b.bbox.x_max) / 2
                by = (b.bbox.y_min + b.bbox.y_max) / 2
                distances.append(math.sqrt((kx - bx) ** 2 + (ky - by) ** 2))
        metrics["kitchen_bedroom_distance"] = min(distances) if distances else 0.0
    else:
        metrics["kitchen_bedroom_distance"] = 0.5

    # 6. Corridor area ratio (lower is better)
    corridor_area = sum(
        (r.bbox.x_max - r.bbox.x_min) * (r.bbox.y_max - r.bbox.y_min)
        for r in rooms if r.room_spec and r.room_spec.room_type == RoomType.CORRIDOR
    )
    total_area = sum(
        (r.bbox.x_max - r.bbox.x_min) * (r.bbox.y_max - r.bbox.y_min)
        for r in rooms
    )
    metrics["corridor_ratio"] = 1.0 - min(corridor_area / max(total_area, 1e-6), 0.3)

    return metrics


def optimise_constraint_weights(
    dataset_samples: list[dict],
    n_iterations: int = 100,
    output_path: str = "models/constraint_weights.json",
) -> dict[str, float]:
    """Optimise constraint weights using real floor plan data.

    Uses a simple evolutionary strategy: randomly perturb weights,
    evaluate against real layouts, keep improvements.

    Args:
        dataset_samples: List of dicts with layout_graph key
        n_iterations: Number of optimisation steps
        output_path: Path to save optimal weights

    Returns:
        Optimised weight dict
    """
    # Evaluate all ground-truth layouts
    gt_metrics = []
    for sample in dataset_samples:
        try:
            layout = LayoutGraph.model_validate(sample["layout_graph"])
            metrics = evaluate_layout_quality(layout)
            if metrics:
                gt_metrics.append(metrics)
        except Exception:
            continue

    if not gt_metrics:
        logger.warning("No valid layouts for weight optimisation")
        return DEFAULT_WEIGHTS

    logger.info(f"Evaluating {len(gt_metrics)} ground-truth layouts for weight tuning")

    # Compute target metric distributions (mean, std)
    all_keys = set()
    for m in gt_metrics:
        all_keys.update(m.keys())

    target_means = {}
    for key in all_keys:
        values = [m[key] for m in gt_metrics if key in m]
        target_means[key] = np.mean(values) if values else 0.5

    logger.info(f"Target metrics: {json.dumps({k: round(v, 3) for k, v in target_means.items()})}")

    # Simple evolutionary optimisation
    best_weights = dict(DEFAULT_WEIGHTS)
    best_score = _evaluate_weights(best_weights, target_means)

    for i in range(n_iterations):
        # Perturb weights
        candidate = {}
        for k, v in best_weights.items():
            candidate[k] = max(0.01, v + random.gauss(0, 0.1 * v))

        score = _evaluate_weights(candidate, target_means)

        if score > best_score:
            best_score = score
            best_weights = candidate
            if (i + 1) % 20 == 0:
                logger.info(f"  Step {i+1}/{n_iterations}: score={best_score:.4f}")

    # Save
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(best_weights, f, indent=2)

    logger.info(f"Optimised weights saved → {output_path}")
    logger.info(f"Final weights: {json.dumps({k: round(v, 3) for k, v in best_weights.items()})}")

    return best_weights


def _evaluate_weights(weights: dict[str, float], targets: dict[str, float]) -> float:
    """Score a weight configuration against target metrics."""
    score = 0.0
    for metric, target_val in targets.items():
        weight_key = metric.replace("_rate", "_penalty").replace("_satisfaction", "_reward")
        w = weights.get(weight_key, 1.0)
        score += w * target_val
    return score


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Optimise constraint weights from real data")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--dataset", type=str, default="combined",
                        choices=["synthetic", "cubicasa", "swiss", "combined"])
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output", type=str, default="models/constraint_weights.json")
    args = parser.parse_args()

    from app.services.train_gnn import load_dataset_by_source
    samples = load_dataset_by_source(args.dataset, args.data_dir)

    optimise_constraint_weights(samples, args.iterations, args.output)
