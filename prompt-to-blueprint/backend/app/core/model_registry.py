"""
Model Registry — manages model lifecycle, loading, and VRAM tracking.

Provides centralised access to the FloorPlanGNN model instance
with optional GPU memory tracking.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Optional

import torch

from app.core.config import settings
from app.services.gnn_engine import FloorPlanGNN

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Thread-safe registry for ML model instances.

    Provides:
    - Lazy model loading on first access
    - VRAM usage tracking
    - Model warm-up (forward pass with dummy data)
    - Thread-safe singleton access
    """

    _instance: Optional["ModelRegistry"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "ModelRegistry":
        """Ensure singleton pattern — only one registry per process."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialised = False
            return cls._instance

    def __init__(self) -> None:
        if self._initialised:
            return
        self._initialised = True
        self._models: dict[str, torch.nn.Module] = {}
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._vram_allocated_mb: float = 0.0
        logger.info(f"ModelRegistry initialised on device: {self._device}")

    @property
    def device(self) -> torch.device:
        """Get the current compute device."""
        return self._device

    @property
    def vram_usage_mb(self) -> float:
        """Get current VRAM usage in megabytes."""
        if self._device.type == "cuda":
            return torch.cuda.memory_allocated(self._device) / (1024 ** 2)
        return 0.0

    def load_gnn(self, model_path: str | None = None) -> FloorPlanGNN:
        """Load the FloorPlanGNN model.

        Args:
            model_path: Path to model weights (defaults to settings.model_weights_path)

        Returns:
            Loaded FloorPlanGNN model on the appropriate device
        """
        if "floorplan_gnn" in self._models:
            return self._models["floorplan_gnn"]

        model_path = model_path or settings.gnn_model_path

        model = FloorPlanGNN()

        if os.path.exists(model_path):
            state_dict = torch.load(
                model_path, map_location=self._device, weights_only=True
            )
            model.load_state_dict(state_dict)
            logger.info(f"Loaded GNN weights from {model_path}")
        else:
            logger.warning(
                f"No weights at {model_path} — using random initialisation"
            )

        model = model.to(self._device)
        model.eval()
        self._models["floorplan_gnn"] = model

        logger.info(
            f"GNN model loaded — VRAM: {self.vram_usage_mb:.1f} MB"
        )
        return model

    def warm_up(self) -> None:
        """Warm up models with a dummy forward pass to ensure CUDA kernels are loaded."""
        gnn = self.load_gnn()

        dummy_x = torch.randn(3, 18, device=self._device)
        dummy_edge = torch.tensor([[0, 1, 2], [1, 2, 0]], device=self._device)
        dummy_edge_attr = torch.randn(3, 3, device=self._device)

        with torch.inference_mode():
            gnn(dummy_x, dummy_edge, dummy_edge_attr)

        logger.info("Model warm-up complete")

    def unload_all(self) -> None:
        """Unload all models and free VRAM."""
        self._models.clear()
        if self._device.type == "cuda":
            torch.cuda.empty_cache()
        self._vram_allocated_mb = 0.0
        logger.info("All models unloaded")

    def status(self) -> dict:
        """Return registry status for health checks."""
        return {
            "device": str(self._device),
            "loaded_models": list(self._models.keys()),
            "vram_usage_mb": round(self.vram_usage_mb, 2),
            "cuda_available": torch.cuda.is_available(),
        }
