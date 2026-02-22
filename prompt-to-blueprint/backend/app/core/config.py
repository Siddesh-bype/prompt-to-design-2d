"""Pydantic settings for the Prompt-to-Blueprint AI backend."""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # --- Server ---
    backend_port: int = 8000
    frontend_url: str = "http://localhost:3000"
    secret_key: str = "change-me-in-production"
    debug: bool = True

    # --- Redis ---
    redis_url: str = "redis://localhost:6379/0"

    # --- Ollama ---
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:1.5b"
    ollama_timeout: int = 30

    # --- Model Paths ---
    gnn_model_path: str = "models/floorplan_gnn.pt"
    gnn_checkpoint_dir: str = "models/"

    # --- Worker ---
    rq_queue_name: str = "layout"
    job_ttl: int = 86400

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


# Singleton settings instance
settings = Settings()
