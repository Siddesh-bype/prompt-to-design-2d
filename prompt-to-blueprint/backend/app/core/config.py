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

    # --- NLP Provider ---
    nlp_provider: str = "claude"  # "claude" or "openrouter"

    # --- Claude API (primary) ---
    anthropic_api_key: str = ""
    claude_model: str = "claude-haiku-4-5-20251001"
    claude_max_tokens: int = 4096
    claude_timeout: int = 45

    # --- OpenRouter (backup) ---
    openrouter_api_key: str = ""
    openrouter_model: str = "openai/gpt-oss-120b:free"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_timeout: int = 30

    # --- Model Paths ---
    gnn_model_path: str = "models/floorplan_gnn.pt"
    gnn_checkpoint_dir: str = "models/"

    model_config = {
        "env_file": ("../.env", ".env"),
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


# Singleton settings instance
settings = Settings()
