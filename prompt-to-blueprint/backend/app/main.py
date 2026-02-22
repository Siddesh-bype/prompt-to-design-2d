"""
FastAPI Application — Prompt-to-Blueprint AI backend.

Main application entry point with CORS, lifespan management,
router inclusion, and health checks.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─── Lifespan ────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — startup and shutdown hooks."""
    # Startup
    logger.info("=" * 60)
    logger.info("  Prompt-to-Blueprint AI — Starting Up")
    logger.info("=" * 60)

    # Check Redis
    try:
        import redis
        r = redis.Redis(host=settings.redis_host, port=settings.redis_port, db=0)
        r.ping()
        logger.info(f"✓ Redis connected at {settings.redis_host}:{settings.redis_port}")
    except Exception as e:
        logger.warning(f"✗ Redis not available: {e}")

    # Check Ollama
    try:
        import httpx
        resp = httpx.get(f"{settings.ollama_host}/api/version", timeout=5)
        version = resp.json().get("version", "unknown")
        logger.info(f"✓ Ollama connected at {settings.ollama_host} (v{version})")
    except Exception as e:
        logger.warning(f"✗ Ollama not available: {e}")

    # Warm up model registry
    try:
        from app.core.model_registry import ModelRegistry
        registry = ModelRegistry()
        registry.load_gnn()
        logger.info(f"✓ Model registry ready — {registry.status()}")
    except Exception as e:
        logger.warning(f"✗ Model registry warm-up failed: {e}")

    logger.info(f"  Backend running on port {settings.backend_port}")
    logger.info("=" * 60)

    yield

    # Shutdown
    logger.info("Shutting down...")
    try:
        from app.core.model_registry import ModelRegistry
        ModelRegistry().unload_all()
    except Exception:
        pass


# ─── Application ─────────────────────────────────────────────────────────────


app = FastAPI(
    title="Prompt-to-Blueprint AI",
    description="Converts natural language descriptions into 2D architectural floor plans using AI",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        settings.frontend_url,
        "http://localhost:3000",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
from app.api.routes.generate import router as generate_router
from app.api.routes.vastu import router as vastu_router
from app.api.routes.export import router as export_router

app.include_router(generate_router)
app.include_router(vastu_router)
app.include_router(export_router)


# ─── Root ────────────────────────────────────────────────────────────────────


@app.get("/")
async def root():
    """Root endpoint — basic API info."""
    return {
        "name": "Prompt-to-Blueprint AI",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/api/v1/health",
    }
