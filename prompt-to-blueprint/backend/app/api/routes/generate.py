"""
API Routes — /api/v1/generate, /api/v1/jobs/{id}, /api/v1/health

Handles job creation, WebSocket streaming, and status polling.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.job_queue import enqueue_job, get_job_status, update_job_status
from app.models.schemas import (
    GenerateRequest,
    GenerateResponse,
    JobStatusResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["generate"])


# ─── POST /generate ──────────────────────────────────────────────────────────


@router.post("/generate", response_model=GenerateResponse)
async def generate_layout(request: GenerateRequest) -> GenerateResponse:
    """Submit a prompt for floor plan generation.

    Creates a background job and returns a job_id + WebSocket URL.
    """
    try:
        job_id = enqueue_job(
            prompt=request.prompt,
            plot_sqm=request.plot_sqm,
            facing=request.facing.value,
            vastu_enabled=request.vastu_enabled,
        )
    except Exception as e:
        logger.error(f"Failed to enqueue job: {e}")
        raise HTTPException(status_code=503, detail="Job queue unavailable")

    # Trigger the worker in background
    # In production this would be handled by RQ; for dev, run inline
    try:
        from app.workers.layout_worker import run_layout_pipeline
        asyncio.get_event_loop().run_in_executor(
            None,
            run_layout_pipeline,
            job_id,
            request.prompt,
            request.plot_sqm,
            request.facing.value,
            request.vastu_enabled,
        )
    except Exception as e:
        logger.warning(f"Background worker scheduling failed: {e}")

    ws_url = f"ws://localhost:{settings.backend_port}/api/v1/ws/{job_id}"

    return GenerateResponse(
        job_id=job_id,
        ws_url=ws_url,
        estimated_seconds=15,
    )


# ─── GET /jobs/{job_id} ─────────────────────────────────────────────────────


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job(job_id: str) -> JobStatusResponse:
    """Poll job status by ID."""
    result = get_job_status(job_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return result


# ─── WebSocket /ws/{job_id} ──────────────────────────────────────────────────


@router.websocket("/ws/{job_id}")
async def websocket_job_stream(websocket: WebSocket, job_id: str):
    """Stream real-time job updates via WebSocket.

    Subscribes to Redis pub/sub channel for the job and forwards
    all status updates to the connected client.
    """
    await websocket.accept()

    try:
        import redis

        r = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=0,
            decode_responses=True,
        )
        pubsub = r.pubsub()
        channel = f"blueprint:ws:{job_id}"
        pubsub.subscribe(channel)

        logger.info(f"WS connected for job {job_id}")

        # Send current status immediately
        current = get_job_status(job_id)
        if current:
            await websocket.send_json(current.model_dump())

        # Listen for updates
        while True:
            message = pubsub.get_message(timeout=0.5)
            if message and message["type"] == "message":
                data = message["data"]
                if isinstance(data, str):
                    parsed = json.loads(data)
                    await websocket.send_json(parsed)

                    # Close WebSocket when job is complete or errored
                    status = parsed.get("status", "")
                    if status in ("complete", "error"):
                        break

            # Small sleep to avoid busy loop
            await asyncio.sleep(0.1)

    except WebSocketDisconnect:
        logger.info(f"WS disconnected for job {job_id}")
    except Exception as e:
        logger.error(f"WS error for job {job_id}: {e}")
        try:
            await websocket.close(code=1011, reason=str(e))
        except Exception:
            pass
    finally:
        try:
            pubsub.unsubscribe(channel)
            pubsub.close()
        except Exception:
            pass


# ─── GET /health ─────────────────────────────────────────────────────────────


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    health = {
        "status": "healthy",
        "services": {},
    }

    # Check Redis
    try:
        import redis
        r = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=0,
        )
        r.ping()
        health["services"]["redis"] = "connected"
    except Exception:
        health["services"]["redis"] = "disconnected"

    # Check Ollama
    try:
        import httpx
        resp = httpx.get(f"{settings.ollama_host}/api/version", timeout=3)
        health["services"]["ollama"] = "connected"
    except Exception:
        health["services"]["ollama"] = "disconnected"

    # Model registry
    try:
        from app.core.model_registry import ModelRegistry
        registry = ModelRegistry()
        health["services"]["model_registry"] = registry.status()
    except Exception:
        health["services"]["model_registry"] = "unavailable"

    return health
