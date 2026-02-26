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
from app.core.job_queue import enqueue_job, get_job_status, update_job_status, subscribe, unsubscribe
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
    try:
        from app.workers.layout_worker import run_layout_pipeline
        asyncio.get_running_loop().run_in_executor(
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

    Uses in-memory pub/sub queue to forward status updates to the client.
    """
    await websocket.accept()

    queue = subscribe(job_id)

    try:
        logger.info(f"WS connected for job {job_id}")

        # Send current status immediately
        current = get_job_status(job_id)
        if current:
            await websocket.send_json(current.model_dump())

        # Listen for updates from the in-memory queue
        while True:
            try:
                message = queue.get_nowait()
            except asyncio.QueueEmpty:
                await asyncio.sleep(0.2)
                continue

            await websocket.send_json(message)

            # Close WebSocket when job is complete or errored
            status = message.get("status", "")
            if status in ("complete", "error"):
                break

    except WebSocketDisconnect:
        logger.info(f"WS disconnected for job {job_id}")
    except Exception as e:
        logger.error(f"WS error for job {job_id}: {e}")
        try:
            await websocket.close(code=1011, reason=str(e))
        except Exception:
            pass
    finally:
        unsubscribe(job_id, queue)


# ─── GET /health ─────────────────────────────────────────────────────────────


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    health = {
        "status": "healthy",
        "services": {},
    }

    # NLP Provider
    health["services"]["nlp_provider"] = settings.nlp_provider
    from app.services.nlp_parser import _is_valid_key
    providers = []
    if _is_valid_key(settings.anthropic_api_key):
        providers.append("claude")
    if _is_valid_key(settings.openrouter_api_key):
        providers.append("openrouter")
    health["services"]["nlp_status"] = " → ".join(providers) if providers else "none configured"

    # Model registry
    try:
        from app.core.model_registry import ModelRegistry
        registry = ModelRegistry()
        health["services"]["model_registry"] = registry.status()
    except Exception:
        health["services"]["model_registry"] = "unavailable"

    return health
