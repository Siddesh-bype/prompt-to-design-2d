"""
Job Queue — Redis + RQ integration for background task processing.

Provides functions to enqueue layout generation jobs, poll status,
and manage job lifecycle.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Optional

import redis

from app.core.config import settings
from app.models.schemas import JobStatusResponse, LayoutGraph

logger = logging.getLogger(__name__)


# ─── Redis Connection ────────────────────────────────────────────────────────


def get_redis_connection() -> redis.Redis:
    """Get a Redis client connection.

    Returns:
        redis.Redis client instance
    """
    return redis.Redis(
        host=settings.redis_host,
        port=settings.redis_port,
        db=0,
        decode_responses=True,
    )


# ─── Job Key Helpers ─────────────────────────────────────────────────────────


def _job_key(job_id: str) -> str:
    """Generate Redis key for a job."""
    return f"blueprint:job:{job_id}"


def _ws_channel(job_id: str) -> str:
    """Generate Redis pub/sub channel for job WebSocket updates."""
    return f"blueprint:ws:{job_id}"


# ─── Enqueue ─────────────────────────────────────────────────────────────────


def enqueue_job(
    prompt: str,
    plot_sqm: float,
    facing: str,
    vastu_enabled: bool,
) -> str:
    """Create a new layout generation job and store it in Redis.

    Args:
        prompt: User prompt text
        plot_sqm: Plot area in square metres
        facing: Compass facing string
        vastu_enabled: Whether Vastu checking is enabled

    Returns:
        job_id string (UUID)
    """
    job_id = str(uuid.uuid4())
    r = get_redis_connection()

    job_data = {
        "job_id": job_id,
        "status": "queued",
        "progress_pct": 0,
        "prompt": prompt,
        "plot_sqm": plot_sqm,
        "facing": facing,
        "vastu_enabled": str(vastu_enabled).lower(),
        "created_at": datetime.utcnow().isoformat(),
        "result": "",
        "svg_string": "",
        "error": "",
    }

    r.hset(_job_key(job_id), mapping=job_data)
    r.expire(_job_key(job_id), int(timedelta(hours=1).total_seconds()))

    # Publish creation event
    r.publish(_ws_channel(job_id), json.dumps({
        "type": "status",
        "status": "queued",
        "progress_pct": 0,
    }))

    logger.info(f"Job {job_id} enqueued: '{prompt[:50]}...'")
    return job_id


# ─── Status Updates ─────────────────────────────────────────────────────────


def update_job_status(
    job_id: str,
    status: str,
    progress_pct: int,
    result: Optional[dict] = None,
    svg_string: Optional[str] = None,
    error: Optional[dict] = None,
) -> None:
    """Update a job's status in Redis and publish via pub/sub.

    Args:
        job_id: Job UUID
        status: New status string
        progress_pct: Progress percentage (0–100)
        result: Optional layout graph dict
        svg_string: Optional SVG string
        error: Optional error detail dict
    """
    r = get_redis_connection()
    key = _job_key(job_id)

    updates = {
        "status": status,
        "progress_pct": str(progress_pct),
    }

    if result is not None:
        updates["result"] = json.dumps(result)
    if svg_string is not None:
        updates["svg_string"] = svg_string
    if error is not None:
        updates["error"] = json.dumps(error)

    r.hset(key, mapping=updates)

    # Publish WebSocket update
    ws_msg = {
        "type": "status",
        "status": status,
        "progress_pct": progress_pct,
    }
    if result is not None:
        ws_msg["result"] = result
    if svg_string is not None:
        ws_msg["svg_string"] = svg_string
    if error is not None:
        ws_msg["error"] = error

    r.publish(_ws_channel(job_id), json.dumps(ws_msg))

    logger.info(f"Job {job_id}: {status} ({progress_pct}%)")


# ─── Get Job Status ─────────────────────────────────────────────────────────


def get_job_status(job_id: str) -> Optional[JobStatusResponse]:
    """Retrieve current job status from Redis.

    Args:
        job_id: Job UUID

    Returns:
        JobStatusResponse or None if job not found
    """
    r = get_redis_connection()
    data = r.hgetall(_job_key(job_id))

    if not data:
        return None

    result = None
    svg_string = None
    error = None

    if data.get("result"):
        try:
            result = LayoutGraph.model_validate_json(data["result"])
        except Exception:
            try:
                result = LayoutGraph.model_validate(json.loads(data["result"]))
            except Exception:
                pass

    if data.get("svg_string"):
        svg_string = data["svg_string"]

    if data.get("error"):
        try:
            error_data = json.loads(data["error"])
            from app.models.schemas import ErrorDetail
            error = ErrorDetail.model_validate(error_data)
        except Exception:
            pass

    return JobStatusResponse(
        status=data.get("status", "queued"),
        progress_pct=int(data.get("progress_pct", 0)),
        result=result,
        svg_string=svg_string,
        error=error,
    )


# ─── Cleanup ─────────────────────────────────────────────────────────────────


def delete_job(job_id: str) -> bool:
    """Delete a job from Redis.

    Args:
        job_id: Job UUID

    Returns:
        True if job was deleted, False if not found
    """
    r = get_redis_connection()
    return r.delete(_job_key(job_id)) > 0
