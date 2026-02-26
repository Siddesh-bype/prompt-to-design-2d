"""
Job Queue — In-memory job management for background task processing.

Provides functions to enqueue layout generation jobs, poll status,
and manage job lifecycle using a simple thread-safe dictionary.
"""

from __future__ import annotations

import json
import logging
import uuid
import asyncio
import threading
from datetime import datetime
from typing import Optional

from app.models.schemas import JobStatusResponse, LayoutGraph

logger = logging.getLogger(__name__)


# ─── In-Memory Store ─────────────────────────────────────────────────────────

_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()

# Subscribers: job_id -> list of asyncio.Queue
_subscribers: dict[str, list[asyncio.Queue]] = {}
_subscribers_lock = threading.Lock()


# ─── Job Key Helpers ─────────────────────────────────────────────────────────


def _ws_channel(job_id: str) -> str:
    """Generate channel name for job WebSocket updates."""
    return f"blueprint:ws:{job_id}"


# ─── Enqueue ─────────────────────────────────────────────────────────────────


def enqueue_job(
    prompt: str,
    plot_sqm: float,
    facing: str,
    vastu_enabled: bool,
) -> str:
    """Create a new layout generation job and store it in memory.

    Args:
        prompt: User prompt text
        plot_sqm: Plot area in square metres
        facing: Compass facing string
        vastu_enabled: Whether Vastu checking is enabled

    Returns:
        job_id string (UUID)
    """
    job_id = str(uuid.uuid4())

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

    with _jobs_lock:
        _jobs[job_id] = job_data

    # Publish creation event
    _publish(job_id, {
        "type": "status",
        "status": "queued",
        "progress_pct": 0,
    })

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
    """Update a job's status in memory and notify subscribers.

    Args:
        job_id: Job UUID
        status: New status string
        progress_pct: Progress percentage (0–100)
        result: Optional layout graph dict
        svg_string: Optional SVG string
        error: Optional error detail dict
    """
    with _jobs_lock:
        if job_id not in _jobs:
            return

        _jobs[job_id]["status"] = status
        _jobs[job_id]["progress_pct"] = str(progress_pct)

        if result is not None:
            _jobs[job_id]["result"] = json.dumps(result)
        if svg_string is not None:
            _jobs[job_id]["svg_string"] = svg_string
        if error is not None:
            _jobs[job_id]["error"] = json.dumps(error)

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

    _publish(job_id, ws_msg)

    logger.info(f"Job {job_id}: {status} ({progress_pct}%)")


# ─── Pub/Sub (In-Memory) ────────────────────────────────────────────────────


def _publish(job_id: str, message: dict) -> None:
    """Publish a message to all subscribers of a job."""
    with _subscribers_lock:
        queues = _subscribers.get(job_id, [])
        for q in queues:
            try:
                q.put_nowait(message)
            except asyncio.QueueFull:
                pass


def subscribe(job_id: str) -> asyncio.Queue:
    """Subscribe to updates for a job. Returns an asyncio.Queue."""
    q: asyncio.Queue = asyncio.Queue(maxsize=100)
    with _subscribers_lock:
        if job_id not in _subscribers:
            _subscribers[job_id] = []
        _subscribers[job_id].append(q)
    return q


def unsubscribe(job_id: str, q: asyncio.Queue) -> None:
    """Unsubscribe from job updates."""
    with _subscribers_lock:
        if job_id in _subscribers:
            try:
                _subscribers[job_id].remove(q)
            except ValueError:
                pass
            if not _subscribers[job_id]:
                del _subscribers[job_id]


# ─── Get Job Status ─────────────────────────────────────────────────────────


def get_job_status(job_id: str) -> Optional[JobStatusResponse]:
    """Retrieve current job status from memory.

    Args:
        job_id: Job UUID

    Returns:
        JobStatusResponse or None if job not found
    """
    with _jobs_lock:
        data = _jobs.get(job_id)
        if not data:
            return None
        data = dict(data)  # Copy to avoid race

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
    """Delete a job from memory.

    Args:
        job_id: Job UUID

    Returns:
        True if job was deleted, False if not found
    """
    with _jobs_lock:
        if job_id in _jobs:
            del _jobs[job_id]
            return True
        return False
