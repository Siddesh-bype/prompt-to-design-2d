"""
Vastu API Routes — /api/v1/vastu

Provides Vastu Shastra compliance checking for floor plan layouts.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from app.models.schemas import VastuRequest, VastuResponse
from app.services.vastu_engine import check_vastu

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["vastu"])


@router.post("/vastu", response_model=VastuResponse)
async def check_vastu_compliance(request: VastuRequest) -> VastuResponse:
    """Check Vastu Shastra compliance for a layout.

    Accepts a LayoutGraph and returns a score, suggestions,
    and room-to-zone mapping.
    """
    try:
        result = check_vastu(request.layout_graph)
        return result
    except Exception as e:
        logger.error(f"Vastu check failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Vastu compliance check failed: {str(e)}",
        )
