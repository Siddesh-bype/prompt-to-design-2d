"""
Export API Routes — /api/v1/export/dxf, /api/v1/export/svg

Handles conversion of layout graphs to downloadable DXF/SVG files.
"""

from __future__ import annotations

import logging
import math

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from app.models.schemas import ExportRequest
from app.services.renderer import layout_to_svg, layout_to_dxf, HAS_EZDXF

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/export", tags=["export"])


@router.post("/dxf")
async def export_dxf(request: ExportRequest) -> Response:
    """Export a layout as a DXF file.

    Returns downloadable DXF bytes with appropriate headers.
    """
    if not HAS_EZDXF:
        raise HTTPException(
            status_code=501,
            detail="DXF export requires ezdxf library (not installed)",
        )

    try:
        plot_side = math.sqrt(request.layout_graph.plot_area_sqm)
        dxf_bytes = layout_to_dxf(
            request.layout_graph,
            plot_width_m=plot_side,
            plot_height_m=plot_side,
            scale_str=request.scale,
        )

        return Response(
            content=dxf_bytes,
            media_type="application/dxf",
            headers={
                "Content-Disposition": "attachment; filename=floorplan.dxf",
            },
        )
    except Exception as e:
        logger.error(f"DXF export failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"DXF export failed: {str(e)}",
        )


@router.post("/svg")
async def export_svg(request: ExportRequest) -> Response:
    """Export a layout as an SVG file.

    Returns downloadable SVG string with appropriate headers.
    """
    try:
        svg_string = layout_to_svg(request.layout_graph)

        return Response(
            content=svg_string,
            media_type="image/svg+xml",
            headers={
                "Content-Disposition": "attachment; filename=floorplan.svg",
            },
        )
    except Exception as e:
        logger.error(f"SVG export failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"SVG export failed: {str(e)}",
        )
