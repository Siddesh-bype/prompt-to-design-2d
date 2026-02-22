"""
Layout Worker — RQ background task for the full generation pipeline.

Pipeline stages:
1. NLP Parsing (0–25%)
2. GNN Layout Generation (25–50%)
3. Constraint Solving (50–75%)
4. SVG Rendering (75–95%)
5. Vastu Scoring (95–100%)
"""

from __future__ import annotations

import logging
import traceback
from typing import Optional

from app.core.job_queue import update_job_status
from app.models.schemas import CompassFacing, ErrorDetail

logger = logging.getLogger(__name__)


def run_layout_pipeline(
    job_id: str,
    prompt: str,
    plot_sqm: float,
    facing: str,
    vastu_enabled: bool,
) -> None:
    """Execute the full layout generation pipeline as a background task.

    Args:
        job_id: Job UUID for status updates
        prompt: User prompt text
        plot_sqm: Plot area in square metres
        facing: Compass facing string (NORTH, SOUTH, EAST, WEST)
        vastu_enabled: Whether to run Vastu compliance check
    """
    import math

    try:
        # ── Stage 1: NLP Parsing ─────────────────────────────────
        update_job_status(job_id, "parsing", 5)

        from app.services.nlp_parser import parse_prompt
        parsed = parse_prompt(prompt)

        # Override with user-specified values
        parsed = parsed.model_copy(update={
            "plot_area_sqm": plot_sqm,
            "facing": CompassFacing(facing),
            "vastu_enabled": vastu_enabled,
        })

        update_job_status(job_id, "parsing", 25)
        logger.info(f"Job {job_id}: parsed {len(parsed.rooms)} rooms")

        # ── Stage 2: GNN Layout ──────────────────────────────────
        update_job_status(job_id, "layout_generating", 30)

        from app.services.gnn_engine import run_gnn_inference
        layout = run_gnn_inference(parsed)

        update_job_status(job_id, "layout_generating", 50)
        logger.info(f"Job {job_id}: GNN generated {len(layout.rooms)} rooms")

        # ── Stage 3: Constraint Solving ──────────────────────────
        update_job_status(job_id, "solving_constraints", 55)

        from app.services.constraint_solver import run_constraint_pipeline
        plot_side = math.sqrt(plot_sqm)
        solved = run_constraint_pipeline(layout, plot_side, plot_side)

        update_job_status(job_id, "solving_constraints", 75)
        logger.info(
            f"Job {job_id}: constraints solved — "
            f"overlap={solved.overlap_rate:.3f}, "
            f"adj={solved.adjacency_satisfaction:.3f}"
        )

        # ── Stage 4: SVG Rendering ───────────────────────────────
        update_job_status(job_id, "rendering", 80)

        from app.services.renderer import layout_to_svg
        svg_string = layout_to_svg(solved)

        update_job_status(job_id, "rendering", 90)

        # ── Stage 5: Vastu Check (optional) ──────────────────────
        vastu_result = None
        if vastu_enabled:
            update_job_status(job_id, "rendering", 92)

            from app.services.vastu_engine import check_vastu
            vastu_result = check_vastu(solved)

            logger.info(
                f"Job {job_id}: Vastu score = {vastu_result.score}/100"
            )

        # ── Complete ─────────────────────────────────────────────
        result_dict = solved.model_dump()
        if vastu_result:
            result_dict["vastu"] = vastu_result.model_dump()

        update_job_status(
            job_id=job_id,
            status="complete",
            progress_pct=100,
            result=result_dict,
            svg_string=svg_string,
        )
        logger.info(f"Job {job_id}: complete ✓")

    except Exception as e:
        error_tb = traceback.format_exc()
        logger.error(f"Job {job_id} failed: {e}\n{error_tb}")

        error_detail = ErrorDetail(
            error_code="PIPELINE_FAILURE",
            message=str(e),
            fallback_used=False,
            details={"traceback": error_tb[:500]},
        )

        update_job_status(
            job_id=job_id,
            status="error",
            progress_pct=0,
            error=error_detail.model_dump(),
        )
