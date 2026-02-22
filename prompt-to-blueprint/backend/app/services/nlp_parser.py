"""
NLP Parser Service — Ollama SLM integration for natural language to structured JSON.

Converts user prompts into ParsedLayout objects using a local Small Language Model
served via Ollama, with repair logic for handling inconsistent outputs.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

import httpx

from app.models.schemas import (
    AdjacencyEdge,
    CompassFacing,
    ConnectionType,
    ParsedLayout,
    RoomSpec,
    RoomType,
)
from app.core.config import settings

logger = logging.getLogger(__name__)


# ─── Error Classes ───────────────────────────────────────────────────────────


class OllamaConnectionError(Exception):
    """Raised when the Ollama server is unreachable."""
    pass


class ParseValidationError(Exception):
    """Raised when the NLP output cannot be validated after repair attempts."""
    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.details = details or {}


# ─── System Prompt ───────────────────────────────────────────────────────────


def build_system_prompt() -> str:
    """Build the system prompt that instructs the SLM to extract structured data."""
    return """You are a specialised architectural floor plan parser. Your ONLY job is to extract structured information from a natural language description of a home/apartment and output VALID JSON.

RULES:
1. Output ONLY a valid JSON object. No markdown, no explanation, no prose outside the JSON.
2. The JSON must match this exact schema:
{
  "rooms": [
    {
      "room_id": "room_1",
      "room_type": "<ROOM_TYPE>",
      "target_area_sqm": <float 4.0-80.0>,
      "compass_preference": "<NORTH|SOUTH|EAST|WEST|null>",
      "label": "<optional string or null>"
    }
  ],
  "plot_area_sqm": <float 30.0-500.0>,
  "facing": "<NORTH|SOUTH|EAST|WEST>",
  "style_hints": ["<string>"],
  "adjacency_constraints": [
    {
      "room_a_id": "room_1",
      "room_b_id": "room_2",
      "connection_type": "<DOOR|OPENING|WALL>",
      "required": true
    }
  ],
  "vastu_enabled": false
}

ROOM_TYPE values: LIVING_ROOM, MASTER_BEDROOM, BEDROOM, KITCHEN, BATHROOM, TOILET, CORRIDOR, BALCONY, STUDY, DINING, UTILITY, GARAGE

INDIAN TERMINOLOGY MAPPINGS:
- "BHK" = Bedroom-Hall-Kitchen configuration
  - 1BHK = 1 BEDROOM + 1 LIVING_ROOM + 1 KITCHEN + 1 BATHROOM
  - 2BHK = 1 MASTER_BEDROOM + 1 BEDROOM + 1 LIVING_ROOM + 1 KITCHEN + 2 BATHROOMS
  - 3BHK = 1 MASTER_BEDROOM + 2 BEDROOMS + 1 LIVING_ROOM + 1 KITCHEN + 2 BATHROOMS
  - 4BHK = 1 MASTER_BEDROOM + 3 BEDROOMS + 1 LIVING_ROOM + 1 KITCHEN + 3 BATHROOMS
- "Pooja room" → STUDY
- "Verandah" → BALCONY
- "Drawing room" → LIVING_ROOM
- "Hall" → LIVING_ROOM
- "Wash area" → UTILITY
- "Store room" → UTILITY
- "Car parking" → GARAGE

DEFAULTS:
- If no plot area mentioned, use 100.0 sqm
- If no facing mentioned, use NORTH
- Assign reasonable areas: LIVING_ROOM=20-30, BEDROOM=12-18, MASTER_BEDROOM=16-24, KITCHEN=8-14, BATHROOM=4-8, BALCONY=4-8, CORRIDOR=4-6
- Always add adjacency constraints between: MASTER_BEDROOM↔BATHROOM(DOOR), LIVING_ROOM↔KITCHEN(OPENING), KITCHEN↔DINING(OPENING) if applicable
- Generate unique room_id values like "room_1", "room_2", etc.
"""


# ─── Ollama API Call ─────────────────────────────────────────────────────────


def call_ollama(
    prompt: str,
    system: str,
    model: str | None = None,
) -> dict:
    """Call Ollama's chat API synchronously and return the parsed JSON response.
    
    Args:
        prompt: User prompt text
        system: System prompt text
        model: Ollama model name (defaults to settings.ollama_model)
    
    Returns:
        Parsed JSON dict from the model response
        
    Raises:
        OllamaConnectionError: If the Ollama server is unreachable
        ParseValidationError: If the response cannot be parsed as JSON
    """
    model = model or settings.ollama_model
    
    try:
        response = httpx.post(
            f"{settings.ollama_host}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 1024,
                },
            },
            timeout=settings.ollama_timeout,
        )
        response.raise_for_status()
    except httpx.ConnectError as e:
        raise OllamaConnectionError(
            f"Cannot connect to Ollama at {settings.ollama_host}: {e}"
        )
    except httpx.TimeoutException as e:
        raise OllamaConnectionError(
            f"Ollama request timed out after {settings.ollama_timeout}s: {e}"
        )
    except httpx.HTTPStatusError as e:
        raise OllamaConnectionError(
            f"Ollama returned HTTP {e.response.status_code}: {e}"
        )

    data = response.json()
    content = data.get("message", {}).get("content", "")

    # Try to extract JSON from the response (handle markdown code blocks)
    json_str = content.strip()
    if json_str.startswith("```"):
        # Remove markdown code fences
        lines = json_str.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        json_str = "\n".join(lines)

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        # Try to find JSON object in the response
        match = re.search(r"\{.*\}", json_str, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        raise ParseValidationError(
            f"Failed to parse JSON from Ollama response: {e}",
            details={"raw_content": content[:500]},
        )


# ─── Async Ollama Call ───────────────────────────────────────────────────────


async def acall_ollama(
    prompt: str,
    system: str,
    model: str | None = None,
) -> dict:
    """Async version of call_ollama using httpx.AsyncClient."""
    model = model or settings.ollama_model

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{settings.ollama_host}/api/chat",
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 1024,
                    },
                },
                timeout=settings.ollama_timeout,
            )
            response.raise_for_status()
    except httpx.ConnectError as e:
        raise OllamaConnectionError(
            f"Cannot connect to Ollama at {settings.ollama_host}: {e}"
        )
    except httpx.TimeoutException as e:
        raise OllamaConnectionError(
            f"Ollama request timed out after {settings.ollama_timeout}s: {e}"
        )

    data = response.json()
    content = data.get("message", {}).get("content", "")

    json_str = content.strip()
    if json_str.startswith("```"):
        lines = json_str.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        json_str = "\n".join(lines)

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        match = re.search(r"\{.*\}", json_str, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        raise ParseValidationError(
            f"Failed to parse JSON from Ollama response: {e}",
            details={"raw_content": content[:500]},
        )


# ─── Validate and Repair ────────────────────────────────────────────────────


def validate_and_repair(
    raw_json: dict,
    original_prompt: str,
    stated_bedroom_count: int,
) -> ParsedLayout:
    """Validate raw JSON against ParsedLayout schema with automatic repair.

    If the bedroom count in the parsed output is less than stated_bedroom_count,
    runs up to 2 repair passes via call_ollama with explicit correction prompts.
    
    Args:
        raw_json: Raw JSON dict from the Ollama response
        original_prompt: Original user prompt for repair context
        stated_bedroom_count: Expected number of bedrooms from prompt analysis
        
    Returns:
        Validated ParsedLayout
        
    Raises:
        ParseValidationError: If validation fails after all repair attempts
    """
    max_repairs = 2

    for attempt in range(max_repairs + 1):
        try:
            layout = ParsedLayout.model_validate(raw_json)
        except Exception as e:
            if attempt >= max_repairs:
                raise ParseValidationError(
                    f"Failed to parse layout after {max_repairs} repair attempts: {e}",
                    details={
                        "raw_json": raw_json,
                        "last_error": str(e),
                        "attempts": attempt + 1,
                    },
                )
            # Try to fix common issues
            raw_json = _fix_common_issues(raw_json)
            continue

        # Check bedroom count
        bedroom_types = {RoomType.BEDROOM, RoomType.MASTER_BEDROOM}
        actual_bedrooms = sum(
            1 for r in layout.rooms if r.room_type in bedroom_types
        )

        if actual_bedrooms >= stated_bedroom_count or stated_bedroom_count == 0:
            logger.info(
                f"Parse validated on attempt {attempt + 1}: "
                f"{len(layout.rooms)} rooms, {actual_bedrooms} bedrooms"
            )
            return layout

        if attempt >= max_repairs:
            # Accept what we have
            logger.warning(
                f"Bedroom count mismatch after {max_repairs} repairs: "
                f"expected {stated_bedroom_count}, got {actual_bedrooms}. "
                f"Accepting as-is."
            )
            return layout

        # Repair pass: explicitly request missing rooms
        missing = stated_bedroom_count - actual_bedrooms
        repair_prompt = (
            f"The previous output was missing bedrooms. The user asked for "
            f"{stated_bedroom_count} bedrooms but only {actual_bedrooms} were generated. "
            f"Please add {missing} more BEDROOM room(s) to the layout.\n\n"
            f"Original prompt: {original_prompt}\n\n"
            f"Previous output: {json.dumps(raw_json, indent=2)}\n\n"
            f"Generate the corrected complete JSON output with all rooms."
        )

        logger.info(
            f"Repair pass {attempt + 1}: requesting {missing} more bedrooms"
        )
        try:
            raw_json = call_ollama(repair_prompt, build_system_prompt())
        except (OllamaConnectionError, ParseValidationError) as e:
            logger.warning(f"Repair call failed: {e}")
            # Return what we have
            return layout

    # Should not reach here, but just in case
    raise ParseValidationError(
        "Unexpected error in validate_and_repair",
        details={"raw_json": raw_json},
    )


def _fix_common_issues(raw_json: dict) -> dict:
    """Fix common issues in raw JSON output from the SLM."""
    fixed = dict(raw_json)

    # Ensure rooms is a list
    if "rooms" not in fixed or not isinstance(fixed.get("rooms"), list):
        fixed["rooms"] = []

    # Fix room_type casing
    for room in fixed.get("rooms", []):
        if isinstance(room, dict):
            rt = room.get("room_type", "")
            if isinstance(rt, str):
                room["room_type"] = rt.upper().replace(" ", "_")

    # Ensure plot_area_sqm
    if "plot_area_sqm" not in fixed:
        fixed["plot_area_sqm"] = 100.0

    # Ensure facing
    if "facing" not in fixed:
        fixed["facing"] = "NORTH"
    elif isinstance(fixed["facing"], str):
        fixed["facing"] = fixed["facing"].upper()

    # Ensure adjacency_constraints
    if "adjacency_constraints" not in fixed:
        fixed["adjacency_constraints"] = []

    # Ensure style_hints
    if "style_hints" not in fixed:
        fixed["style_hints"] = []

    return fixed


# ─── Main Entrypoint ────────────────────────────────────────────────────────


def parse_prompt(user_prompt: str) -> ParsedLayout:
    """Parse a natural language prompt into a structured ParsedLayout.

    Main synchronous entrypoint for the NLP parser service.
    
    Args:
        user_prompt: Natural language description of the floor plan
        
    Returns:
        Validated ParsedLayout object
    """
    # Extract bedroom count hint from prompt
    bedroom_count = _extract_bedroom_count(user_prompt)
    logger.info(
        f"Parsing prompt: '{user_prompt[:80]}...' "
        f"(detected {bedroom_count} bedrooms)"
    )

    # Call Ollama
    system = build_system_prompt()
    raw_json = call_ollama(user_prompt, system)

    # Validate and repair
    parsed = validate_and_repair(raw_json, user_prompt, bedroom_count)

    logger.info(
        f"Parse complete: {len(parsed.rooms)} rooms, "
        f"model={settings.ollama_model}, "
        f"facing={parsed.facing.value}"
    )

    return parsed


async def aparse_prompt(user_prompt: str) -> ParsedLayout:
    """Async version of parse_prompt for use in FastAPI async routes."""
    bedroom_count = _extract_bedroom_count(user_prompt)
    logger.info(
        f"[async] Parsing prompt: '{user_prompt[:80]}...' "
        f"(detected {bedroom_count} bedrooms)"
    )

    system = build_system_prompt()
    raw_json = await acall_ollama(user_prompt, system)

    # validate_and_repair is sync (repair calls are sync)
    parsed = validate_and_repair(raw_json, user_prompt, bedroom_count)

    logger.info(
        f"[async] Parse complete: {len(parsed.rooms)} rooms, "
        f"model={settings.ollama_model}"
    )

    return parsed


def _extract_bedroom_count(prompt: str) -> int:
    """Extract the expected bedroom count from a prompt string.
    
    Looks for patterns like "3BHK", "3 BHK", "2 bedroom", "4-bedroom".
    
    Returns:
        Expected bedroom count (0 if not detected)
    """
    prompt_lower = prompt.lower()

    # Match "3BHK", "3 BHK", "3-BHK"
    bhk_match = re.search(r"(\d+)\s*[-]?\s*bhk", prompt_lower)
    if bhk_match:
        return int(bhk_match.group(1))

    # Match "3 bedroom", "3 bedrooms", "3-bedroom"
    bedroom_match = re.search(
        r"(\d+)\s*[-]?\s*bed\s*room", prompt_lower
    )
    if bedroom_match:
        return int(bedroom_match.group(1))

    return 0
