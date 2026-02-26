"""
NLP Parser Service — Claude (primary) + OpenRouter (backup).

Converts user prompts into ParsedLayout objects using Claude as
the primary provider and OpenRouter as automatic backup.
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


class ClaudeConnectionError(Exception):
    """Raised when the Claude API call fails."""
    pass


class OpenRouterConnectionError(Exception):
    """Raised when the OpenRouter API call fails."""
    pass


class ParseValidationError(Exception):
    """Raised when the NLP output cannot be validated after repair attempts."""
    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.details = details or {}


# ─── System Prompt ───────────────────────────────────────────────────────────


def build_system_prompt() -> str:
    """Build the system prompt that instructs the LLM to extract structured data."""
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
      "compass_preference": "<NORTH|SOUTH|EAST|WEST|null>"  // ONLY cardinal directions, never NE/SW/etc,
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


# ─── JSON Extraction Helper ─────────────────────────────────────────────────


def _extract_json(content: str) -> dict:
    """Extract a JSON object from an LLM response string."""
    json_str = content.strip()

    # Remove markdown code fences
    if json_str.startswith("```"):
        lines = json_str.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        json_str = "\n".join(lines)

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    # Try to find a JSON object in the response
    match = re.search(r"\{.*\}", json_str, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    raise ParseValidationError(
        "Failed to extract JSON from LLM response",
        details={"raw_content": content[:500]},
    )


# ─── Claude API Call ─────────────────────────────────────────────────────────


def call_claude(prompt: str, system: str, model: str | None = None) -> dict:
    """Call Claude API synchronously and return parsed JSON."""
    model = model or settings.claude_model

    try:
        import anthropic
        client = anthropic.Anthropic(
            api_key=settings.anthropic_api_key,
            timeout=settings.claude_timeout,
        )

        message = client.messages.create(
            model=model,
            max_tokens=settings.claude_max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )

        content = ""
        for block in message.content:
            if block.type == "text":
                content += block.text

        if not content:
            raise ClaudeConnectionError("Claude returned an empty response")

        logger.debug(f"Claude raw response: {content[:200]}...")
        return _extract_json(content)

    except ImportError:
        raise ClaudeConnectionError("anthropic package not installed")
    except ParseValidationError:
        raise
    except Exception as e:
        raise ClaudeConnectionError(f"Claude API error: {e}")


async def acall_claude(prompt: str, system: str, model: str | None = None) -> dict:
    """Async version of call_claude."""
    model = model or settings.claude_model

    try:
        import anthropic
        client = anthropic.AsyncAnthropic(
            api_key=settings.anthropic_api_key,
            timeout=settings.claude_timeout,
        )

        message = await client.messages.create(
            model=model,
            max_tokens=settings.claude_max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )

        content = ""
        for block in message.content:
            if block.type == "text":
                content += block.text

        if not content:
            raise ClaudeConnectionError("Claude returned an empty response")

        return _extract_json(content)

    except ImportError:
        raise ClaudeConnectionError("anthropic package not installed")
    except ParseValidationError:
        raise
    except Exception as e:
        raise ClaudeConnectionError(f"Claude API error: {e}")


# ─── OpenRouter API Call ─────────────────────────────────────────────────────


def _is_valid_key(key: str) -> bool:
    """Check if an API key looks real (not a placeholder)."""
    if not key:
        return False
    placeholders = {"", "your-openrouter-key-here", "your-anthropic-api-key-here"}
    return key not in placeholders


def call_openrouter(prompt: str, system: str, model: str | None = None) -> dict:
    """Call OpenRouter API synchronously via OpenAI-compatible interface."""
    model = model or settings.openrouter_model

    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
            timeout=settings.openrouter_timeout,
        )

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=2048,
        )

        content = response.choices[0].message.content or ""
        if not content:
            raise OpenRouterConnectionError("OpenRouter returned empty response")

        logger.debug(f"OpenRouter raw response: {content[:200]}...")
        return _extract_json(content)

    except ImportError:
        raise OpenRouterConnectionError("openai package not installed")
    except ParseValidationError:
        raise
    except Exception as e:
        raise OpenRouterConnectionError(f"OpenRouter API error: {e}")


async def acall_openrouter(prompt: str, system: str, model: str | None = None) -> dict:
    """Async version of call_openrouter."""
    model = model or settings.openrouter_model

    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
            timeout=settings.openrouter_timeout,
        )

        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=2048,
        )

        content = response.choices[0].message.content or ""
        if not content:
            raise OpenRouterConnectionError("OpenRouter returned empty response")

        return _extract_json(content)

    except ImportError:
        raise OpenRouterConnectionError("openai package not installed")
    except ParseValidationError:
        raise
    except Exception as e:
        raise OpenRouterConnectionError(f"OpenRouter API error: {e}")


# ─── Unified LLM Dispatcher ────────────────────────────────────────────────


def call_llm(prompt: str, system: str) -> dict:
    """Unified sync dispatcher: Claude (primary) → OpenRouter (backup).

    Tries Claude first if configured. Falls back to OpenRouter.
    Raises if both fail.
    """
    use_claude = _is_valid_key(settings.anthropic_api_key)
    use_openrouter = _is_valid_key(settings.openrouter_api_key)

    if use_claude:
        try:
            logger.info(f"Calling Claude ({settings.claude_model})...")
            result = call_claude(prompt, system)
            logger.info("Claude response received ✓")
            return result
        except (ClaudeConnectionError, ParseValidationError) as e:
            logger.warning(f"Claude failed: {e}")
            if use_openrouter:
                logger.info("Trying OpenRouter backup...")
            else:
                raise

    if use_openrouter:
        try:
            logger.info(f"Calling OpenRouter ({settings.openrouter_model})...")
            result = call_openrouter(prompt, system)
            logger.info("OpenRouter response received ✓")
            return result
        except (OpenRouterConnectionError, ParseValidationError) as e:
            raise OpenRouterConnectionError(
                f"All NLP providers failed. Last error: {e}"
            )

    raise ClaudeConnectionError(
        "No NLP provider configured. Set ANTHROPIC_API_KEY or OPENROUTER_API_KEY in .env"
    )


async def acall_llm(prompt: str, system: str) -> dict:
    """Unified async dispatcher: Claude → OpenRouter."""
    use_claude = _is_valid_key(settings.anthropic_api_key)
    use_openrouter = _is_valid_key(settings.openrouter_api_key)

    if use_claude:
        try:
            logger.info(f"[async] Calling Claude ({settings.claude_model})...")
            result = await acall_claude(prompt, system)
            logger.info("[async] Claude response received ✓")
            return result
        except (ClaudeConnectionError, ParseValidationError) as e:
            logger.warning(f"[async] Claude failed: {e}")
            if use_openrouter:
                logger.info("[async] Trying OpenRouter backup...")
            else:
                raise

    if use_openrouter:
        try:
            logger.info(f"[async] Calling OpenRouter ({settings.openrouter_model})...")
            result = await acall_openrouter(prompt, system)
            logger.info("[async] OpenRouter response received ✓")
            return result
        except (OpenRouterConnectionError, ParseValidationError) as e:
            raise OpenRouterConnectionError(
                f"[async] All NLP providers failed. Last error: {e}"
            )

    raise ClaudeConnectionError(
        "No NLP provider configured. Set ANTHROPIC_API_KEY or OPENROUTER_API_KEY in .env"
    )


# ─── Validate and Repair ────────────────────────────────────────────────────


def validate_and_repair(
    raw_json: dict,
    original_prompt: str,
    stated_bedroom_count: int,
) -> ParsedLayout:
    """Validate raw JSON against ParsedLayout schema with automatic repair."""
    max_repairs = 2

    # Always fix common issues before first attempt
    raw_json = _fix_common_issues(raw_json)

    for attempt in range(max_repairs + 1):
        try:
            layout = ParsedLayout.model_validate(raw_json)
        except Exception as e:
            if attempt >= max_repairs:
                raise ParseValidationError(
                    f"Failed to parse layout after {max_repairs} repair attempts: {e}",
                    details={"raw_json": raw_json, "last_error": str(e)},
                )
            raw_json = _fix_common_issues(raw_json)
            continue

        # Check bedroom count
        bedroom_types = {RoomType.BEDROOM, RoomType.MASTER_BEDROOM}
        actual_bedrooms = sum(1 for r in layout.rooms if r.room_type in bedroom_types)

        if actual_bedrooms >= stated_bedroom_count or stated_bedroom_count == 0:
            logger.info(
                f"Parse validated on attempt {attempt + 1}: "
                f"{len(layout.rooms)} rooms, {actual_bedrooms} bedrooms"
            )
            return layout

        if attempt >= max_repairs:
            logger.warning(
                f"Bedroom count mismatch after {max_repairs} repairs: "
                f"expected {stated_bedroom_count}, got {actual_bedrooms}. Accepting."
            )
            return layout

        # Repair pass
        missing = stated_bedroom_count - actual_bedrooms
        repair_prompt = (
            f"The previous output was missing bedrooms. The user asked for "
            f"{stated_bedroom_count} bedrooms but only {actual_bedrooms} were generated. "
            f"Please add {missing} more BEDROOM room(s).\n\n"
            f"Original prompt: {original_prompt}\n\n"
            f"Previous output: {json.dumps(raw_json, indent=2)}\n\n"
            f"Generate the corrected complete JSON output with all rooms."
        )

        logger.info(f"Repair pass {attempt + 1}: requesting {missing} more bedrooms")
        try:
            raw_json = call_llm(repair_prompt, build_system_prompt())
        except Exception as e:
            logger.warning(f"Repair call failed: {e}")
            return layout

    raise ParseValidationError("Unexpected error in validate_and_repair")


def _fix_common_issues(raw_json: dict) -> dict:
    """Fix common issues in raw JSON output from the LLM."""
    fixed = dict(raw_json)

    if "rooms" not in fixed or not isinstance(fixed.get("rooms"), list):
        fixed["rooms"] = []

    # Map intercardinal directions to nearest cardinal
    _compass_map = {
        "NORTH": "NORTH", "SOUTH": "SOUTH", "EAST": "EAST", "WEST": "WEST",
        "NORTHEAST": "NORTH", "NORTHWEST": "NORTH",
        "SOUTHEAST": "SOUTH", "SOUTHWEST": "SOUTH",
        "NE": "NORTH", "NW": "NORTH", "SE": "SOUTH", "SW": "SOUTH",
    }

    for room in fixed.get("rooms", []):
        if isinstance(room, dict):
            rt = room.get("room_type", "")
            if isinstance(rt, str):
                room["room_type"] = rt.upper().replace(" ", "_")

            # Fix compass_preference
            cp = room.get("compass_preference")
            if isinstance(cp, str):
                cp_upper = cp.upper().strip()
                if cp_upper in _compass_map:
                    room["compass_preference"] = _compass_map[cp_upper]
                else:
                    room["compass_preference"] = None

    if "plot_area_sqm" not in fixed:
        fixed["plot_area_sqm"] = 100.0
    if "facing" not in fixed:
        fixed["facing"] = "NORTH"
    elif isinstance(fixed["facing"], str):
        facing_upper = fixed["facing"].upper().strip()
        fixed["facing"] = _compass_map.get(facing_upper, "NORTH")
    if "adjacency_constraints" not in fixed:
        fixed["adjacency_constraints"] = []
    if "style_hints" not in fixed:
        fixed["style_hints"] = []

    return fixed


# ─── Main Entrypoints ───────────────────────────────────────────────────────


def parse_prompt(user_prompt: str) -> ParsedLayout:
    """Parse a natural language prompt into a structured ParsedLayout.

    Uses Claude as primary provider with OpenRouter as backup.
    """
    bedroom_count = _extract_bedroom_count(user_prompt)
    logger.info(
        f"Parsing prompt: '{user_prompt[:80]}...' "
        f"(detected {bedroom_count} bedrooms)"
    )

    system = build_system_prompt()
    raw_json = call_llm(user_prompt, system)

    parsed = validate_and_repair(raw_json, user_prompt, bedroom_count)

    logger.info(
        f"Parse complete: {len(parsed.rooms)} rooms, "
        f"facing={parsed.facing.value}"
    )
    return parsed


async def aparse_prompt(user_prompt: str) -> ParsedLayout:
    """Async version of parse_prompt."""
    bedroom_count = _extract_bedroom_count(user_prompt)
    logger.info(
        f"[async] Parsing prompt: '{user_prompt[:80]}...' "
        f"(detected {bedroom_count} bedrooms)"
    )

    system = build_system_prompt()
    raw_json = await acall_llm(user_prompt, system)
    parsed = validate_and_repair(raw_json, user_prompt, bedroom_count)

    logger.info(f"[async] Parse complete: {len(parsed.rooms)} rooms")
    return parsed


def _extract_bedroom_count(prompt: str) -> int:
    """Extract the expected bedroom count from a prompt string."""
    prompt_lower = prompt.lower()

    bhk_match = re.search(r"(\d+)\s*[-]?\s*bhk", prompt_lower)
    if bhk_match:
        return int(bhk_match.group(1))

    bedroom_match = re.search(r"(\d+)\s*[-]?\s*bed\s*room", prompt_lower)
    if bedroom_match:
        return int(bedroom_match.group(1))

    return 0
