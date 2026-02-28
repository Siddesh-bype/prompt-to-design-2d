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

# Defaults for auto-repair when the LLM output is incomplete or inconsistent.
ROOM_DEFAULT_AREA = {
    RoomType.LIVING_ROOM: 24.0,
    RoomType.MASTER_BEDROOM: 20.0,
    RoomType.BEDROOM: 14.0,
    RoomType.KITCHEN: 10.0,
    RoomType.BATHROOM: 5.0,
    RoomType.TOILET: 4.5,
    RoomType.CORRIDOR: 5.0,
    RoomType.BALCONY: 6.0,
    RoomType.STUDY: 8.0,
    RoomType.DINING: 10.0,
    RoomType.UTILITY: 5.0,
    RoomType.GARAGE: 18.0,
}

BHK_TARGETS = {
    1: {
        RoomType.BEDROOM: 1,
        RoomType.LIVING_ROOM: 1,
        RoomType.KITCHEN: 1,
        RoomType.BATHROOM: 1,
    },
    2: {
        RoomType.MASTER_BEDROOM: 1,
        RoomType.BEDROOM: 1,
        RoomType.LIVING_ROOM: 1,
        RoomType.KITCHEN: 1,
        RoomType.BATHROOM: 2,
        RoomType.CORRIDOR: 1,
    },
    3: {
        RoomType.MASTER_BEDROOM: 1,
        RoomType.BEDROOM: 2,
        RoomType.LIVING_ROOM: 1,
        RoomType.KITCHEN: 1,
        RoomType.BATHROOM: 3,
        RoomType.CORRIDOR: 1,
    },
    4: {
        RoomType.MASTER_BEDROOM: 1,
        RoomType.BEDROOM: 3,
        RoomType.LIVING_ROOM: 1,
        RoomType.KITCHEN: 1,
        RoomType.BATHROOM: 4,
        RoomType.CORRIDOR: 1,
    },
}

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
    return """You are a specialised architectural floor plan parser. Your job is to extract structured information from a natural language description of a home/apartment and output VALID JSON.

To ensure high accuracy, you MUST follow a rigorous thinking and verification process before generating the final JSON layout.

RULES:
1. Output ONLY a valid JSON object. No markdown, no explanation, no prose outside the JSON.
2. The JSON must match this exact schema:
{
  "thinking_": "<one concise sentence describing your analysis>",
  "verification_": "<one concise sentence confirming all rooms connected and correct count>",
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
compass_preference: ONLY cardinal directions (NORTH, SOUTH, EAST, WEST). NEVER use NE, NW, SE, SW, NORTHEAST, etc.

INDIAN TERMINOLOGY MAPPINGS:
- "BHK" = Bedroom-Hall-Kitchen configuration
  - 1BHK = 1 BEDROOM + 1 LIVING_ROOM + 1 KITCHEN + 1 BATHROOM
  - 2BHK = 1 MASTER_BEDROOM + 1 BEDROOM + 1 LIVING_ROOM + 1 KITCHEN + 2 BATHROOMS + 1 CORRIDOR
  - 3BHK = 1 MASTER_BEDROOM + 2 BEDROOMS + 1 LIVING_ROOM + 1 KITCHEN + 3 BATHROOMS + 1 CORRIDOR
  - 4BHK = 1 MASTER_BEDROOM + 3 BEDROOMS + 1 LIVING_ROOM + 1 KITCHEN + 4 BATHROOMS + 1 CORRIDOR
- "Pooja room" → STUDY
- "Verandah" → BALCONY
- "Drawing room" → LIVING_ROOM
- "Hall" → LIVING_ROOM
- "Wash area" → UTILITY
- "Store room" → UTILITY
- "Car parking" → GARAGE

ARCHITECTURAL RULES (MUST FOLLOW):
- Every BEDROOM and MASTER_BEDROOM MUST have its own attached BATHROOM connected via a DOOR adjacency.
- For 2+ bedroom layouts, always include exactly one CORRIDOR room (4-6 sqm) that acts as a central circulation spine.
- The CORRIDOR connects to all major private rooms (each BEDROOM/MASTER_BEDROOM) via DOOR adjacencies.
- Bathrooms are accessed ONLY through their parent bedroom (DOOR), never directly from the Living Room or Kitchen.
- LIVING_ROOM connects to KITCHEN via OPENING, promoting an open-plan living layout.
- KITCHEN connects to DINING (if present) via OPENING.
- Ensure logical flow: Entrance -> Living Room -> Dining/Kitchen -> Corridor -> Bedrooms.

10. EN-SUITE BATHROOMS: If a prompt says "bedroom with attached bathroom", you MUST add a DOOR connection between that specific BEDROOM and that specific BATHROOM. Double check this in your `verification_` step.
11. VASTU COMPLIANCE: If Vastu is enabled or requested, use these zones (based on plot facing):
    - MASTER_BEDROOM must be in the South-West (SW).
    - KITCHEN must be in the South-East (SE) or North-West (NW).
    - STUDY (Pooja/Prayer room) must be in the North-East (NE).
    - LIVING_ROOM is ideal in North, East, or North-East.
12. CONNECTIVITY: Never leave a room floating. Ensure every room is connected to a Hallway/Corridor or Living Room.
13. THINKING STEP: Use the `thinking_` and `verification_` fields to write out your logic before outputting the structural arrays. This improves your accuracy.

DEFAULTS (Real-world architectural standards):
- If no plot area mentioned, use 120.0 sqm
- If no facing mentioned, use NORTH
- Assign realistic standard areas: LIVING_ROOM=20-35, BEDROOM=12-15, MASTER_BEDROOM=16-24, KITCHEN=12-16, BATHROOM=4.5-6.0 (en-suite), BALCONY=4-8, CORRIDOR=4-8
- Generate unique room_id values like "room_1", "room_2", etc.
"""


# ─── JSON Extraction Helper ─────────────────────────────────────────────────


def _extract_json(content: str) -> dict:
    """Extract a JSON object from an LLM response string.
    
    Uses a brace-balanced scanner so nested objects inside strings
    don't trip up the extraction.
    """
    text = content.strip()

    # 1. Remove markdown code fences (```json ... ``` or ``` ... ```)
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    text = text.strip()

    # 2. Try direct parse first (fastest path — LLM did what it was told)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 3. Brace-balanced scanner: find the largest valid JSON object
    start = text.find('{')
    if start == -1:
        raise ParseValidationError(
            "Failed to extract JSON from LLM response — no '{' found",
            details={"raw_content": content[:600]},
        )

    depth = 0
    in_str = False
    escape = False
    end = -1
    for i, ch in enumerate(text[start:], start=start):
        if escape:
            escape = False
            continue
        if ch == '\\' and in_str:
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                end = i
                break

    if end == -1:
        raise ParseValidationError(
            "Failed to extract JSON from LLM response — unbalanced braces",
            details={"raw_content": content[:600]},
        )

    candidate = text[start:end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as e:
        raise ParseValidationError(
            f"Failed to parse extracted JSON: {e}",
            details={"raw_content": candidate[:600]},
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
            messages=[
                {"role": "user", "content": prompt},
                # Prefill forces Claude to start with `{` — guarantees JSON-only output
                {"role": "assistant", "content": "{"},
            ],
            temperature=0.0,
        )

        content = "{"
        for block in message.content:
            if block.type == "text":
                content += block.text

        if not content.strip():
            raise ClaudeConnectionError("Claude returned an empty response")

        logger.debug(f"Claude raw response (first 300 chars): {content[:300]}")
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
            messages=[
                {"role": "user", "content": prompt},
                # Prefill forces Claude to start with `{` — guarantees JSON-only output
                {"role": "assistant", "content": "{"},
            ],
            temperature=0.0,
        )

        content = "{"
        for block in message.content:
            if block.type == "text":
                content += block.text

        if not content.strip():
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
    raw_json = _enforce_layout_rules(raw_json, stated_bedroom_count)

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
            raw_json = _enforce_layout_rules(raw_json, stated_bedroom_count)
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
            raw_json = _fix_common_issues(raw_json)
            raw_json = _enforce_layout_rules(raw_json, stated_bedroom_count)
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


def _room_type_from_string(value: str) -> Optional[RoomType]:
    if not value:
        return None
    key = value.upper().replace(" ", "_").strip()
    synonyms = {
        "MASTER": "MASTER_BEDROOM",
        "MASTER_ROOM": "MASTER_BEDROOM",
        "MASTERBEDROOM": "MASTER_BEDROOM",
        "MBEDROOM": "MASTER_BEDROOM",
        "LIVING": "LIVING_ROOM",
        "HALL": "LIVING_ROOM",
        "DRAWING_ROOM": "LIVING_ROOM",
        "DINING_ROOM": "DINING",
        "POOJA_ROOM": "STUDY",
        "VERANDAH": "BALCONY",
        "WASHROOM": "BATHROOM",
        "BATH": "BATHROOM",
        "WC": "TOILET",
        "RESTROOM": "TOILET",
        "STORE_ROOM": "UTILITY",
        "WASH_AREA": "UTILITY",
        "CAR_PARKING": "GARAGE",
    }
    if key in synonyms:
        key = synonyms[key]
    if key in RoomType.__members__:
        return RoomType[key]
    return None


def _enforce_layout_rules(raw_json: dict, bedroom_count: int) -> dict:
    fixed = dict(raw_json)
    rooms_in = fixed.get("rooms", [])
    rooms: list[dict] = []
    used_ids: set[str] = set()
    next_id = 1

    def reserve_id(rid: str | None) -> str:
        nonlocal next_id
        if isinstance(rid, str):
            rid = rid.strip()
        if not rid or rid in used_ids:
            rid = f"room_{next_id}"
            next_id += 1
        else:
            m = re.match(r"room_(\d+)$", rid)
            if m:
                next_id = max(next_id, int(m.group(1)) + 1)
        used_ids.add(rid)
        return rid

    for room in rooms_in:
        if not isinstance(room, dict):
            continue
        rt = _room_type_from_string(str(room.get("room_type", "")))
        if rt is None and isinstance(room.get("label"), str):
            rt = _room_type_from_string(room["label"])
        if rt is None:
            continue

        rid = reserve_id(room.get("room_id"))
        area = room.get("target_area_sqm")
        if not isinstance(area, (int, float)):
            area = ROOM_DEFAULT_AREA.get(rt, 10.0)
        area = max(4.0, min(float(area), 200.0))

        cp = room.get("compass_preference")
        if isinstance(cp, str):
            cp_upper = cp.upper().strip()
            cp_val = cp_upper if cp_upper in {"NORTH", "SOUTH", "EAST", "WEST"} else None
        else:
            cp_val = None

        label = room.get("label") if isinstance(room.get("label"), str) else None

        rooms.append({
            "room_id": rid,
            "room_type": rt.value,
            "target_area_sqm": round(area, 1),
            "compass_preference": cp_val,
            "label": label,
        })

    def add_room(rt: RoomType, count: int = 1) -> list[str]:
        """Add room(s) and return list of new room IDs."""
        nonlocal next_id
        new_ids = []
        for _ in range(count):
            rid = f"room_{next_id}"
            rooms.append({
                "room_id": rid,
                "room_type": rt.value,
                "target_area_sqm": ROOM_DEFAULT_AREA.get(rt, 10.0),
                "compass_preference": None,
                "label": None,
            })
            new_ids.append(rid)
            next_id += 1
        return new_ids

    def count_rooms(rt: RoomType) -> int:
        return sum(1 for r in rooms if r["room_type"] == rt.value)

    def count_bathrooms() -> int:
        return sum(1 for r in rooms if r["room_type"] in {
            RoomType.BATHROOM.value, RoomType.TOILET.value
        })

    if not rooms:
        add_room(RoomType.LIVING_ROOM, 1)
        add_room(RoomType.KITCHEN, 1)
        add_room(RoomType.BEDROOM, 1)
        add_room(RoomType.BATHROOM, 1)

    if bedroom_count in BHK_TARGETS:
        targets = BHK_TARGETS[bedroom_count]
        for rt, target_count in targets.items():
            existing = count_rooms(rt)
            if rt == RoomType.BATHROOM:
                existing = count_bathrooms()
            if existing < target_count:
                add_room(rt, target_count - existing)
    elif bedroom_count > 0:
        # Minimum viable home
        if count_rooms(RoomType.LIVING_ROOM) < 1:
            add_room(RoomType.LIVING_ROOM, 1)
        if count_rooms(RoomType.KITCHEN) < 1:
            add_room(RoomType.KITCHEN, 1)
        if count_bathrooms() < 1:
            add_room(RoomType.BATHROOM, 1)

        # Ensure bedrooms count, with a master bedroom if possible
        existing_master = count_rooms(RoomType.MASTER_BEDROOM)
        existing_bed = count_rooms(RoomType.BEDROOM)
        total_bed = existing_master + existing_bed
        if bedroom_count >= 2 and existing_master == 0:
            add_room(RoomType.MASTER_BEDROOM, 1)
            total_bed += 1
        if total_bed < bedroom_count:
            add_room(RoomType.BEDROOM, bedroom_count - total_bed)

        # Add corridor for 2+ bedrooms
        if bedroom_count >= 2 and count_rooms(RoomType.CORRIDOR) < 1:
            add_room(RoomType.CORRIDOR, 1)

    # Rebuild adjacency constraints with valid room ids
    valid_ids = {r["room_id"] for r in rooms}
    edges_in = fixed.get("adjacency_constraints", [])
    edges: list[dict] = []
    for edge in edges_in:
        if not isinstance(edge, dict):
            continue
        a = edge.get("room_a_id")
        b = edge.get("room_b_id")
        if a not in valid_ids or b not in valid_ids or a == b:
            continue
        ct = edge.get("connection_type")
        if isinstance(ct, str):
            ct_upper = ct.upper().strip()
            ct_val = ct_upper if ct_upper in ConnectionType.__members__ else "OPENING"
        else:
            ct_val = "OPENING"
        edges.append({
            "room_a_id": a,
            "room_b_id": b,
            "connection_type": ct_val,
            "required": bool(edge.get("required", True)),
        })

    def edge_exists(a: str, b: str, ct: str | None = None) -> bool:
        for e in edges:
            if {e["room_a_id"], e["room_b_id"]} == {a, b}:
                if ct is None or e["connection_type"] == ct:
                    return True
        return False

    def any_edge_exists(a: str, b: str) -> bool:
        return edge_exists(a, b, None)

    def first_room_id(rt: RoomType) -> Optional[str]:
        for r in rooms:
            if r["room_type"] == rt.value:
                return r["room_id"]
        return None

    # ── RULE: Every bedroom/master_bedroom must have a DOOR to a bathroom ──
    bedroom_types = {RoomType.BEDROOM.value, RoomType.MASTER_BEDROOM.value}
    bathroom_types = {RoomType.BATHROOM.value, RoomType.TOILET.value}

    # Collect bedroom IDs and bathroom IDs
    bedroom_ids = [r["room_id"] for r in rooms if r["room_type"] in bedroom_types]
    bathroom_ids = [r["room_id"] for r in rooms if r["room_type"] in bathroom_types]

    # Track which bathrooms are already "claimed" by a bedroom
    claimed_bathrooms: set[str] = set()
    for e in edges:
        if e["connection_type"] == "DOOR":
            a_type = next((r["room_type"] for r in rooms if r["room_id"] == e["room_a_id"]), None)
            b_type = next((r["room_type"] for r in rooms if r["room_id"] == e["room_b_id"]), None)
            if a_type in bedroom_types and b_type in bathroom_types:
                claimed_bathrooms.add(e["room_b_id"])
            elif b_type in bedroom_types and a_type in bathroom_types:
                claimed_bathrooms.add(e["room_a_id"])

    # Assign one bathroom per bedroom
    for bed_id in bedroom_ids:
        # Check if this bedroom already has a bathroom DOOR edge
        has_bath = False
        for e in edges:
            if e["connection_type"] == "DOOR" and bed_id in {e["room_a_id"], e["room_b_id"]}:
                other = e["room_b_id"] if e["room_a_id"] == bed_id else e["room_a_id"]
                other_type = next((r["room_type"] for r in rooms if r["room_id"] == other), None)
                if other_type in bathroom_types:
                    has_bath = True
                    break
        if has_bath:
            continue

        # Find an unclaimed bathroom
        assigned = False
        for bath_id in bathroom_ids:
            if bath_id not in claimed_bathrooms:
                edges.append({
                    "room_a_id": bed_id,
                    "room_b_id": bath_id,
                    "connection_type": "DOOR",
                    "required": True,
                })
                claimed_bathrooms.add(bath_id)
                assigned = True
                break

        # If no unclaimed bathroom, create a new one
        if not assigned:
            new_ids = add_room(RoomType.BATHROOM, 1)
            new_bath_id = new_ids[0]
            bathroom_ids.append(new_bath_id)
            edges.append({
                "room_a_id": bed_id,
                "room_b_id": new_bath_id,
                "connection_type": "DOOR",
                "required": True,
            })
            claimed_bathrooms.add(new_bath_id)

    # ── RULE: Living room ↔ Kitchen (OPENING) ──
    living_id = first_room_id(RoomType.LIVING_ROOM)
    kitchen_id = first_room_id(RoomType.KITCHEN)
    dining_id = first_room_id(RoomType.DINING)

    if living_id and kitchen_id and not edge_exists(living_id, kitchen_id, "OPENING"):
        edges.append({
            "room_a_id": living_id,
            "room_b_id": kitchen_id,
            "connection_type": "OPENING",
            "required": True,
        })
    if kitchen_id and dining_id and not edge_exists(kitchen_id, dining_id, "OPENING"):
        edges.append({
            "room_a_id": kitchen_id,
            "room_b_id": dining_id,
            "connection_type": "OPENING",
            "required": True,
        })

    # ── RULE: Corridor connects to all major rooms (not bathrooms) ──
    corridor_id = first_room_id(RoomType.CORRIDOR)
    living_id = first_room_id(RoomType.LIVING_ROOM)
    if corridor_id:
        corridor_connectable_types = {
            RoomType.LIVING_ROOM.value, RoomType.KITCHEN.value,
            RoomType.MASTER_BEDROOM.value, RoomType.BEDROOM.value,
            RoomType.STUDY.value, RoomType.DINING.value,
            RoomType.BALCONY.value, RoomType.UTILITY.value,
            RoomType.GARAGE.value,
        }
        for r in rooms:
            if r["room_type"] in corridor_connectable_types and r["room_id"] != corridor_id:
                if not any_edge_exists(corridor_id, r["room_id"]):
                    edges.append({
                        "room_a_id": corridor_id,
                        "room_b_id": r["room_id"],
                        "connection_type": "DOOR",
                        "required": True,
                    })

    # ── RULE: Connectivity Guarantee ──
    # Ensure every room has at least one edge (to prevent isolated floating rooms).
    # We connect isolated rooms to a 'hub' (Corridor, or Living Room, or the first room)
    hub_id = corridor_id or living_id or (rooms[0]["room_id"] if rooms else None)
    if hub_id:
        for r in rooms:
            if r["room_id"] == hub_id:
                continue
            # Check if this room is part of any edge
            has_edge = any(e["room_a_id"] == r["room_id"] or e["room_b_id"] == r["room_id"] for e in edges)
            if not has_edge:
                # Force a connection
                edges.append({
                    "room_a_id": hub_id,
                    "room_b_id": r["room_id"],
                    "connection_type": "DOOR",
                    "required": True,
                })

    # Plot area sanity
    total_area = sum(float(r["target_area_sqm"]) for r in rooms) if rooms else 0.0
    plot = fixed.get("plot_area_sqm")
    if not isinstance(plot, (int, float)):
        plot = 100.0
    min_plot = max(30.0, total_area * 1.1)
    plot = max(float(plot), min_plot)
    plot = min(plot, 2000.0)

    # Scale room areas proportionally to plot size
    rooms = _scale_areas_to_plot(rooms, plot)

    fixed["rooms"] = rooms
    fixed["adjacency_constraints"] = edges
    fixed["plot_area_sqm"] = round(plot, 1)

    if not isinstance(fixed.get("style_hints"), list):
        fixed["style_hints"] = []

    return fixed


def _scale_areas_to_plot(rooms: list[dict], plot_sqm: float) -> list[dict]:
    """Scale room areas proportionally so they fill ~85-92% of the plot.

    This ensures rooms are right-sized for both small and large plots.
    Default room areas in ROOM_DEFAULT_AREA assume a ~100 sqm plot.
    """
    if not rooms or plot_sqm <= 0:
        return rooms

    total_room_area = sum(float(r.get("target_area_sqm", 10.0)) for r in rooms)
    if total_room_area <= 0:
        return rooms

    # Target: rooms should fill 88% of the plot (rest is walls)
    target_total = plot_sqm * 0.88
    scale = target_total / total_room_area

    # Only scale if significantly off (>15% difference)
    if 0.85 <= scale <= 1.15:
        return rooms

    # Clamp scale factor to avoid extreme sizing
    scale = max(0.5, min(scale, 3.0))

    result = []
    for r in rooms:
        r_copy = dict(r)
        area = float(r_copy.get("target_area_sqm", 10.0))
        scaled = area * scale
        # Enforce min/max per room
        scaled = max(3.0, min(scaled, 200.0))
        r_copy["target_area_sqm"] = round(scaled, 1)
        result.append(r_copy)

    return result


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
