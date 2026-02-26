"""Tests for the NLP parser service (Claude/OpenRouter version)."""

import json
from unittest.mock import patch

import pytest

from app.models.schemas import RoomType
from app.services.nlp_parser import (
    ClaudeConnectionError,
    _extract_bedroom_count,
    _extract_json,
    call_llm,
    parse_prompt,
    validate_and_repair,
)


def _minimal_home_json() -> dict:
    return {
        "rooms": [
            {"room_id": "room_1", "room_type": "LIVING_ROOM", "target_area_sqm": 24.0},
            {"room_id": "room_2", "room_type": "KITCHEN", "target_area_sqm": 10.0},
            {"room_id": "room_3", "room_type": "BEDROOM", "target_area_sqm": 14.0},
            {"room_id": "room_4", "room_type": "BATHROOM", "target_area_sqm": 5.0},
        ],
        "plot_area_sqm": 100.0,
        "facing": "NORTH",
        "style_hints": [],
        "adjacency_constraints": [],
        "vastu_enabled": False,
    }


def test_extract_bedroom_count():
    assert _extract_bedroom_count("3BHK apartment") == 3
    assert _extract_bedroom_count("2 BHK flat") == 2
    assert _extract_bedroom_count("4-BHK house") == 4
    assert _extract_bedroom_count("3 bedroom house") == 3
    assert _extract_bedroom_count("simple studio") == 0


def test_extract_json_with_code_fences():
    data = {"rooms": []}
    content = f"```json\n{json.dumps(data)}\n```"
    result = _extract_json(content)
    assert result == data


def test_extract_json_failure():
    with pytest.raises(Exception):
        _extract_json("not json")


def test_validate_and_repair_enforces_bhk():
    # Missing bedrooms/bathrooms for 3BHK should be auto-repaired.
    raw = {
        "rooms": [
            {"room_id": "room_1", "room_type": "LIVING_ROOM", "target_area_sqm": 22.0},
            {"room_id": "room_2", "room_type": "KITCHEN", "target_area_sqm": 10.0},
            {"room_id": "room_3", "room_type": "MASTER_BEDROOM", "target_area_sqm": 18.0},
        ],
        "plot_area_sqm": 100.0,
        "facing": "NORTH",
        "adjacency_constraints": [],
    }
    parsed = validate_and_repair(raw, "Design 3BHK", stated_bedroom_count=3)
    bedroom_count = sum(
        1 for r in parsed.rooms if r.room_type in {RoomType.BEDROOM, RoomType.MASTER_BEDROOM}
    )
    bathroom_count = sum(1 for r in parsed.rooms if r.room_type in {RoomType.BATHROOM, RoomType.TOILET})

    assert bedroom_count >= 3
    assert bathroom_count >= 2
    assert any(r.room_type == RoomType.LIVING_ROOM for r in parsed.rooms)
    assert any(r.room_type == RoomType.KITCHEN for r in parsed.rooms)


def test_parse_prompt_uses_call_llm():
    mock_out = _minimal_home_json()
    with patch("app.services.nlp_parser.call_llm", return_value=mock_out):
        parsed = parse_prompt("Design a compact 1BHK")
    assert len(parsed.rooms) >= 4


def test_call_llm_prefers_openrouter_when_only_openrouter_key():
    with patch("app.services.nlp_parser.settings") as s:
        s.anthropic_api_key = ""
        s.openrouter_api_key = "or-key"
        s.openrouter_model = "test-model"
        with patch("app.services.nlp_parser.call_openrouter", return_value=_minimal_home_json()) as m:
            out = call_llm("prompt", "system")
    assert "rooms" in out
    m.assert_called_once()


def test_call_llm_no_provider_configured():
    with patch("app.services.nlp_parser.settings") as s:
        s.anthropic_api_key = ""
        s.openrouter_api_key = ""
        with pytest.raises(ClaudeConnectionError):
            call_llm("prompt", "system")
