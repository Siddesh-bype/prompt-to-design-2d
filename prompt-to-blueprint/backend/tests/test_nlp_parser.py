"""Integration tests for the NLP parser service (mocked Ollama)."""

import json
import pytest
from unittest.mock import patch, MagicMock

from app.services.nlp_parser import (
    parse_prompt,
    validate_and_repair,
    call_ollama,
    build_system_prompt,
    _extract_bedroom_count,
    OllamaConnectionError,
    ParseValidationError,
)
from app.models.schemas import RoomType, CompassFacing


# ─── Fixtures ────────────────────────────────────────────────────────────────


MOCK_3BHK_RESPONSE = {
    "rooms": [
        {"room_id": "room_1", "room_type": "LIVING_ROOM", "target_area_sqm": 25.0},
        {"room_id": "room_2", "room_type": "MASTER_BEDROOM", "target_area_sqm": 18.0,
         "compass_preference": "SOUTH"},
        {"room_id": "room_3", "room_type": "BEDROOM", "target_area_sqm": 14.0},
        {"room_id": "room_4", "room_type": "BEDROOM", "target_area_sqm": 12.0},
        {"room_id": "room_5", "room_type": "KITCHEN", "target_area_sqm": 10.0},
        {"room_id": "room_6", "room_type": "BATHROOM", "target_area_sqm": 5.0},
        {"room_id": "room_7", "room_type": "BATHROOM", "target_area_sqm": 4.5},
    ],
    "plot_area_sqm": 120.0,
    "facing": "NORTH",
    "style_hints": ["modern"],
    "adjacency_constraints": [
        {"room_a_id": "room_2", "room_b_id": "room_6",
         "connection_type": "DOOR", "required": True},
    ],
    "vastu_enabled": False,
}

MOCK_UNDERCOUNTED_RESPONSE = {
    "rooms": [
        {"room_id": "room_1", "room_type": "LIVING_ROOM", "target_area_sqm": 25.0},
        {"room_id": "room_2", "room_type": "MASTER_BEDROOM", "target_area_sqm": 18.0},
        {"room_id": "room_3", "room_type": "KITCHEN", "target_area_sqm": 10.0},
        {"room_id": "room_4", "room_type": "BATHROOM", "target_area_sqm": 5.0},
    ],
    "plot_area_sqm": 120.0,
    "facing": "NORTH",
    "adjacency_constraints": [],
}


# ─── Tests ───────────────────────────────────────────────────────────────────


def test_parse_3bhk_prompt():
    """Test parsing a 3BHK prompt with mocked Ollama response."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "message": {"content": json.dumps(MOCK_3BHK_RESPONSE)}
    }
    mock_response.raise_for_status = MagicMock()

    with patch("app.services.nlp_parser.httpx.post", return_value=mock_response):
        result = parse_prompt("Design a 3BHK apartment with modern style")

    assert len(result.rooms) == 7
    bedroom_types = {RoomType.BEDROOM, RoomType.MASTER_BEDROOM}
    bedrooms = [r for r in result.rooms if r.room_type in bedroom_types]
    assert len(bedrooms) == 3
    assert result.facing == CompassFacing.NORTH
    assert result.plot_area_sqm == 120.0


def test_repair_pass_triggered():
    """Test that repair pass is triggered when bedroom count is under."""
    # First call returns undercounted, second call returns corrected
    mock_response_1 = MagicMock()
    mock_response_1.status_code = 200
    mock_response_1.json.return_value = {
        "message": {"content": json.dumps(MOCK_UNDERCOUNTED_RESPONSE)}
    }
    mock_response_1.raise_for_status = MagicMock()

    mock_response_2 = MagicMock()
    mock_response_2.status_code = 200
    mock_response_2.json.return_value = {
        "message": {"content": json.dumps(MOCK_3BHK_RESPONSE)}
    }
    mock_response_2.raise_for_status = MagicMock()

    with patch(
        "app.services.nlp_parser.httpx.post",
        side_effect=[mock_response_1, mock_response_2],
    ):
        result = parse_prompt("Create a 3BHK flat with open kitchen")

    bedroom_types = {RoomType.BEDROOM, RoomType.MASTER_BEDROOM}
    bedrooms = [r for r in result.rooms if r.room_type in bedroom_types]
    assert len(bedrooms) >= 3


def test_ollama_unreachable():
    """Test that OllamaConnectionError is raised when server is unreachable."""
    import httpx as httpx_mod

    with patch(
        "app.services.nlp_parser.httpx.post",
        side_effect=httpx_mod.ConnectError("Connection refused"),
    ):
        with pytest.raises(OllamaConnectionError):
            parse_prompt("Build a 2BHK house")


def test_extract_bedroom_count():
    """Test bedroom count extraction from various prompt formats."""
    assert _extract_bedroom_count("3BHK apartment") == 3
    assert _extract_bedroom_count("2 BHK flat") == 2
    assert _extract_bedroom_count("4-BHK house") == 4
    assert _extract_bedroom_count("3 bedroom house") == 3
    assert _extract_bedroom_count("a house with garden") == 0


def test_build_system_prompt():
    """Test that system prompt contains key instructions."""
    prompt = build_system_prompt()
    assert "JSON" in prompt
    assert "ROOM_TYPE" in prompt
    assert "BHK" in prompt
    assert "Pooja" in prompt
    assert "LIVING_ROOM" in prompt
