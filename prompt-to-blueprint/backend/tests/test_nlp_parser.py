"""Integration tests for the NLP parser service (mocked Claude + Ollama)."""

import json
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from app.services.nlp_parser import (
    parse_prompt,
    validate_and_repair,
    call_ollama,
    call_claude,
    call_llm,
    build_system_prompt,
    _extract_bedroom_count,
    _extract_json,
    OllamaConnectionError,
    ClaudeConnectionError,
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


# ─── Helper: mock Ollama httpx response ──────────────────────────────────────


def _mock_ollama_response(data: dict) -> MagicMock:
    """Create a mock httpx response that mimics Ollama."""
    mock = MagicMock()
    mock.status_code = 200
    mock.json.return_value = {
        "message": {"content": json.dumps(data)}
    }
    mock.raise_for_status = MagicMock()
    return mock


# ─── Helper: mock Claude message response ───────────────────────────────────


def _mock_claude_message(data: dict) -> MagicMock:
    """Create a mock anthropic Message that mimics Claude response."""
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = json.dumps(data)

    message = MagicMock()
    message.content = [text_block]
    return message


# ─── Tests: Ollama path (existing) ──────────────────────────────────────────


def test_parse_3bhk_via_ollama():
    """Test parsing a 3BHK prompt via Ollama with mocked response."""
    with patch("app.services.nlp_parser.settings") as mock_settings:
        mock_settings.nlp_provider = "ollama"
        mock_settings.anthropic_api_key = ""
        mock_settings.ollama_host = "http://localhost:11434"
        mock_settings.ollama_model = "qwen2.5:1.5b"
        mock_settings.ollama_timeout = 30

        with patch("app.services.nlp_parser.httpx.post",
                    return_value=_mock_ollama_response(MOCK_3BHK_RESPONSE)):
            result = parse_prompt("Design a 3BHK apartment with modern style")

    assert len(result.rooms) == 7
    bedroom_types = {RoomType.BEDROOM, RoomType.MASTER_BEDROOM}
    bedrooms = [r for r in result.rooms if r.room_type in bedroom_types]
    assert len(bedrooms) == 3
    assert result.facing == CompassFacing.NORTH
    assert result.plot_area_sqm == 120.0


def test_ollama_unreachable():
    """Test that OllamaConnectionError is raised when server is unreachable."""
    import httpx as httpx_mod

    with patch("app.services.nlp_parser.settings") as mock_settings:
        mock_settings.nlp_provider = "ollama"
        mock_settings.anthropic_api_key = ""
        mock_settings.ollama_host = "http://localhost:11434"
        mock_settings.ollama_model = "qwen2.5:1.5b"
        mock_settings.ollama_timeout = 30

        with patch("app.services.nlp_parser.httpx.post",
                    side_effect=httpx_mod.ConnectError("Connection refused")):
            with pytest.raises(OllamaConnectionError):
                parse_prompt("Build a 2BHK house")


# ─── Tests: Claude path ─────────────────────────────────────────────────────


def test_parse_3bhk_via_claude():
    """Test parsing a 3BHK prompt via Claude with mocked Anthropic client."""
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _mock_claude_message(MOCK_3BHK_RESPONSE)

    with patch("app.services.nlp_parser.settings") as mock_settings:
        mock_settings.nlp_provider = "claude"
        mock_settings.anthropic_api_key = "sk-test-key"
        mock_settings.claude_model = "claude-sonnet-4-20250514"
        mock_settings.claude_max_tokens = 2048

        with patch("anthropic.Anthropic", return_value=mock_client):
            result = parse_prompt("Design a 3BHK apartment with modern style")

    assert len(result.rooms) == 7
    bedroom_types = {RoomType.BEDROOM, RoomType.MASTER_BEDROOM}
    bedrooms = [r for r in result.rooms if r.room_type in bedroom_types]
    assert len(bedrooms) == 3
    assert result.facing == CompassFacing.NORTH


def test_claude_fallback_to_ollama():
    """Test that when Claude fails, it falls back to Ollama."""
    with patch("app.services.nlp_parser.settings") as mock_settings:
        mock_settings.nlp_provider = "claude"
        mock_settings.anthropic_api_key = "sk-test-key"
        mock_settings.claude_model = "claude-sonnet-4-20250514"
        mock_settings.claude_max_tokens = 2048
        mock_settings.ollama_host = "http://localhost:11434"
        mock_settings.ollama_model = "qwen2.5:1.5b"
        mock_settings.ollama_timeout = 30

        with patch("app.services.nlp_parser.call_claude",
                    side_effect=ClaudeConnectionError("API key invalid")):
            with patch("app.services.nlp_parser.httpx.post",
                        return_value=_mock_ollama_response(MOCK_3BHK_RESPONSE)):
                result = parse_prompt("Design a 3BHK apartment")

    assert len(result.rooms) == 7


def test_no_api_key_uses_ollama_directly():
    """Test that when no Claude API key is set, Ollama is used directly."""
    with patch("app.services.nlp_parser.settings") as mock_settings:
        mock_settings.nlp_provider = "claude"
        mock_settings.anthropic_api_key = ""  # No key
        mock_settings.ollama_host = "http://localhost:11434"
        mock_settings.ollama_model = "qwen2.5:1.5b"
        mock_settings.ollama_timeout = 30

        with patch("app.services.nlp_parser.call_claude") as mock_claude:
            with patch("app.services.nlp_parser.httpx.post",
                        return_value=_mock_ollama_response(MOCK_3BHK_RESPONSE)):
                result = parse_prompt("Design a 2BHK apartment")

    # Claude should NOT have been called
    mock_claude.assert_not_called()
    assert len(result.rooms) == 7


# ─── Tests: Repair logic ────────────────────────────────────────────────────


def test_repair_pass_triggered():
    """Test that repair pass is triggered when bedroom count is under."""
    with patch("app.services.nlp_parser.settings") as mock_settings:
        mock_settings.nlp_provider = "ollama"
        mock_settings.anthropic_api_key = ""
        mock_settings.ollama_host = "http://localhost:11434"
        mock_settings.ollama_model = "qwen2.5:1.5b"
        mock_settings.ollama_timeout = 30

        with patch("app.services.nlp_parser.httpx.post", side_effect=[
            _mock_ollama_response(MOCK_UNDERCOUNTED_RESPONSE),
            _mock_ollama_response(MOCK_3BHK_RESPONSE),
        ]):
            result = parse_prompt("Create a 3BHK flat with open kitchen")

    bedroom_types = {RoomType.BEDROOM, RoomType.MASTER_BEDROOM}
    bedrooms = [r for r in result.rooms if r.room_type in bedroom_types]
    assert len(bedrooms) >= 3


# ─── Tests: Utilities ───────────────────────────────────────────────────────


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


def test_extract_json_plain():
    """Test JSON extraction from plain JSON string."""
    data = {"key": "value"}
    result = _extract_json(json.dumps(data))
    assert result == data


def test_extract_json_with_code_fences():
    """Test JSON extraction from markdown code-fenced response."""
    data = {"rooms": []}
    content = f"```json\n{json.dumps(data)}\n```"
    result = _extract_json(content)
    assert result == data


def test_extract_json_failure():
    """Test that ParseValidationError is raised on invalid JSON."""
    with pytest.raises(ParseValidationError):
        _extract_json("This is not JSON at all")
