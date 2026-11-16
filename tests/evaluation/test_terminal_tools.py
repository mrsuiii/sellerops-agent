"""Eval terminal tools (final / ask_user) and their to_action protocol."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from sellerops.evaluation.terminal_tools import TERMINAL_TOOL_NAMES, AskUser, FinalAnswer, terminal_action
from sellerops.schemas import SchemaError


def test_terminal_action_validates() -> None:
    assert TERMINAL_TOOL_NAMES == {"final", "ask_user"}
    action = terminal_action("final", {"status": "success", "message": "ok", "data": {"products": []}})
    assert action == {"type": "final", "status": "success", "message": "ok", "data": {"products": []}}
    assert terminal_action("ask_user", {"question": "Ukuran?", "missing_fields": ["size"]})["type"] == "ask_user"
    with pytest.raises(ValidationError):
        terminal_action("final", {"status": "done", "message": "ok", "data": {}})
    with pytest.raises((ValidationError, SchemaError)):
        terminal_action("ask_user", {"question": "?", "missing_fields": []})


def test_to_action_protocol() -> None:
    final = FinalAnswer.model_validate({"status": "not_found", "message": "Tidak ada.", "data": {"products": []}})
    assert final.to_action()["type"] == "final"
    assert AskUser.model_validate({"question": "Merek?", "missing_fields": ["brand"]}).to_action()["type"] == "ask_user"
    with pytest.raises(ValueError):
        terminal_action("search_products", {})
