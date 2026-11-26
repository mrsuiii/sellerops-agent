"""Eval terminal tools (final / ask_user) and their to_action protocol."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from sellerops.evaluation.terminal_tools import TERMINAL_TOOL_NAMES, AskUser, FinalAnswer, terminal_action


def test_terminal_action_validates() -> None:
    assert TERMINAL_TOOL_NAMES == {"final", "ask_user"}
    action = terminal_action("final", {"message": "Omzet Rp1 jt.", "data": {"revenue_idr": 1000000}})
    assert action == {"type": "final", "message": "Omzet Rp1 jt.", "data": {"revenue_idr": 1000000}}
    assert terminal_action("ask_user", {"question": "Periode apa?"}) == {"type": "ask_user", "question": "Periode apa?"}
    with pytest.raises(ValidationError):
        terminal_action("final", {"message": "", "data": {}})
    with pytest.raises(ValidationError):
        terminal_action("final", {"message": "ok", "data": "x"})
    with pytest.raises(ValidationError):
        terminal_action("ask_user", {"question": ""})
    with pytest.raises(ValueError):
        terminal_action("get_product", {})


def test_tool_names_are_the_titles() -> None:
    assert FinalAnswer.model_config["title"] == "final"
    assert AskUser.model_config["title"] == "ask_user"


@pytest.mark.parametrize("data", [
    {"product_ids": ["FS186", "FS188"], "cause": "out_of_stock"},       # RCA-5: cause needs one product_id
    {"product_id": ["FS186"], "cause": "out_of_stock"},
    {"product_id": "FS186", "cause": "stock habis"},
    {"complaint": "kekecilan"},
    {"product_ids": "FS1,FS2"},
])
def test_final_rejects_malformed_data(data) -> None:
    with pytest.raises(ValidationError):
        FinalAnswer.model_validate({"message": "m", "data": data})


@pytest.mark.parametrize("data", [
    {"product_id": "FS186", "cause": "out_of_stock"},
    {"product_id": "FS188", "complaint": "ukuran kekecilan"},
    {"period": "last_30_days", "product_ids": ["FS1", "FS2"]},
    {"period": "last_30_days", "revenue_idr": 1, "orders": 2},
    {},
])
def test_final_accepts_well_formed_data(data) -> None:
    assert FinalAnswer.model_validate({"message": "m", "data": data}).to_action()["data"] == data
