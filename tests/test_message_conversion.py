"""Check the message boundary without loading model weights."""

import json

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from sellerops.agent.message_conversion import (
    action_to_ai_message,
    model_output_to_ai_message,
    observation_to_tool_message,
    tool_call_to_action,
)
from sellerops.schemas import SchemaError
from sellerops.services.episode_service.episode_service import EpisodeService


def inventory_action():
    return {"type": "tool_call", "tool": "get_inventory", "arguments": {"product_id": "P1001"}}


def test_tool_request_round_trip_through_real_episode():
    action = inventory_action()
    message = model_output_to_ai_message(json.dumps(action), tool_call_id="call_demo")
    assert isinstance(message, AIMessage)
    assert json.loads(message.content) == action
    call = message.tool_calls[0]
    assert tool_call_to_action(call) == action
    service = EpisodeService()
    episode = service.create_episode("message-demo")
    observation = service.execute_tool(episode.session_id, tool_call_to_action(call))
    reply = observation_to_tool_message(observation, tool_call=call)
    assert isinstance(reply, ToolMessage)
    assert reply.tool_call_id == "call_demo"
    assert reply.name == "get_inventory"
    assert reply.status == "success"
    assert json.loads(reply.content) == observation


@pytest.mark.parametrize("action", [
    {"type": "final", "status": "success", "message": "Stok tersedia.", "data": {"available_stock": 7}},
    {"type": "ask_user", "question": "Berapa anggarannya?", "missing_fields": ["budget_idr"]},
])
def test_terminal_actions_preserve_structured_fields(action):
    message = action_to_ai_message(action)
    assert message.tool_calls == []
    assert json.loads(message.content) == action


def test_tool_error_preserves_envelope_and_call_id():
    call = action_to_ai_message(inventory_action()).tool_calls[0]
    observation = {
        "tool": "get_inventory", "ok": False, "result": None,
        "error": {"code": "product_not_found", "message": "Unknown product"},
    }
    reply = observation_to_tool_message(observation, tool_call=call)
    assert reply.status == "error"
    assert reply.tool_call_id == call["id"]
    assert json.loads(reply.content) == observation


@pytest.mark.parametrize("text", [
    "not JSON", '```json\n{}\n```', "[]", "null", "{}",
    '{"type":"tool_call","tool":"delete_product","arguments":{}}',
    '{"type":"tool_call","tool":"get_inventory","arguments":{}}',
])
def test_invalid_output_is_rejected(text):
    with pytest.raises(SchemaError):
        model_output_to_ai_message(text)


def test_call_ids_are_unique_and_arguments_are_independent():
    action = inventory_action()
    first = action_to_ai_message(action)
    second = action_to_ai_message(action)
    assert first.tool_calls[0]["id"] != second.tool_calls[0]["id"]
    restored = tool_call_to_action(first.tool_calls[0])
    restored["arguments"]["product_id"] = "changed"
    action["arguments"]["product_id"] = "also changed"
    assert first.tool_calls[0]["args"]["product_id"] == "P1001"


def test_mismatched_observation_is_rejected():
    call = action_to_ai_message(inventory_action()).tool_calls[0]
    with pytest.raises(SchemaError, match="does not match"):
        observation_to_tool_message(
            {"tool": "get_discount_policy", "ok": True, "result": {}, "error": None},
            tool_call=call,
        )


@pytest.mark.parametrize("call_id", ["", " ", 123])
def test_invalid_call_id_is_rejected(call_id):
    with pytest.raises(SchemaError):
        action_to_ai_message(inventory_action(), tool_call_id=call_id)
