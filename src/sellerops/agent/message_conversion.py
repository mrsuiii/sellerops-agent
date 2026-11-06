"""Bridge validated SellerOps JSON and LangChain messages without running tools.

AIMessage.content retains canonical action JSON, including tool calls, so the
local model can receive the same assistant content format used during SFT.
Only decoded completion text (not the prompt) belongs in model_output_to_ai_message.
"""

from copy import deepcopy
import json
from typing import Any
from uuid import uuid4

from langchain_core.messages import AIMessage, ToolMessage

from sellerops.schemas import (
    SchemaError,
    canonical_json,
    validate_action,
    validate_tool_observation,
)


def _require_call_id(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SchemaError("tool_call_id must be a non-empty string")
    return value


def action_to_ai_message(
    action: dict[str, Any], *, tool_call_id: str | None = None
) -> AIMessage:
    """Validate an action and expose tool requests through AIMessage.tool_calls.

    Final answers and clarification requests have no tool calls; their complete
    structured action is available via json.loads(message.content). An optional
    call ID supports reproducible examples; otherwise each call gets a fresh ID.
    """
    validate_action(action)
    content = canonical_json(action)
    if action["type"] != "tool_call":
        if tool_call_id is not None:
            raise SchemaError("tool_call_id is only valid for tool_call actions")
        return AIMessage(content=content)
    call_id = _require_call_id(
        tool_call_id if tool_call_id is not None else f"call_{uuid4().hex}"
    )
    return AIMessage(
        content=content,
        tool_calls=[{
            "name": action["tool"],
            "args": deepcopy(action["arguments"]),
            "id": call_id,
            "type": "tool_call",
        }],
    )


def model_output_to_ai_message(
    text: str, *, tool_call_id: str | None = None
) -> AIMessage:
    """Parse one JSON action; reject prose, code fences, and invalid schemas.

    JSON/schema errors become SchemaError for the caller's retry logic. This
    function does not repair output, run generation, or execute tools.
    """
    if not isinstance(text, str):
        raise SchemaError("model output must be a JSON string")
    try:
        action = json.loads(text)
    except json.JSONDecodeError as error:
        raise SchemaError(f"model output is not valid JSON: {error.msg}") from error
    return action_to_ai_message(action, tool_call_id=tool_call_id)


def tool_call_to_action(tool_call: dict[str, Any]) -> dict[str, Any]:
    """Convert a normalized LangChain tool call for EpisodeService.execute_tool."""
    if not isinstance(tool_call, dict) or set(tool_call) != {"name", "args", "id", "type"}:
        raise SchemaError("expected a normalized LangChain tool call")
    if tool_call["type"] != "tool_call":
        raise SchemaError("expected type=tool_call")
    _require_call_id(tool_call["id"])
    action = {
        "type": "tool_call",
        "tool": tool_call["name"],
        "arguments": deepcopy(tool_call["args"]),
    }
    validate_action(action)
    return action


def observation_to_tool_message(
    observation: dict[str, Any], *, tool_call: dict[str, Any]
) -> ToolMessage:
    """Keep the full success/error envelope and link it to its originating call.

    The caller must pass the call that produced this observation. Observations
    contain a tool name but no call ID, so argument identity cannot be checked.
    """
    action = tool_call_to_action(tool_call)
    validate_tool_observation(observation)
    if observation["tool"] != action["tool"]:
        raise SchemaError("observation tool does not match the requested tool")
    return ToolMessage(
        content=canonical_json(observation),
        name=observation["tool"],
        tool_call_id=tool_call["id"],
        status="success" if observation["ok"] else "error",
    )
