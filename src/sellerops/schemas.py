"""Strict, dependency-free schemas for SellerOps actions and datasets."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

ACTION_TYPES = {"tool_call", "ask_user", "final"}
TOOL_NAMES = {
    "search_products",
    "get_inventory",
    "get_discount_policy",
    "calculate_offer",
}
SPLITS = {"train", "dev", "test"}
FAMILIES = {"product_search", "inventory_offer", "policy_qa"}
MAX_AGENT_STEPS = 8
MAX_AGENT_RETRIES = 2


class SchemaError(ValueError):
    """Raised when generated or model-produced data violates a contract."""


def canonical_json(value: Any) -> str:
    """Serialize a value as stable compact JSON for model-facing records.

    Args:
        value: JSON-serializable Python value to encode.

    Returns:
        UTF-8-compatible JSON text with sorted keys and no unnecessary spaces.
    """

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    """Require an object-like value and return it for further validation.

    Args:
        value: Candidate value to check.
        name: Field path used in the validation error.

    Returns:
        The input value typed as a string-keyed mapping.

    Raises:
        SchemaError: If ``value`` is not a mapping.
    """
    if not isinstance(value, Mapping):
        raise SchemaError(f"{name} must be an object")
    return value


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    """Require that a mapping has exactly the expected field names.

    Args:
        value: Mapping whose keys are checked.
        expected: Complete allowed key set.
        name: Field path used in the validation error.

    Returns:
        ``None`` when the keys match exactly.

    Raises:
        SchemaError: If any key is missing or unexpected.
    """
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise SchemaError(f"{name} keys mismatch; missing={missing}, extra={extra}")


def _require_nonempty_string(value: Any, name: str) -> str:
    """Require a non-blank string and return it unchanged.

    Args:
        value: Candidate value to check.
        name: Field path used in the validation error.

    Returns:
        The validated string.

    Raises:
        SchemaError: If the value is not a non-empty string.
    """
    if not isinstance(value, str) or not value.strip():
        raise SchemaError(f"{name} must be a non-empty string")
    return value


def _require_int(value: Any, name: str, minimum: int | None = None) -> int:
    """Require an integer, optionally enforcing a lower bound.

    Args:
        value: Candidate numeric value; booleans are rejected explicitly.
        name: Field path used in the validation error.
        minimum: Inclusive lower bound, when one is required.

    Returns:
        The validated integer.

    Raises:
        SchemaError: If the type or lower-bound constraint is violated.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise SchemaError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise SchemaError(f"{name} must be >= {minimum}")
    return value


def validate_action(action: Any) -> None:
    """Validate one tagged assistant action with strict per-type fields.

    Args:
        action: Candidate ``tool_call``, ``ask_user``, or ``final`` object.

    Returns:
        ``None`` when every field and nested argument satisfies the contract.

    Raises:
        SchemaError: If the action type, fields, or tool arguments are invalid.
    """

    action = _require_mapping(action, "action")
    action_type = action.get("type")
    if action_type not in ACTION_TYPES:
        raise SchemaError(f"unsupported action type: {action_type!r}")

    if action_type == "tool_call":
        _require_exact_keys(action, {"type", "tool", "arguments"}, "tool_call")
        tool = action["tool"]
        if tool not in TOOL_NAMES:
            raise SchemaError(f"unsupported tool: {tool!r}")
        arguments = _require_mapping(action["arguments"], "tool_call.arguments")
        _validate_tool_arguments(tool, arguments)
        return

    if action_type == "ask_user":
        _require_exact_keys(action, {"type", "question", "missing_fields"}, "ask_user")
        _require_nonempty_string(action["question"], "ask_user.question")
        missing_fields = action["missing_fields"]
        if (
            not isinstance(missing_fields, Sequence)
            or isinstance(missing_fields, (str, bytes))
            or not missing_fields
        ):
            raise SchemaError("ask_user.missing_fields must be a non-empty list")
        for index, field in enumerate(missing_fields):
            _require_nonempty_string(field, f"ask_user.missing_fields[{index}]")
        return

    _require_exact_keys(action, {"type", "status", "message", "data"}, "final")
    if action["status"] not in {"success", "unavailable", "not_found"}:
        raise SchemaError(f"unsupported final status: {action['status']!r}")
    _require_nonempty_string(action["message"], "final.message")
    _require_mapping(action["data"], "final.data")


def _validate_tool_arguments(tool: str, arguments: Mapping[str, Any]) -> None:
    """Validate the exact argument schema associated with one tool.

    Args:
        tool: Supported tool name whose argument contract is selected.
        arguments: Candidate argument mapping for that tool.

    Returns:
        ``None`` when the argument names and values are valid.

    Raises:
        SchemaError: If required fields, types, ranges, or allowed keys fail.
    """
    if tool == "search_products":
        _require_exact_keys(arguments, {"query", "filters"}, "search_products arguments")
        _require_nonempty_string(arguments["query"], "search_products.query")
        filters = _require_mapping(arguments["filters"], "search_products.filters")
        allowed = {"brand", "category", "color", "in_stock", "max_price_idr"}
        extra = set(filters) - allowed
        if extra:
            raise SchemaError(f"unsupported search filter(s): {sorted(extra)}")
        for key in ("brand", "category", "color"):
            if key in filters:
                _require_nonempty_string(filters[key], f"search_products.filters.{key}")
        if "in_stock" in filters and not isinstance(filters["in_stock"], bool):
            raise SchemaError("search_products.filters.in_stock must be boolean")
        if "max_price_idr" in filters:
            _require_int(filters["max_price_idr"], "search_products.filters.max_price_idr", 0)
        return

    if tool in {"get_inventory", "get_discount_policy"}:
        _require_exact_keys(arguments, {"product_id"}, f"{tool} arguments")
        _require_nonempty_string(arguments["product_id"], f"{tool}.product_id")
        return

    _require_exact_keys(
        arguments,
        {"discount_percent", "product_id", "quantity"},
        "calculate_offer arguments",
    )
    _require_nonempty_string(arguments["product_id"], "calculate_offer.product_id")
    discount = _require_int(arguments["discount_percent"], "calculate_offer.discount_percent", 0)
    if discount > 100:
        raise SchemaError("calculate_offer.discount_percent must be <= 100")
    _require_int(arguments["quantity"], "calculate_offer.quantity", 1)


def validate_tool_observation(observation: Any) -> None:
    """Validate a successful or failed local-tool observation envelope.

    Args:
        observation: Candidate object returned by a SellerOps tool.

    Returns:
        ``None`` when success/error fields are mutually consistent.

    Raises:
        SchemaError: If the tool, envelope, or nested payload is invalid.
    """
    observation = _require_mapping(observation, "tool observation")
    _require_exact_keys(observation, {"tool", "ok", "result", "error"}, "tool observation")
    if observation["tool"] not in TOOL_NAMES:
        raise SchemaError(f"unsupported observation tool: {observation['tool']!r}")
    if not isinstance(observation["ok"], bool):
        raise SchemaError("tool observation ok must be boolean")
    if observation["ok"]:
        if observation["error"] is not None:
            raise SchemaError("successful observation must have error=null")
        _require_mapping(observation["result"], "tool observation result")
    else:
        if observation["result"] is not None:
            raise SchemaError("failed observation must have result=null")
        error = _require_mapping(observation["error"], "tool observation error")
        _require_exact_keys(error, {"code", "message"}, "tool observation error")
        _require_nonempty_string(error["code"], "tool observation error.code")
        _require_nonempty_string(error["message"], "tool observation error.message")


def validate_evaluation_result(result: Any) -> None:
    """Validate the evaluator-to-runtime routing contract.

    Args:
        result: Candidate decision object containing a route and reason.

    Returns:
        ``None`` when the decision, reason code, and details are valid.

    Raises:
        SchemaError: If the routing contract is malformed.
    """

    result = _require_mapping(result, "evaluation result")
    _require_exact_keys(
        result,
        {"decision", "reason_code", "details"},
        "evaluation result",
    )
    if result["decision"] not in {"pass", "retry", "replan", "fail"}:
        raise SchemaError(f"unsupported evaluation decision: {result['decision']!r}")
    _require_nonempty_string(result["reason_code"], "evaluation result.reason_code")
    _require_mapping(result["details"], "evaluation result.details")


def validate_task(task: Any) -> None:
    """Validate a generated task and its alternating oracle trajectory.

    Args:
        task: Candidate task with prompt metadata, trajectory, and verifier.

    Returns:
        ``None`` when metadata, roles, tool ordering, and terminal action are valid.

    Raises:
        SchemaError: If any task field or trajectory invariant is violated.
    """
    task = _require_mapping(task, "task")
    expected_keys = {
        "context",
        "environment_version",
        "family",
        "locale",
        "prompt",
        "provenance",
        "seed",
        "split",
        "task_id",
        "template_id",
        "trajectory",
        "verifier",
    }
    _require_exact_keys(task, expected_keys, "task")
    for key in ("task_id", "template_id", "environment_version", "prompt"):
        _require_nonempty_string(task[key], f"task.{key}")
    if task["split"] not in SPLITS:
        raise SchemaError(f"unsupported split: {task['split']!r}")
    if task["family"] not in FAMILIES:
        raise SchemaError(f"unsupported family: {task['family']!r}")
    if task["locale"] not in {"id", "en"}:
        raise SchemaError(f"unsupported locale: {task['locale']!r}")
    _require_int(task["seed"], "task.seed", 0)
    _require_mapping(task["context"], "task.context")
    _require_mapping(task["provenance"], "task.provenance")
    _require_mapping(task["verifier"], "task.verifier")

    trajectory = task["trajectory"]
    if not isinstance(trajectory, list) or not trajectory:
        raise SchemaError("task.trajectory must be a non-empty list")
    expecting_tool: str | None = None
    terminal_seen = False
    for index, event in enumerate(trajectory):
        event = _require_mapping(event, f"trajectory[{index}]")
        _require_exact_keys(event, {"role", "content"}, f"trajectory[{index}]")
        if terminal_seen:
            raise SchemaError("trajectory cannot continue after a terminal assistant action")
        if event["role"] == "assistant":
            if expecting_tool is not None:
                raise SchemaError("tool_call must be followed by its tool observation")
            validate_action(event["content"])
            if event["content"]["type"] == "tool_call":
                expecting_tool = event["content"]["tool"]
            else:
                terminal_seen = True
        elif event["role"] == "tool":
            if expecting_tool is None:
                raise SchemaError("tool observation must follow a tool_call")
            validate_tool_observation(event["content"])
            if event["content"]["tool"] != expecting_tool:
                raise SchemaError("tool observation does not match preceding tool_call")
            expecting_tool = None
        else:
            raise SchemaError("trajectory role must be assistant or tool")
    if expecting_tool is not None:
        raise SchemaError("trajectory ends before tool observation")
    if not terminal_seen:
        raise SchemaError("trajectory must end in ask_user or final")


def validate_sft_record(record: Any) -> None:
    """Validate one model-independent SFT decision record.

    Args:
        record: Candidate train/dev record containing serialized messages and
            the final assistant target index.

    Returns:
        ``None`` when split, message roles, JSON content, and target position are valid.

    Raises:
        SchemaError: If the record contains frozen-test data or invalid messages.
    """
    record = _require_mapping(record, "SFT record")
    expected_keys = {
        "dataset_version",
        "environment_version",
        "family",
        "locale",
        "messages",
        "provenance",
        "record_id",
        "split",
        "target_message_index",
        "task_id",
        "template_id",
    }
    _require_exact_keys(record, expected_keys, "SFT record")
    if record["split"] not in {"train", "dev"}:
        raise SchemaError("SFT records may only come from train or dev")
    messages = record["messages"]
    if not isinstance(messages, list) or len(messages) < 3:
        raise SchemaError("SFT messages must contain system, user, and target")
    if messages[0].get("role") != "system" or messages[1].get("role") != "user":
        raise SchemaError("SFT messages must begin with system then user")
    if record["target_message_index"] != len(messages) - 1:
        raise SchemaError("target_message_index must point to the final message")
    if messages[-1].get("role") != "assistant":
        raise SchemaError("SFT target must be an assistant message")

    for index, message in enumerate(messages):
        message = _require_mapping(message, f"messages[{index}]")
        _require_exact_keys(message, {"role", "content"}, f"messages[{index}]")
        _require_nonempty_string(message["content"], f"messages[{index}].content")
        if message["role"] == "assistant":
            try:
                action = json.loads(message["content"])
            except json.JSONDecodeError as error:
                raise SchemaError(f"assistant content is not valid JSON: {error}") from error
            validate_action(action)
        elif message["role"] == "tool":
            try:
                observation = json.loads(message["content"])
            except json.JSONDecodeError as error:
                raise SchemaError(f"tool content is not valid JSON: {error}") from error
            validate_tool_observation(observation)
        elif message["role"] not in {"system", "user"}:
            raise SchemaError(f"unsupported message role: {message['role']!r}")
