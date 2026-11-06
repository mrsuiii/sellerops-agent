"""Verify structured SellerOps outcomes without choosing agent actions or routes."""

from copy import deepcopy
import json
from typing import Any

from ..schemas import FAMILIES, canonical_json, validate_action
from ..services.episode_service.episode_service import EpisodeService


def _report(reason: str, **details: Any) -> dict[str, Any]:
    """Build a pass/fail evaluation without assigning retry or replan behavior.

    Args:
        reason: Stable reason code; only outcome_valid denotes a pass.
        **details: Diagnostic values copied into the result.

    Returns:
        A dictionary conforming to the existing evaluation-result schema.
    """
    return {
        "decision": "pass" if reason == "outcome_valid" else "fail",
        "reason_code": reason,
        "details": deepcopy(details),
    }


def _validate_task_context(task: dict[str, Any]) -> None:
    """Validate trusted task constraints without reading its oracle trajectory.

    Args:
        task: Task metadata and structured context, with optional dataset fields.

    Raises:
        ValueError: If the task specification is incomplete or unsupported.
    """
    if not isinstance(task, dict) or not isinstance(task.get("task_id"), str) or not task["task_id"].strip():
        raise ValueError("Task requires a non-empty task_id")
    family = task.get("family")
    if not isinstance(family, str) or family not in FAMILIES:
        raise ValueError("Unsupported task family")
    context = task.get("context")
    if not isinstance(context, dict):
        raise ValueError("Task context must be a dictionary")
    if family == "product_search":
        if "missing_fields" in context:
            fields = context["missing_fields"]
            if not isinstance(fields, list) or not fields or any(
                not isinstance(field, str) or not field.strip() for field in fields
            ) or len(set(fields)) != len(fields):
                raise ValueError("missing_fields must contain unique non-empty strings")
        else:
            validate_action({"type": "tool_call", "tool": "search_products", "arguments": {
                "query": context.get("query"), "filters": context.get("filters"),
            }})
        return
    product_id = context.get("product_id")
    if not isinstance(product_id, str) or not product_id.strip():
        raise ValueError("Task context requires product_id")
    if family == "inventory_offer":
        mode = context.get("mode", "offer")
        if not isinstance(mode, str) or mode not in {"offer", "stock_only"}:
            raise ValueError("Unsupported inventory_offer mode")
        discount = 0 if context.get("mode") == "stock_only" else context.get("requested_discount_percent")
        quantity = context.get("quantity")
        if type(quantity) is not int or quantity < 1 or type(discount) is not int or discount < 0:
            raise ValueError("Offer context requires positive quantity and non-negative requested discount")
    else:
        question_type = context.get("question_type")
        if not isinstance(question_type, str) or question_type not in {"limit", "source", "eligibility", "proposal", "check_request", "audit"}:
            raise ValueError("Unsupported policy question_type")
        if context["question_type"] in {"proposal", "check_request", "audit"} and "proposed_discount_percent" not in context:
            raise ValueError("Policy proposal question requires proposed_discount_percent")
        if "proposed_discount_percent" in context and type(context["proposed_discount_percent"]) is not int:
            raise ValueError("proposed_discount_percent must be an integer")


def _evidence_steps(history: list[dict[str, Any]], tool: str, **arguments: Any) -> list[int]:
    """Find successful calls matching specified arguments, independent of order.

    Args:
        history: Replay-verified tool history.
        tool: Tool whose evidence is required.
        **arguments: Required argument values; additional arguments are allowed.

    Returns:
        Zero-based history indexes with matching successful observations.
    """
    return [
        index for index, event in enumerate(history)
        if event["action"]["tool"] == tool and event["result"]["ok"]
        and all(canonical_json(event["action"]["arguments"].get(key)) == canonical_json(value)
                for key, value in arguments.items())
    ]


def _check_final(final: dict[str, Any], status: str, data: dict[str, Any]) -> dict[str, Any]:
    """Compare status and typed structured facts, allowing any non-empty message.

    Args:
        final: Schema-validated final action.
        status: Required final status for the task outcome.
        data: Required exact data fields and values; prose is not inspected.

    Returns:
        A status/data mismatch report or a passing report.
    """
    if final["status"] != status:
        return _report("status_mismatch", expected=status, actual=final["status"])
    if canonical_json(final["data"]) != canonical_json(data):
        return _report("data_mismatch", expected=data, actual=final["data"])
    return _report("outcome_valid")


def _verify_search(
    context: dict[str, Any], initial: dict[str, Any],
    history: list[dict[str, Any]], final: dict[str, Any],
) -> dict[str, Any]:
    """Check recommendations against constraints and observed search results.

    Args:
        context: Validated search query and filters.
        initial: Authoritative starting environment data.
        history: Replay-verified tool observations.
        final: Schema-validated final action.

    Returns:
        A report accepting any non-empty, duplicate-free subset of eligible
        products in any order, or not_found when no eligible product exists.
    """
    eligible = {}
    filters = context["filters"]
    for product_id, product in initial["catalog"].items():
        text = " ".join([product["name"], product["category"], product["brand"], *product["tags"]])
        if context["query"].casefold() not in text.casefold():
            continue
        if "brand" in filters and product["brand"].casefold() != filters["brand"].casefold():
            continue
        if any(key in filters and product[key] != filters[key] for key in ("category", "color")):
            continue
        stock = initial["inventory"][product_id]["stock"]
        if filters.get("in_stock") is True and stock <= 0:
            continue
        if "max_price_idr" in filters and product["price_idr"] > filters["max_price_idr"]:
            continue
        eligible[product_id] = {"product_id": product_id, "name": product["name"],
                                "price_idr": product["price_idr"], "stock": stock}
    search_steps = _evidence_steps(history, "search_products")
    if not search_steps:
        return _report("missing_tool_evidence", tool="search_products")
    if not eligible:
        return _check_final(final, "not_found", {"products": []})
    if final["status"] != "success":
        return _report("status_mismatch", expected="success", actual=final["status"])
    products = final["data"].get("products")
    if set(final["data"]) != {"products"} or not isinstance(products, list) or not products:
        return _report("invalid_products", message="Expected a non-empty products list")
    observed_ids = {
        product["product_id"] for index in search_steps
        for product in history[index]["result"]["result"]["products"]
    }
    seen = set()
    for product in products:
        product_id = product.get("product_id") if isinstance(product, dict) else None
        if not isinstance(product_id, str) or product_id not in eligible:
            return _report("product_constraint_violation", product=product)
        if product_id in seen:
            return _report("duplicate_product", product_id=product_id)
        seen.add(product_id)
        if canonical_json(product) != canonical_json(eligible[product_id]):
            return _report("data_mismatch", expected=eligible[product_id], actual=product)
        if product_id not in observed_ids:
            return _report("missing_tool_evidence", tool="search_products", product_id=product_id)
    return _report("outcome_valid")


def _verify_offer(
    context: dict[str, Any], initial: dict[str, Any],
    history: list[dict[str, Any]], final: dict[str, Any],
) -> dict[str, Any]:
    """Check stock, requested quantity, capped discount, price, and tool evidence.

    Args:
        context: Validated inventory/offer constraints.
        initial: Authoritative starting environment data.
        history: Replay-verified tool observations.
        final: Schema-validated final action.

    Returns:
        A report allowing stock/policy lookup permutations before calculation.
    """
    product_id = context["product_id"]
    quantity = context["quantity"]
    inventory_steps = _evidence_steps(history, "get_inventory", product_id=product_id)
    if not inventory_steps:
        return _report("missing_tool_evidence", tool="get_inventory", product_id=product_id)
    stock = initial["inventory"][product_id]["stock"]
    if stock < quantity or context.get("mode") == "stock_only":
        if any(event["action"]["tool"] == "calculate_offer" for event in history):
            return _report("unexpected_offer", message="This outcome must not create an offer")
        data = {"product_id": product_id, "available_stock": stock}
        if stock < quantity:
            data["requested_quantity"] = quantity
        return _check_final(final, "unavailable" if stock < quantity else "success", data)

    maximum = initial["policies"][product_id]["max_discount_percent"]
    claimed_discount = final["data"].get("discount_percent")
    if type(claimed_discount) in {int, float} and claimed_discount > maximum:
        return _report("policy_violation", product_id=product_id, maximum=maximum, actual=claimed_discount)
    discount = min(context["requested_discount_percent"], maximum)
    unit_price = initial["catalog"][product_id]["price_idr"] * (100 - discount) // 100
    expected = {"product_id": product_id, "quantity": quantity, "discount_percent": discount,
                "unit_price_idr": unit_price, "total_idr": unit_price * quantity}
    outcome = _check_final(final, "success", expected)
    if outcome["decision"] == "fail":
        return outcome
    policy_steps = _evidence_steps(history, "get_discount_policy", product_id=product_id)
    offer_steps = _evidence_steps(history, "calculate_offer", product_id=product_id,
                                  quantity=quantity, discount_percent=discount)
    if not policy_steps or not offer_steps:
        return _report("missing_tool_evidence", tools=[
            name for name, steps in (("get_discount_policy", policy_steps), ("calculate_offer", offer_steps))
            if not steps
        ])
    if not any(min(inventory_steps) < index and min(policy_steps) < index for index in offer_steps):
        return _report("missing_prior_checks", message="Check stock and policy before calculating the offer")
    return _report("outcome_valid")


def _verify_policy(
    context: dict[str, Any], initial: dict[str, Any],
    history: list[dict[str, Any]], final: dict[str, Any],
) -> dict[str, Any]:
    """Check the policy limit, source, and optional eligibility or proposal facts.

    Args:
        context: Validated policy-question constraints.
        initial: Authoritative starting environment data.
        history: Replay-verified tool observations.
        final: Schema-validated final action.

    Returns:
        A report requiring product-specific policy evidence and correct facts.
    """
    product_id = context["product_id"]
    if not _evidence_steps(history, "get_discount_policy", product_id=product_id):
        return _report("missing_tool_evidence", tool="get_discount_policy", product_id=product_id)
    policy = initial["policies"][product_id]
    maximum = policy["max_discount_percent"]
    expected = {"product_id": product_id, "max_discount_percent": maximum, "source": policy["source"]}
    if "proposed_discount_percent" in context:
        proposal = context["proposed_discount_percent"]
        expected.update(proposed_discount_percent=proposal, proposal_allowed=0 <= proposal <= maximum)
    elif context["question_type"] == "eligibility":
        expected["discount_eligible"] = maximum > 0
    return _check_final(final, "success", expected)


def verify_outcome(
    task: dict[str, Any], snapshot: dict[str, Any], final_action: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate structured outcome facts using trusted task and runtime data.

    Oracle trajectory, expected_action_types, and expected_final_status are never
    consulted. Tool evidence must replay successfully, but need not follow one
    canonical sequence. Message/question prose is checked only for schema validity,
    not semantic truth. This is not an authenticity check for caller-supplied
    environment data: task context and the runtime snapshot must be trusted.

    Args:
        task: Task containing task_id, family, and structured context.
        snapshot: EpisodeService snapshot for that same task, with tool history.
        final_action: Candidate final or ask_user tagged action.

    Returns:
        The existing evaluation contract: decision (pass/fail), reason_code,
        details. It does not mutate inputs or choose retry/replan routes.

    Raises:
        ValueError: If the trusted task specification is invalid or unsupported.
        Exception: Unexpected environment implementation errors propagate.
    """
    _validate_task_context(task)
    try:
        json.dumps(final_action, allow_nan=False)
        validate_action(final_action)
    except (TypeError, ValueError) as error:
        return _report("invalid_action", message=str(error))
    if final_action["type"] == "tool_call":
        return _report("incomplete_outcome")
    try:
        EpisodeService._validate_replay_snapshot(snapshot)
    except (TypeError, ValueError) as error:
        return _report("invalid_snapshot", message=str(error))
    if snapshot["task_id"] != task["task_id"]:
        return _report("task_mismatch", expected=task["task_id"], actual=snapshot["task_id"])
    replay = EpisodeService().replay(snapshot)
    if not replay["matched"]:
        return _report("history_mismatch", **replay["first_mismatch"])
    initial = snapshot["initial_environment"]
    if canonical_json(initial) != canonical_json(snapshot["environment"]):
        return _report("environment_changed", message="Read-only tools cannot explain environment mutations")
    history = snapshot["history"]
    for step, event in enumerate(history, start=1):
        error = event["result"]["error"]
        if error and error["code"] == "discount_policy_violation":
            return _report("policy_violation", step=step, action=event["action"])

    context = task["context"]
    if "missing_fields" in context and task["family"] == "product_search":
        if final_action["type"] != "ask_user":
            return _report("clarification_required", missing_fields=context["missing_fields"])
        fields = final_action["missing_fields"]
        if len(fields) != len(set(fields)) or set(fields) != set(context["missing_fields"]):
            return _report("missing_fields_mismatch", expected=context["missing_fields"], actual=fields)
        return _report("outcome_valid")
    if final_action["type"] != "final":
        return _report("unexpected_clarification")
    if task["family"] == "product_search":
        return _verify_search(context, initial, history, final_action)
    if task["family"] == "inventory_offer":
        return _verify_offer(context, initial, history, final_action)
    return _verify_policy(context, initial, history, final_action)
