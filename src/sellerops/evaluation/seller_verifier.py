"""Outcome verifier for multi-turn seller conversations.

Judges the end of the conversation, not the path: any tool sequence is fine
as long as the final answer is right and grounded. Checks:

- ``finished``: the conversation ended with a valid ``final`` action;
- ``answer``: ``final.data`` matches the task's answer key (per family);
- ``grounded``: every number in ``final.data`` appears in a successful tool
  result of this episode ("angka tidak boleh dikarang");
- ``caption_flow`` (caption tasks): a pending proposal exists for the right
  product, changes only allowed fields, covers the required ones, and the
  product itself is unchanged.

Reported but not pass/fail: number of turns and whether the agent asked when
the task needed clarification. The free-text ``message`` is not graded here
(that is the LLM judge's job, later).
"""

from __future__ import annotations

from typing import Any


def _numbers(value: Any) -> list[float]:
    if isinstance(value, bool):
        return []
    if isinstance(value, int | float):
        return [float(value)]
    if isinstance(value, dict):
        return [n for v in value.values() for n in _numbers(v)]
    if isinstance(value, list):
        return [n for v in value for n in _numbers(v)]
    return []


def _observed_numbers(snapshot: dict[str, Any]) -> set[float]:
    return {n for step in snapshot["history"] if step["observation"]["ok"]
            for n in _numbers(step["observation"]["result"])}


def _ids(value: Any) -> list[str] | None:
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        return None
    return [v.strip().upper() for v in value]


def _check_answer(task: dict[str, Any], data: dict[str, Any], snapshot: dict[str, Any]) -> tuple[bool, str]:
    family, expected = task["family"], task["expected"]
    if family == "sales_summary":
        for key in ("revenue_idr", "orders"):
            if data.get(key) != expected[key]:
                return False, f"{key}={data.get(key)!r}, expected {expected[key]}"
        return True, "totals match"
    if family == "top_category":
        ok = str(data.get("category", "")).strip().lower() == expected["category"]
        return ok, f"category={data.get('category')!r}, expected {expected['category']!r}"
    if family == "top_products":
        ok = _ids(data.get("product_ids")) == expected["product_ids"]
        return ok, f"product_ids={data.get('product_ids')!r}, expected {expected['product_ids']}"
    if family in ("slow_movers", "restock"):
        ids = _ids(data.get("product_ids"))
        ok = ids is not None and sorted(ids) == expected["product_ids"]
        return ok, f"product_ids={data.get('product_ids')!r}, expected (any order) {expected['product_ids']}"
    if family == "review_insight":
        if str(data.get("product_id", "")).strip().upper() != expected["product_id"]:
            return False, f"product_id={data.get('product_id')!r}, expected {expected['product_id']}"
        complaint = str(data.get("complaint", "")).lower()
        ok = any(word in complaint for word in expected["complaint_keywords"])
        return ok, f"complaint={data.get('complaint')!r}, expected issue {expected['issue']}"
    if family == "sales_spike":
        ok = str(data.get("product_id", "")).strip().upper() == expected["product_id"]
        return ok, f"product_id={data.get('product_id')!r}, expected {expected['product_id']}"
    if family == "sales_drop":
        ok = (str(data.get("product_id", "")).strip().upper() == expected["product_id"]
              and data.get("cause") == expected["cause"])
        return ok, f"product_id={data.get('product_id')!r} cause={data.get('cause')!r}, expected {expected}"
    if family == "caption":
        proposals = {p["proposal_id"]: p for p in snapshot["proposals"]}
        proposal = proposals.get(str(data.get("proposal_id", "")).strip())
        if proposal is None:
            return False, f"proposal_id={data.get('proposal_id')!r} not created in this episode"
        ok = proposal["product_id"] == expected["product_id"]
        return ok, f"proposal for {proposal['product_id']}, expected {expected['product_id']}"
    raise ValueError(f"unknown family {family!r}")


def _check_caption_flow(task: dict[str, Any], snapshot: dict[str, Any], store: Any) -> tuple[bool, str]:
    expected = task["expected"]
    proposals = [p for p in snapshot["proposals"] if p["product_id"] == expected["product_id"]]
    if not proposals:
        return False, "no proposal for the product"
    latest = proposals[-1]
    fields = set(latest["changes"])
    if latest["status"] != "pending":
        return False, f"proposal status {latest['status']!r}, expected pending"
    if not fields <= set(expected["allowed_fields"]):
        return False, f"changed fields {sorted(fields)} outside {expected['allowed_fields']}"
    if fields != set(expected["required_fields"]):
        return False, f"changed fields {sorted(fields)}, seller asked for {expected['required_fields']}"
    if store is not None:
        product = store.product(expected["product_id"])
        if product["caption_version"] != latest["base_caption_version"]:
            return False, "product caption changed without approval"
    return True, f"pending proposal {latest['proposal_id']} with fields {sorted(fields)}"


def verify_conversation(task: dict[str, Any], snapshot: dict[str, Any], result: dict[str, Any],
                        *, store: Any = None) -> dict[str, Any]:
    """Grade one conversation. ``result`` comes from ``run_conversation``;
    ``store`` (the episode's ``SellerStore``) enables the product-unchanged check."""

    checks: dict[str, dict[str, Any]] = {}
    final = result.get("final_action")
    finished = isinstance(final, dict) and final.get("type") == "final" and isinstance(final.get("data"), dict)
    checks["finished"] = {"pass": finished,
                          "detail": f"stop_reason={result.get('stop_reason')!r}"}
    if finished:
        data = final["data"]
        ok, detail = _check_answer(task, data, snapshot)
        checks["answer"] = {"pass": ok, "detail": detail}
        observed = _observed_numbers(snapshot)
        missing = [n for n in _numbers(data) if not any(abs(n - o) <= 0.01 for o in observed)]
        checks["grounded"] = {"pass": not missing,
                              "detail": "all numbers from tools" if not missing else f"not in any tool result: {missing}"}
        if task["family"] == "caption":
            ok, detail = _check_caption_flow(task, snapshot, store)
            checks["caption_flow"] = {"pass": ok, "detail": detail}

    asked = any(turn.get("stop_reason") == "ask_user" for turn in result.get("turns", []))
    passed = finished and all(check["pass"] for check in checks.values())
    return {
        "task_id": task["task_id"],
        "family": task["family"],
        "decision": "pass" if passed else "fail",
        "checks": checks,
        "failed": sorted(name for name, check in checks.items() if not check["pass"]),
        "metrics": {
            "agent_turns": len(result.get("turns", [])),
            "tool_calls": len(snapshot["history"]),
            "asked_clarification": asked,
            "needs_clarification": task["needs_clarification"],
        },
    }
