from __future__ import annotations

import hashlib
import json

import pytest

from sellerops.dataset import (
    DATASET_VERSION,
    SYSTEM_PROMPT,
    build_dataset_files,
    generate_sft_records,
    generate_tasks,
)
from sellerops.environment import SellerEnvironment
from sellerops.schemas import (
    MAX_AGENT_RETRIES,
    MAX_AGENT_STEPS,
    SchemaError,
    validate_action,
    validate_evaluation_result,
    validate_sft_record,
    validate_task,
)


def test_action_schema_rejects_cross_type_and_unsafe_shape() -> None:
    """Ensure action validation rejects cross-type fields and unsafe discounts.

    Returns:
        ``None``; pytest reports a failure if an invalid action is accepted.
    """
    validate_action(
        {
            "type": "tool_call",
            "tool": "calculate_offer",
            "arguments": {"product_id": "P1001", "discount_percent": 10, "quantity": 1},
        }
    )
    with pytest.raises(SchemaError, match="extra"):
        validate_action(
            {
                "type": "final",
                "status": "success",
                "message": "done",
                "data": {},
                "tool": "calculate_offer",
            }
        )
    with pytest.raises(SchemaError, match="<= 100"):
        validate_action(
            {
                "type": "tool_call",
                "tool": "calculate_offer",
                "arguments": {"product_id": "P1001", "discount_percent": 101, "quantity": 1},
            }
        )


def test_system_prompt_declares_custom_action_and_tool_contracts() -> None:
    """Ensure the model-facing prompt names the complete action/tool contract.

    Returns:
        ``None``; pytest reports a failure if a required contract element is missing.
    """
    assert DATASET_VERSION == "sellerops-sft-v0.2"
    assert '"type":"tool_call"' in SYSTEM_PROMPT
    assert '"tool":"<tool_name>"' in SYSTEM_PROMPT
    assert '"arguments"' in SYSTEM_PROMPT
    for tool_name in (
        "search_products",
        "get_inventory",
        "get_discount_policy",
        "calculate_offer",
    ):
        assert tool_name in SYSTEM_PROMPT


def test_evaluation_result_and_runtime_limits_are_explicit() -> None:
    """Ensure evaluator decisions and runtime step/retry limits stay explicit.

    Returns:
        ``None``; pytest reports a failure if routing validation accepts bad data.
    """
    assert MAX_AGENT_STEPS == 8
    assert MAX_AGENT_RETRIES == 2
    validate_evaluation_result(
        {"decision": "retry", "reason_code": "invalid_arguments", "details": {}}
    )
    with pytest.raises(SchemaError, match="unsupported evaluation decision"):
        validate_evaluation_result(
            {"decision": "continue_forever", "reason_code": "bad", "details": {}}
        )


def test_environment_is_deterministic_and_enforces_policy() -> None:
    """Ensure tools enforce discount policy without mutating inventory state.

    Returns:
        ``None``; pytest reports a failure if policy or read-only behavior changes.
    """
    environment = SellerEnvironment()
    inventory_before = json.dumps(environment.inventory, sort_keys=True)
    safe = environment.calculate_offer("P1001", discount_percent=10, quantity=1)
    unsafe = environment.calculate_offer("P1001", discount_percent=11, quantity=1)
    assert safe["ok"] is True
    assert safe["result"]["unit_price_idr"] == 224_100
    assert unsafe["ok"] is False
    assert unsafe["error"]["code"] == "discount_policy_violation"
    assert json.dumps(environment.inventory, sort_keys=True) == inventory_before


def test_task_counts_validation_and_template_isolation() -> None:
    """Ensure split counts, task schemas, families, and template isolation hold.

    Returns:
        ``None``; pytest reports a failure when generated tasks violate invariants.
    """
    tasks = generate_tasks()
    assert {split: len(rows) for split, rows in tasks.items()} == {
        "train": 180,
        "dev": 30,
        "test": 48,
    }
    template_sets = {
        split: {task["template_id"] for task in rows}
        for split, rows in tasks.items()
    }
    assert template_sets["train"].isdisjoint(template_sets["dev"])
    assert template_sets["train"].isdisjoint(template_sets["test"])
    assert template_sets["dev"].isdisjoint(template_sets["test"])
    assert {task["family"] for split in tasks.values() for task in split} == {
        "product_search",
        "inventory_offer",
        "policy_qa",
    }
    for split, rows in tasks.items():
        for task in rows:
            assert task["split"] == split
            validate_task(task)


def test_sft_records_exclude_frozen_test_and_have_explicit_target() -> None:
    """Ensure SFT records exclude frozen tasks and identify one target message.

    Returns:
        ``None``; pytest reports a failure if split isolation or target metadata breaks.
    """
    tasks = generate_tasks()
    records = generate_sft_records(tasks)
    assert 400 <= len(records["train"]) <= 700
    test_task_ids = {task["task_id"] for task in tasks["test"]}
    all_sft = records["train"] + records["dev"]
    assert test_task_ids.isdisjoint({record["task_id"] for record in all_sft})
    for record in all_sft:
        validate_sft_record(record)
        assert record["target_message_index"] == len(record["messages"]) - 1
        assert record["provenance"]["masking_contract"] == "final_assistant_message_only"


def test_generation_is_byte_reproducible_and_manifest_hashes_match() -> None:
    """Ensure identical seeds produce identical bytes and matching manifest hashes.

    Returns:
        ``None``; pytest reports a failure if generation is nondeterministic.
    """
    first = build_dataset_files()
    second = build_dataset_files()
    assert first == second
    manifest = json.loads(first["manifest.json"])
    assert manifest["frozen_test_sha256"] == hashlib.sha256(first["tasks/test.jsonl"]).hexdigest()
    for path, expected_hash in manifest["file_sha256"].items():
        assert hashlib.sha256(first[path]).hexdigest() == expected_hash
