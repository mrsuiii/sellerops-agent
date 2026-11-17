"""Seller eval tasks: reproducible, frozen test split, and verifier behavior."""

from __future__ import annotations

import copy
import json

import pytest

from sellerops.evaluation.seller_tasks import task_files
from sellerops.evaluation.seller_verifier import verify_conversation
from sellerops.simulator import SellerEpisode
from sellerops.simulator.store import default_sim_dir, load_sim_data


def result_for(final: dict | None, *, asked: bool = False) -> dict:
    turns = ([{"stop_reason": "ask_user"}] if asked else []) + [{"stop_reason": "final" if final else "max_steps"}]
    return {"final_action": final, "stop_reason": "final" if final else "max_steps", "turns": turns}


def test_task_files_match_committed_files() -> None:
    files = task_files(load_sim_data())
    for name, text in files.items():
        assert (default_sim_dir() / name).read_text(encoding="utf-8") == text, f"{name} is stale; regenerate"
    manifest = json.loads(files["tasks.manifest.json"])
    assert manifest["frozen"] == ["tasks.test.jsonl"]
    assert manifest["counts"]["dev"] >= 10 and manifest["counts"]["test"] >= 9


def test_tasks_are_well_formed(all_tasks) -> None:
    ids = [t["task_id"] for t in all_tasks]
    assert len(ids) == len(set(ids))
    for task in all_tasks:
        assert task["split"] in {"dev", "test"} and task["task_id"].startswith(f"seller-{task['split']}-")
        assert task["opening_message"] and task["persona"] and task["goal"]
        assert task["needs_clarification"] == bool(task["hidden_info"])
        for fact in task["hidden_info"]:
            assert fact["fact"] and fact["keywords"]
        # The answer key never leaks into what the user simulator or agent sees.
        visible = task["opening_message"] + " ".join(f["fact"] for f in task["hidden_info"])
        for value in task["expected"].values():
            if isinstance(value, str) and value.startswith("FS"):
                assert value not in visible


def test_reference_solution_passes_every_task(all_tasks, solve) -> None:
    for task in all_tasks:
        episode = SellerEpisode()
        final = solve(task, episode)
        report = verify_conversation(task, episode.snapshot(), result_for(final), store=episode.store)
        assert report["decision"] == "pass", (task["task_id"], report)


@pytest.mark.parametrize("mutate", [
    lambda d: d.update(revenue_idr=d["revenue_idr"] + 1000),
    lambda d: d.update(orders=d["orders"] - 1),
])
def test_wrong_or_invented_numbers_fail(dev_tasks, solve, mutate) -> None:
    task = dev_tasks["seller-dev-sales-summary-omzet-30-hari"]
    episode = SellerEpisode()
    final = solve(task, episode)
    mutate(final["data"])
    report = verify_conversation(task, episode.snapshot(), result_for(final))
    assert report["decision"] == "fail"
    assert {"answer", "grounded"} <= set(report["failed"])


def test_answer_without_tool_calls_is_not_grounded(dev_tasks, solve) -> None:
    task = dev_tasks["seller-dev-sales-summary-omzet-30-hari"]
    final = solve(task, SellerEpisode())  # solved in another episode
    report = verify_conversation(task, SellerEpisode().snapshot(), result_for(final))
    assert report["failed"] == ["grounded"]


def test_unfinished_conversation_fails(dev_tasks) -> None:
    task = dev_tasks["seller-dev-sales-summary-omzet-30-hari"]
    report = verify_conversation(task, SellerEpisode().snapshot(), result_for(None))
    assert report["decision"] == "fail" and report["failed"] == ["finished"]


def test_ranking_order_matters_but_sets_do_not(dev_tasks, solve) -> None:
    top = dev_tasks["seller-dev-top-products-sepatu-terlaris"]
    episode = SellerEpisode()
    final = solve(top, episode)
    final["data"]["product_ids"].reverse()
    assert verify_conversation(top, episode.snapshot(), result_for(final))["failed"] == ["answer"]

    restock = dev_tasks["seller-dev-restock-restock-semua"]
    episode = SellerEpisode()
    final = solve(restock, episode)
    final["data"]["product_ids"].reverse()
    assert verify_conversation(restock, episode.snapshot(), result_for(final))["decision"] == "pass"


def test_caption_must_change_only_what_the_seller_asked(dev_tasks, solve) -> None:
    task = dev_tasks["seller-dev-caption-caption-dress-pea"]  # description only
    episode = SellerEpisode()
    episode.call("propose_caption_update", {"product_id": "FS180", "reason": "x", "name": "Dress Pea Baru"})
    final = {"type": "final", "message": "ok", "data": {"proposal_id": "CP-0001", "product_id": "FS180"}}
    report = verify_conversation(task, episode.snapshot(), result_for(final), store=episode.store)
    assert report["failed"] == ["caption_flow"]

    bogus = copy.deepcopy(final)
    bogus["data"]["proposal_id"] = "CP-9999"
    report = verify_conversation(task, SellerEpisode().snapshot(), result_for(bogus))
    assert "answer" in report["failed"]


def test_clarification_metrics(dev_tasks, solve) -> None:
    task = dev_tasks["seller-dev-sales-summary-toko-tanpa-periode"]
    episode = SellerEpisode()
    final = solve(task, episode)
    report = verify_conversation(task, episode.snapshot(), result_for(final, asked=True))
    assert report["metrics"] == {"agent_turns": 2, "tool_calls": 1, "asked_clarification": True,
                                 "needs_clarification": True}
