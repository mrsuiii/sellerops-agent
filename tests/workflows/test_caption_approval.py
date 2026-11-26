"""Tests for the caption approval workflow (tutorial: docs/tutorial-c3-approval-graph.md)."""

from __future__ import annotations

import sys
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from sellerops.harness.trajectory import open_sqlite_checkpointer
from sellerops.workflows.approval.approval_graph import build_approval_graph
from sellerops.workflows.approval.approval_state import thread_config

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
from fake_models import call, fake  # noqa: E402, F401


def start(service, model=None, **fields):
    proposal = service.propose_caption("FS180", "clearer", actor="agent",
                                       **(fields or {"description": "Clearer."}))
    graph = build_approval_graph(service, model or fake(), checkpointer=InMemorySaver())
    config = thread_config(proposal["proposal_id"])
    out = graph.invoke({"proposal_id": proposal["proposal_id"]}, config)
    return graph, config, proposal, out


def resume(graph, config, **answer):
    return graph.invoke(Command(resume={"user": "seller-1", **answer}), config)


DRAFT = lambda text: call("CaptionDraft", {"description": text, "reason": "lebih pendek"})  # noqa: E731


def test_graph_waits_for_the_seller(service) -> None:
    before = service.analytics().product("FS180")
    graph, config, proposal, out = start(service)
    request = out["__interrupt__"][0].value
    assert request["proposal"]["proposal_id"] == proposal["proposal_id"]
    assert request["fields"] == ["description"]
    assert graph.get_state(config).next == ("await_decision",)
    assert service.analytics().product("FS180") == before   # belum berubah


def test_decision_is_stored_and_fields_are_merged(service) -> None:
    graph, config, _, _ = start(service)
    out = resume(graph, config, decision="reject", feedback="  lebih singkat ", add_fields=["name", "description"],
                 user="rina")  # langsung lanjut ke revisi; fake model tanpa draft → revision_failed
    assert (out["decision"], out["feedback"], out["decided_by"]) == ("reject", "lebih singkat", "seller:rina")
    assert out["fields"] == ["description", "name"]


def test_invalid_answer_asks_again(service) -> None:
    graph, config, _, _ = start(service)
    out = resume(graph, config, decision="approve", add_fields=["name"], user="rina")   # add_fields hanya untuk reject
    assert out["__interrupt__"][0].value["error"]
    assert graph.get_state(config).next == ("await_decision",)
    out = resume(graph, config, decision="approve", user="rina")
    assert out["decision"] == "approve" and "__interrupt__" not in out


# ---------------------------------------------------------------- approve / cancel


def test_approve_applies_the_caption(service) -> None:
    graph, config, proposal, _ = start(service)
    out = resume(graph, config, decision="approve")
    assert out["stop_reason"] == "applied"
    assert service.analytics().product("FS180")["description"] == "Clearer."
    assert service.get_proposal(proposal["proposal_id"])["decided_by"] == "seller:seller-1"
    assert graph.get_state(config).next == ()


def test_stale_proposal_is_not_applied(service) -> None:
    graph, config, _, _ = start(service)
    other = service.propose_caption("FS180", "x", actor="agent", tags=["dress"])
    service.approve(other["proposal_id"], actor="seller:seller-2")   # caption berubah duluan
    out = resume(graph, config, decision="approve")
    assert out["stop_reason"] == "stale"
    assert service.analytics().product("FS180")["description"] != "Clearer."


def test_apply_is_idempotent_after_a_crash(service) -> None:
    graph, config, proposal, _ = start(service)
    service.approve(proposal["proposal_id"], actor="seller:seller-1")  # "node sempat jalan, lalu crash"
    assert resume(graph, config, decision="approve")["stop_reason"] == "applied"


def test_cancel_stops_without_changes(service) -> None:
    before = service.analytics().product("FS180")
    graph, config, proposal, _ = start(service)
    assert resume(graph, config, decision="cancel")["stop_reason"] == "cancelled"
    assert service.get_proposal(proposal["proposal_id"])["status"] == "cancelled"
    assert service.analytics().product("FS180") == before


def test_cancel_is_idempotent_after_a_crash(service) -> None:
    graph, config, proposal, _ = start(service)
    service.cancel(proposal["proposal_id"], actor="seller:seller-1")
    assert resume(graph, config, decision="cancel")["stop_reason"] == "cancelled"


# ---------------------------------------------------------------- reject → revise


def test_reject_then_revision_then_approve(service) -> None:
    graph, config, first, _ = start(service, model=fake(DRAFT("Short.")))
    out = resume(graph, config, decision="reject", feedback="terlalu panjang")
    revision = out["__interrupt__"][0].value["proposal"]
    assert revision["revision_of"] == first["proposal_id"]
    assert revision["changes"]["description"]["new"] == "Short."
    assert revision["created_by"] == "agent:caption-approval"
    assert resume(graph, config, decision="approve")["stop_reason"] == "applied"
    assert service.analytics().product("FS180")["description"] == "Short."


def test_revision_prompt_carries_every_rejected_version_and_reason(service) -> None:
    model = fake(DRAFT("Short."), DRAFT("Shorter."))
    graph, config, _, _ = start(service, model=model)
    resume(graph, config, decision="reject", feedback="terlalu panjang")
    resume(graph, config, decision="reject")
    prompt = model.calls[-1][-1].content
    assert "Clearer." in prompt and "terlalu panjang" in prompt       # versi pertama + alasannya
    assert "Short." in prompt and "(no reason given)" in prompt       # versi kedua, tanpa alasan
    assert "Fields you may change: description" in prompt
    assert graph.get_state(config).values["feedback_history"] == ["terlalu panjang", "(no reason given)"]


def test_five_rejections_stop_the_flow(service) -> None:
    graph, config, _, _ = start(service, model=fake(*[DRAFT(f"Try {i}.") for i in range(5)]))
    for _ in range(4):
        assert "__interrupt__" in resume(graph, config, decision="reject")
    out = resume(graph, config, decision="reject")
    assert out["stop_reason"] == "max_rejections"
    assert len(service.list_proposals(product_id="FS180")) == 5      # tidak ada proposal ke-6


def test_revision_cannot_touch_fields_that_were_not_asked(service) -> None:
    bad = call("CaptionDraft", {"name": "New Name", "reason": "x"})
    model = fake(bad, bad)
    graph, config, _, _ = start(service, model=model)
    out = resume(graph, config, decision="reject")
    assert out["stop_reason"] == "revision_failed"
    assert "leave ['name'] null" in model.calls[1][-1].content          # percobaan ke-2 melihat error-nya
    assert len(service.list_proposals(product_id="FS180")) == 1


def test_refused_draft_gets_a_second_chance(service) -> None:
    too_long = DRAFT("x" * 700)
    graph, config, _, _ = start(service, model=fake(too_long, DRAFT("Fixed.")))
    out = resume(graph, config, decision="reject")
    assert out["__interrupt__"][0].value["proposal"]["changes"]["description"]["new"] == "Fixed."


def test_model_without_a_draft_counts_as_a_failed_attempt(service) -> None:
    graph, config, _, _ = start(service, model=fake())                # model hanya menjawab teks
    assert resume(graph, config, decision="reject")["stop_reason"] == "revision_failed"


def test_seller_can_add_a_field_when_rejecting(service) -> None:
    draft = call("CaptionDraft", {"name": "Dress Pea Polka", "description": "Short.", "reason": "x"})
    graph, config, _, _ = start(service, model=fake(draft))
    out = resume(graph, config, decision="reject", feedback="ubah juga namanya", add_fields=["name"])
    assert out["__interrupt__"][0].value["proposal"]["changes"]["name"]["new"] == "Dress Pea Polka"


def test_reject_without_feedback_still_revises(service) -> None:
    graph, config, _, _ = start(service, model=fake(DRAFT("Other.")))
    assert "__interrupt__" in resume(graph, config, decision="reject")


def test_submit_is_idempotent_after_a_crash(service) -> None:
    graph, config, first, _ = start(service, model=fake(DRAFT("Short.")))
    service.reject(first["proposal_id"], actor="seller:seller-1")
    early = service.propose_caption("FS180", "r", actor="agent:caption-approval", description="Early.",
                                    revision_of=first["proposal_id"])   # "revisi sempat dibuat, lalu crash"
    out = resume(graph, config, decision="reject")
    assert out["__interrupt__"][0].value["proposal"]["proposal_id"] == early["proposal_id"]
    assert len(service.list_proposals(product_id="FS180")) == 2


# ---------------------------------------------------------------- restart


def test_resume_survives_a_restart(service, tmp_path) -> None:
    proposal = service.propose_caption("FS180", "clearer", actor="agent", description="Clearer.")
    config = thread_config(proposal["proposal_id"])
    first = build_approval_graph(service, fake(), checkpointer=open_sqlite_checkpointer(tmp_path / "a.sqlite"))
    first.invoke({"proposal_id": proposal["proposal_id"]}, config)
    del first                                                         # "server mati"
    again = build_approval_graph(service, fake(), checkpointer=open_sqlite_checkpointer(tmp_path / "a.sqlite"))
    assert again.invoke(Command(resume={"decision": "approve", "user": "s"}), config)["stop_reason"] == "applied"
