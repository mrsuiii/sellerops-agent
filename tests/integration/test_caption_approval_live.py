"""Caption approval graph with a real LLM drafting the revisions (opt-in, costs money).

Each test is one seller session on a fresh database: the agent's first
proposal, the seller's answers, and the model's revisions. About 1-3 model
calls per test.
"""

from __future__ import annotations

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from sellerops.workflows.approval.approval_graph import build_approval_graph
from sellerops.workflows.approval.approval_state import thread_config

pytestmark = pytest.mark.live

PRODUCT = "FS180"  # Dress Pea
LONG_DESCRIPTION = (
    "The Dress Pea is a stylish and comfortable dress with a playful pea pattern. It is perfect for casual "
    "outings, weekend brunches, park walks and relaxed days in the city. The cheerful print adds a fun touch "
    "to your wardrobe, and the easy shape makes it a great choice for everyday wear in any season."
)


def start(service, model, **fields):
    proposal = service.propose_caption(PRODUCT, "make it more engaging", actor="agent",
                                       **(fields or {"description": LONG_DESCRIPTION}))
    graph = build_approval_graph(service, model, checkpointer=InMemorySaver())
    config = thread_config(proposal["proposal_id"])
    graph.invoke({"proposal_id": proposal["proposal_id"]}, config)
    return graph, config, proposal


def decide(graph, config, **answer):
    out = graph.invoke(Command(resume={"user": "tester", **answer}), config)
    if "__interrupt__" in out:
        revision = out["__interrupt__"][0].value["proposal"]
        print(f"\n[live] {revision['proposal_id']} ({revision['reason']}):")
        for field, change in revision["changes"].items():
            print(f"  {field}: {change['new']!r}")
        return out, revision
    print(f"\n[live] stop_reason={out.get('stop_reason')} draft_error={out.get('draft_error')}")
    return out, None


def test_revision_follows_the_feedback_and_can_be_approved(service, live_model) -> None:
    graph, config, _ = start(service, live_model)
    _, revision = decide(graph, config, decision="reject",
                         feedback="Too long. At most two sentences, and keep the pea pattern.")
    assert revision is not None, "the model should produce a valid revision"
    assert set(revision["changes"]) == {"description"}
    new = revision["changes"]["description"]["new"]
    assert len(new) < len(LONG_DESCRIPTION)
    assert "pea" in new.lower()

    out, _ = decide(graph, config, decision="approve")
    assert out["stop_reason"] == "applied"
    assert service.analytics().product(PRODUCT)["description"] == new


def test_revision_never_changes_a_field_the_seller_did_not_allow(service, live_model) -> None:
    name_before = service.analytics().product(PRODUCT)["name"]
    graph, config, _ = start(service, live_model)
    out, revision = decide(graph, config, decision="reject",
                           feedback="Also give it a catchier product name.")   # but no add_fields
    # Either the model keeps to the description, or the code refuses it twice: never a new name.
    if revision is not None:
        assert set(revision["changes"]) == {"description"}
    else:
        assert out["stop_reason"] == "revision_failed"
    assert service.analytics().product(PRODUCT)["name"] == name_before


def test_seller_can_open_the_name_when_rejecting(service, live_model) -> None:
    graph, config, _ = start(service, live_model)
    _, revision = decide(graph, config, decision="reject", add_fields=["name"],
                         feedback="Rename it too: mention that it is a polka-dot pea print dress.")
    assert revision is not None
    assert "name" in revision["changes"]
    assert set(revision["changes"]) <= {"name", "description"}


def test_rejecting_without_a_reason_gives_a_new_version_each_time(service, live_model) -> None:
    graph, config, first = start(service, live_model)
    seen = [first["changes"]["description"]["new"]]
    for _ in range(2):
        _, revision = decide(graph, config, decision="reject")
        assert revision is not None
        new = revision["changes"]["description"]["new"]
        assert new not in seen
        seen.append(new)
    out, _ = decide(graph, config, decision="cancel")
    assert out["stop_reason"] == "cancelled"
