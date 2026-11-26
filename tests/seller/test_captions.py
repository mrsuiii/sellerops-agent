"""Caption approval state machine: a caption never changes without approval, and revert works."""

from __future__ import annotations

import sqlite3
import threading

import pytest

from sellerops.seller import MAX_REJECTIONS, SellerError
from sellerops.seller.db import connect

SELLER = "seller:test"


def caption(service, pid="FS180"):
    product = service.analytics().product(pid)
    return {k: product[k] for k in ("name", "description", "tags", "caption_version")}


def propose(service, pid="FS180", **fields):
    fields = fields or {"description": "A clearer description."}
    return service.propose_caption(pid, "clearer", actor="agent", **fields)


def code_of(call) -> str:
    with pytest.raises(SellerError) as error:
        call()
    return error.value.code


def test_proposing_never_changes_the_product(service) -> None:
    before = caption(service)
    proposal = propose(service)
    assert proposal["status"] == "pending" and proposal["base_caption_version"] == before["caption_version"]
    assert caption(service) == before
    assert [e["action"] for e in service.audit_log(proposal_id=proposal["proposal_id"])] == ["proposed"]


def test_approve_applies_atomically_with_a_version_and_audit_row(service) -> None:
    before = caption(service)
    proposal = propose(service, description="A clearer description.")
    applied = service.approve(proposal["proposal_id"], actor=SELLER, note="ok")
    after = caption(service)
    assert applied["status"] == "applied" and applied["decided_by"] == SELLER
    assert after["description"] == "A clearer description."
    assert after["name"] == before["name"] and after["tags"] == before["tags"]  # untouched fields
    versions = service.caption_history("FS180")
    assert [v["source"] for v in versions] == ["seed", "proposal"]
    assert versions[-1]["version_id"] == applied["applied_version_id"]
    assert versions[-1]["caption_version"] == after["caption_version"]
    audit = service.audit_log(proposal_id=proposal["proposal_id"])
    assert [e["action"] for e in audit] == ["approved", "proposed"]
    assert audit[0]["detail"]["from_version"] == before["caption_version"]
    assert audit[0]["detail"]["to_version"] == after["caption_version"]


@pytest.mark.parametrize("action", ["approve", "reject", "cancel"])
def test_decided_proposals_cannot_be_decided_again(service, action) -> None:
    pid = propose(service)["proposal_id"]
    service.approve(pid, actor=SELLER)
    assert code_of(lambda: getattr(service, action)(pid, actor=SELLER)) == "invalid_transition"


def test_applying_one_proposal_makes_other_pending_ones_stale(service) -> None:
    first = propose(service, description="First.")["proposal_id"]
    second = propose(service, tags=["dress", "women"])["proposal_id"]
    service.approve(first, actor=SELLER)
    assert service.get_proposal(second)["status"] == "stale"
    assert code_of(lambda: service.approve(second, actor=SELLER)) == "invalid_transition"
    assert service.audit_log(proposal_id=second)[0]["action"] == "stale"


def test_approving_a_proposal_for_an_older_caption_marks_it_stale_and_keeps_the_caption(service) -> None:
    pid = propose(service)["proposal_id"]
    with connect(service.db_path) as conn:  # e.g. a caption imported from elsewhere meanwhile
        conn.execute("UPDATE caption_proposals SET base_caption_version = 'older' WHERE id = 1")
    before = caption(service)
    assert code_of(lambda: service.approve(pid, actor=SELLER)) == "stale_proposal"
    assert service.get_proposal(pid)["status"] == "stale"  # committed although approve raised
    assert caption(service) == before


def test_failure_mid_approve_rolls_everything_back(service, monkeypatch) -> None:
    pid = propose(service)["proposal_id"]
    before = caption(service)

    def broken_audit(*_args, **_kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(service, "_audit", broken_audit)
    with pytest.raises(RuntimeError):
        service.approve(pid, actor=SELLER)
    monkeypatch.undo()
    assert caption(service) == before
    assert service.get_proposal(pid)["status"] == "pending"
    assert [v["source"] for v in service.caption_history("FS180")] == ["seed"]


def test_concurrent_approvals_apply_once(service) -> None:
    pid = propose(service)["proposal_id"]
    results: list[str] = []

    def worker() -> None:
        try:
            service.approve(pid, actor=SELLER)
            results.append("applied")
        except SellerError as error:
            results.append(error.code)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(results) == ["applied"] + ["invalid_transition"] * 7
    assert [v["source"] for v in service.caption_history("FS180")] == ["seed", "proposal"]


def test_revert_restores_the_previous_caption_as_a_new_version(service) -> None:
    before = caption(service)
    pid = propose(service, name="Dress Pea – Polka", description="New.")["proposal_id"]
    service.approve(pid, actor=SELLER)
    reverted = service.revert(pid, actor=SELLER, note="sales dropped")
    assert reverted["status"] == "reverted" and reverted["reverted_by"] == SELLER
    assert caption(service) == before
    versions = service.caption_history("FS180")
    assert [v["source"] for v in versions] == ["seed", "proposal", "revert"]  # history kept
    assert versions[-1]["caption_version"] == versions[0]["caption_version"]
    assert code_of(lambda: service.revert(pid, actor=SELLER)) == "invalid_transition"


def test_revert_refuses_to_undo_a_later_change(service) -> None:
    first = propose(service, description="First.")["proposal_id"]
    service.approve(first, actor=SELLER)
    second = propose(service, description="Second.")["proposal_id"]
    service.approve(second, actor=SELLER)
    assert code_of(lambda: service.revert(first, actor=SELLER)) == "newer_change_exists"
    service.revert(second, actor=SELLER)
    assert caption(service)["description"] == "First."
    service.revert(first, actor=SELLER)  # now it is the current change again
    assert [v["source"] for v in service.caption_history("FS180")][-2:] == ["revert", "revert"]


@pytest.mark.parametrize("status", ["pending", "rejected"])
def test_only_applied_proposals_can_be_reverted(service, status) -> None:
    pid = propose(service)["proposal_id"]
    if status == "rejected":
        service.reject(pid, actor=SELLER)
    assert code_of(lambda: service.revert(pid, actor=SELLER)) == "invalid_transition"


def test_reject_keeps_feedback_and_allows_one_revision(service) -> None:
    first = propose(service, description="Too long.")["proposal_id"]
    rejected = service.reject(first, actor=SELLER, feedback=" shorter please ")
    assert rejected["status"] == "rejected" and rejected["feedback"] == "shorter please"
    revision = service.propose_caption("FS180", "shorter", actor="agent", description="Short.", revision_of=first)
    assert revision["revision_of"] == first and revision["rejections_in_chain"] == 1
    again = lambda: service.propose_caption("FS180", "x", actor="agent", description="Other.", revision_of=first)  # noqa: E731
    assert code_of(again) == "invalid_revision"


@pytest.mark.parametrize("setup", ["pending", "other_product"])
def test_invalid_revisions(service, setup) -> None:
    pid = propose(service)["proposal_id"]
    if setup == "other_product":
        service.reject(pid, actor=SELLER)
        call = lambda: service.propose_caption("FS179", "x", actor="agent", description="D.", revision_of=pid)  # noqa: E731
    else:
        call = lambda: service.propose_caption("FS180", "x", actor="agent", description="D.", revision_of=pid)  # noqa: E731
    assert code_of(call) == "invalid_revision"


def test_revision_chain_stops_after_max_rejections(service) -> None:
    previous = None
    for i in range(MAX_REJECTIONS):
        pid = service.propose_caption("FS180", "try", actor="agent", description=f"Try {i}.",
                                      revision_of=previous)["proposal_id"]
        assert service.reject(pid, actor=SELLER)["rejections_in_chain"] == i + 1
        previous = pid
    call = lambda: service.propose_caption("FS180", "again", actor="agent", description="Try 9.", revision_of=previous)  # noqa: E731
    assert code_of(call) == "max_rejections"


@pytest.mark.parametrize(("call", "code"), [
    (lambda s: s.propose_caption("NOPE", "r", actor="agent", name="X"), "product_not_found"),
    (lambda s: s.propose_caption("FS180", " ", actor="agent", name="X"), "missing_reason"),
    (lambda s: s.propose_caption("FS180", "r", actor="agent", name="x" * 81), "field_too_long"),
    (lambda s: s.propose_caption("FS180", "r", actor="agent", name="Dress Pea"), "no_changes"),
    (lambda s: s.get_proposal("CP-9999"), "proposal_not_found"),
    (lambda s: s.get_proposal("bogus"), "proposal_not_found"),
    (lambda s: s.list_proposals(status="approved"), "invalid_status"),
])
def test_business_errors(service, call, code) -> None:
    assert code_of(lambda: call(service)) == code


def test_list_proposals_filters(service) -> None:
    a = propose(service)["proposal_id"]
    b = propose(service, "FS179", description="Other.")["proposal_id"]
    service.reject(b, actor=SELLER)
    assert [p["proposal_id"] for p in service.list_proposals()] == [b, a]
    assert [p["proposal_id"] for p in service.list_proposals(status="pending")] == [a]
    assert [p["proposal_id"] for p in service.list_proposals(product_id="FS179")] == [b]


def test_cancel_closes_a_pending_proposal_without_changes_or_revision(service) -> None:
    before = caption(service)
    pid = propose(service)["proposal_id"]
    cancelled = service.cancel(pid, actor=SELLER, note="not now")
    assert cancelled["status"] == "cancelled" and cancelled["decided_by"] == SELLER
    assert caption(service) == before
    assert service.audit_log(proposal_id=pid)[0]["action"] == "cancelled"
    revise = lambda: service.propose_caption("FS180", "x", actor="agent", description="D.", revision_of=pid)  # noqa: E731
    assert code_of(revise) == "invalid_revision"
    assert code_of(lambda: service.revert(pid, actor=SELLER)) == "invalid_transition"


# ------------------------------------------------------- database invariants


def test_schema_blocks_caption_changes_without_a_version(service) -> None:
    with connect(service.db_path) as conn, pytest.raises(sqlite3.IntegrityError, match="caption"):
        conn.execute("UPDATE products SET name = 'Hacked' WHERE product_id = 'FS180'")
    with connect(service.db_path) as conn, pytest.raises(sqlite3.IntegrityError, match="caption"):
        conn.execute("UPDATE products SET current_version_id = 1 WHERE product_id = 'FS180'")  # FS006's version


@pytest.mark.parametrize("sql", ["UPDATE audit_log SET actor = 'x'", "DELETE FROM audit_log",
                                 "UPDATE caption_versions SET name = 'x'", "DELETE FROM caption_versions"])
def test_history_tables_are_append_only(service, sql) -> None:
    with connect(service.db_path) as conn, pytest.raises(sqlite3.IntegrityError, match="append-only"):
        conn.execute(sql)
