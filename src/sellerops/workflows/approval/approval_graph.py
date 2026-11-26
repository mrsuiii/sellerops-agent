"""Caption approval graph (C3): one thread per proposal chain, waiting on the seller.

    START → load → await_decision ──(interrupt: the seller decides)
                     ├─ approve → apply ─────────────────────► END  applied | stale
                     ├─ cancel  → cancel_flow ───────────────► END  cancelled
                     └─ reject  → record_rejection
                                    ├─ 5 rejections ─────────► END  max_rejections
                                    └─ draft_revision (LLM) → submit_revision
                                         ├─ ok ──────────────► await_decision (the revision)
                                         ├─ refused, 1st try ► draft_revision (with the error)
                                         └─ refused again ───► END  revision_failed

Design: docs/c3-approval-graph-design.md · Tutorial: docs/tutorial-c3-approval-graph.md

Decisions:

- The service is the source of truth. Nodes call it and copy what routing
  needs into the state; the caption, statuses and rejection count live in
  the database.
- Nodes decide, routing functions only read the state (as in
  ``harness/graph.py``): a node that ends the flow writes ``stop_reason``,
  and routing sends any state with a ``stop_reason`` to ``END``.
- ``await_decision`` re-runs from the top on every resume, so it only reads.
  An invalid answer asks again instead of failing the thread.
- Side-effect nodes (apply, cancel_flow, record_rejection, submit_revision)
  can run twice if the process dies after the service call but before the
  checkpoint; each treats "already done" as success. They have no
  ``RetryPolicy``. Only ``draft_revision`` (no side effects) retries.
- The LLM only writes text. Which fields it may change is checked in code
  (``fields``), and only fields the seller allowed count: the first
  proposal's fields plus ``add_fields`` from a rejection.
"""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from sellerops.harness.graph import DEFAULT_MODEL_RETRY
from sellerops.seller import MAX_REJECTIONS, SellerError, SellerService
from sellerops.workflows.approval.approval_prompts import NO_REASON, CaptionDraft, revision_messages
from sellerops.workflows.approval.approval_state import ApprovalState

MAX_DRAFT_ATTEMPTS = 2
CAPTION_FIELDS = ("name", "description", "tags")
REVISION_ACTOR = "agent:caption-approval"

CaptionField = Literal["name", "description", "tags"]


class DecisionIn(BaseModel):
    """What the seller decides; the API's request body (the user comes from the request)."""

    model_config = ConfigDict(extra="forbid")

    decision: Literal["approve", "reject", "cancel"]
    feedback: str = Field("", max_length=1000, description="Why it was rejected (used for the revision).")
    add_fields: list[CaptionField] = Field(
        [], description="Reject only: fields the revision may change besides the proposal's own.")

    @model_validator(mode="after")
    def _add_fields_only_on_reject(self) -> DecisionIn:
        if self.add_fields and self.decision != "reject":
            raise ValueError("add_fields is only allowed with decision='reject'")
        return self


class Decision(DecisionIn):
    """The seller's answer as the graph receives it in ``Command(resume=...)``.

    The API validates too, but the payload still crosses a boundary into the
    graph, so the graph checks it again (design §5.3).
    """

    user: str = Field(min_length=1, max_length=64, pattern=r"^[\w.@-]+$")


def merge_fields(fields: list[str], extra: list[str]) -> list[str]:
    """Allowed fields plus the seller's additions, order kept, no duplicates."""
    return list(dict.fromkeys([*fields, *extra]))


def build_approval_graph(service: SellerService, model: BaseChatModel, *,
                         checkpointer: BaseCheckpointSaver | None = None) -> CompiledStateGraph:

    def load(state: ApprovalState) -> dict[str, Any]:
        """Fill the state from the proposal; the graph's input is only ``proposal_id``."""
        proposal = service.get_proposal(state["proposal_id"])
        return {
            "product_id": proposal["product_id"],
            "fields": list(proposal["changes"]),
            "rejections": proposal["rejections_in_chain"],
        }

    def await_decision(state: ApprovalState) -> dict[str, Any]:
        """Wait for the seller. Runs again from the top on every resume: reads only.

        An invalid answer asks again (a new ``interrupt`` carrying the error)
        instead of failing the run: the thread stays usable, and the UI can
        show the error and let the seller retry.
        """
        request: dict[str, Any] = {
            "proposal": service.get_proposal(state["proposal_id"]),
            "fields": state["fields"],
            "rejections": state.get("rejections", 0),
        }
        while True:
            answer = interrupt(request)
            try:
                decision = Decision.model_validate(answer)
                break
            except ValidationError as error:
                request = {**request, "error": error.errors(include_url=False, include_context=False)}
        return {
            "decision": decision.decision,
            "feedback": decision.feedback.strip(),
            "decided_by": f"seller:{decision.user}",
            "fields": merge_fields(state["fields"], decision.add_fields),
        }
    drafter = model.with_structured_output(CaptionDraft)

    def status(proposal_id: str) -> str:
        return service.get_proposal(proposal_id)["status"]

    def apply(state: ApprovalState) -> dict[str, Any]:
        """Approve = apply the caption (one transaction in the service)."""
        try:
            service.approve(state["proposal_id"], actor=state["decided_by"])
        except SellerError as error:
            if error.code == "stale_proposal":  # the caption changed without another proposal
                return {"stop_reason": "stale"}
            if error.code == "invalid_transition":
                # Already applied (this node ran before a crash), or marked stale
                # when another proposal for the product was applied first.
                current = status(state["proposal_id"])
                if current in ("applied", "stale"):
                    return {"stop_reason": current}
            raise
        return {"stop_reason": "applied"}

    def cancel_flow(state: ApprovalState) -> dict[str, Any]:
        try:
            service.cancel(state["proposal_id"], actor=state["decided_by"])
        except SellerError as error:
            if error.code != "invalid_transition" or status(state["proposal_id"]) != "cancelled":
                raise
        return {"stop_reason": "cancelled"}

    def record_rejection(state: ApprovalState) -> dict[str, Any]:
        """Reject in the service; its rejection count decides whether to revise."""
        feedback = state.get("feedback", "")
        try:
            proposal = service.reject(state["proposal_id"], actor=state["decided_by"], feedback=feedback)
        except SellerError as error:
            proposal = service.get_proposal(state["proposal_id"])
            if error.code != "invalid_transition" or proposal["status"] != "rejected":
                raise
        update: dict[str, Any] = {
            "rejections": proposal["rejections_in_chain"],
            "feedback_history": [feedback or NO_REASON],  # reducer operator.add appends
            "draft": None, "draft_error": None, "draft_attempts": 0,
        }
        if proposal["rejections_in_chain"] >= MAX_REJECTIONS:
            update["stop_reason"] = "max_rejections"
        return update

    def rejected_versions(proposal_id: str) -> list[dict[str, Any]]:
        """Every version in the chain with the seller's reason, oldest first (from the service)."""
        versions, current = [], proposal_id
        while current:
            proposal = service.get_proposal(current)
            versions.append({"version": {f: c["new"] for f, c in proposal["changes"].items()},
                             "feedback": proposal["feedback"] or ""})
            current = proposal["revision_of"]
        return versions[::-1]

    def draft_revision(state: ApprovalState) -> dict[str, Any]:
        """The only LLM call. No side effects, so it may retry on transient API errors."""
        messages = revision_messages(
            product=service.analytics().product(state["product_id"]), fields=state["fields"],
            rejected=rejected_versions(state["proposal_id"]), draft_error=state.get("draft_error"))
        draft = drafter.invoke(messages)
        attempts = state.get("draft_attempts", 0) + 1
        if draft is None:  # the model answered without the CaptionDraft tool call
            return {"draft": None, "draft_error": "no CaptionDraft was returned", "draft_attempts": attempts}
        return {"draft": draft.model_dump(), "draft_attempts": attempts}

    def existing_revision(proposal_id: str, product_id: str) -> dict[str, Any] | None:
        for proposal in service.list_proposals(product_id=product_id, limit=200):
            if proposal["revision_of"] == proposal_id:
                return proposal
        return None

    def submit_revision(state: ApprovalState) -> dict[str, Any]:
        """Create the revision in the service; a refused draft goes back to the LLM once."""
        draft = state.get("draft")
        try:
            if draft is None:
                raise SellerError("no_draft", state.get("draft_error") or "no draft")
            outside = [f for f in CAPTION_FIELDS if draft.get(f) is not None and f not in state["fields"]]
            if outside:
                raise SellerError("field_not_allowed",
                                  f"you may only change {state['fields']}; leave {outside} null")
            revision = service.propose_caption(
                state["product_id"], draft["reason"], actor=REVISION_ACTOR, revision_of=state["proposal_id"],
                **{f: draft.get(f) for f in state["fields"]})
        except SellerError as error:
            if error.code == "invalid_revision" and "already revised" in error.message:
                revision = existing_revision(state["proposal_id"], state["product_id"])  # ran before a crash
                if revision is None:
                    raise
            elif state.get("draft_attempts", 0) >= MAX_DRAFT_ATTEMPTS:
                return {"stop_reason": "revision_failed", "draft_error": error.message}
            else:
                return {"draft_error": error.message}
        return {"proposal_id": revision["proposal_id"], "rejections": revision["rejections_in_chain"],
                "decision": None, "feedback": "", "draft": None, "draft_error": None, "draft_attempts": 0}

    # Routing: read the state, return the next node.

    def route_decision(state: ApprovalState) -> Literal["apply", "record_rejection", "cancel_flow"]:
        return {"approve": "apply", "reject": "record_rejection", "cancel": "cancel_flow"}[state["decision"]]

    def after_rejection(state: ApprovalState) -> str:
        return END if state.get("stop_reason") else "draft_revision"

    def after_submit(state: ApprovalState) -> str:
        if state.get("stop_reason"):
            return END
        return "draft_revision" if state.get("draft_error") else "await_decision"

    builder = StateGraph(ApprovalState)
    builder.add_node("load", load)
    builder.add_node("await_decision", await_decision)
    builder.add_node("apply", apply)
    builder.add_node("cancel_flow", cancel_flow)
    builder.add_node("record_rejection", record_rejection)
    builder.add_node("draft_revision", draft_revision, retry_policy=DEFAULT_MODEL_RETRY)
    builder.add_node("submit_revision", submit_revision)

    builder.add_edge(START, "load")
    builder.add_edge("load", "await_decision")
    builder.add_conditional_edges("await_decision", route_decision, ["apply", "record_rejection", "cancel_flow"])
    builder.add_edge("apply", END)
    builder.add_edge("cancel_flow", END)
    builder.add_conditional_edges("record_rejection", after_rejection, ["draft_revision", END])
    builder.add_edge("draft_revision", "submit_revision")
    builder.add_conditional_edges("submit_revision", after_submit, ["await_decision", "draft_revision", END])
    return builder.compile(checkpointer=checkpointer)
