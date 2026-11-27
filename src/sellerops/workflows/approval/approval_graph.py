"""Caption approval graph (C3 + C3b): one thread per proposal chain or caption request.

    START → load ─┬─ {"proposal_id"} ──────────────────► await_decision
                  └─ {"request_id"} (C3b, from scratch) ► draft_caption

    await_decision ──(interrupt: the seller decides)
      ├─ approve → apply ─────────────────────────────► END  applied | stale
      ├─ cancel  → cancel_flow ───────────────────────► END  cancelled
      └─ reject  → record_rejection
                     ├─ 5 rejections ─────────────────► END  max_rejections
                     └─ draft_caption

    draft_caption (LLM: CaptionDraft | AskSeller)
      ├─ AskSeller ──► ask_seller ──(interrupt: the seller answers)──► draft_caption
      └─ CaptionDraft ► submit_draft
                          ├─ ok ──────────────────────► await_decision (the new version)
                          ├─ refused, 1st try ────────► draft_caption (with the error)
                          └─ refused again ───────────► END  revision_failed

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
  proposal's fields (or the request's) plus ``add_fields`` from a rejection.
- C3b: the draft sees three sources (product data, photos when the model
  has vision, seller facts) and may ask the seller once per version
  (``asked_this_version``), only for facts not already known (an
  "unknown" answer counts as known: no asking twice). Questions go through
  ``ask_seller`` (``interrupt``, contract in ``questions.py``); the runner
  stores the answers as product facts, and the draft reads facts from the
  service plus the last answers in the state, so the graph also works
  without the runner (Studio, tests).
- Images are loaded inside ``draft_caption`` on every run, never kept in the
  state: the checkpoint would store the base64 at every step. Loading is
  cached by URL and content-addressed, so a retry or replay sends the same
  bytes; the state keeps only the refs (``draft_inputs``).
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from sellerops.harness.graph import DEFAULT_MODEL_RETRY
from sellerops.harness.models import supports_vision
from sellerops.seller import MAX_REJECTIONS, SellerError, SellerService
from sellerops.seller.images import ProductImages, to_content_block
from sellerops.workflows.approval.approval_prompts import NO_REASON, AskSeller, CaptionDraft, draft_messages
from sellerops.workflows.approval.approval_state import ApprovalState
from sellerops.workflows.approval.questions import Answer, SellerQuestion, question_request

MAX_DRAFT_ATTEMPTS = 2
IMAGE_LIMIT = 1   # one 1000 px photo (~1.3k tokens) is enough for colour, pattern and cut
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
                         checkpointer: BaseCheckpointSaver | None = None,
                         images: ProductImages | None = None) -> CompiledStateGraph:
    """``images``: product photos for the draft (C3b); ``None`` or a model without vision = text only."""
    see_photos = images is not None and supports_vision(model)

    def load(state: ApprovalState) -> dict[str, Any]:
        """Fill the state from the service; the input is only ``proposal_id`` or ``request_id``."""
        if state.get("request_id") and not state.get("proposal_id"):
            request = service.get_request(state["request_id"])
            return {"product_id": request["product_id"], "fields": request["fields"],
                    "instruction": request["instruction"], "rejections": 0}
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
            "kind": "decision",
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
    draft_only = model.bind_tools([CaptionDraft], tool_choice="any")
    draft_or_ask = model.bind_tools([CaptionDraft, AskSeller], tool_choice="any")

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
            "questions": None, "asked_this_version": False,   # a new version may ask again
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

    def known_facts(state: ApprovalState) -> dict[str, str | None]:
        """Seller facts from memory, plus the answers just given (the runner stores them too)."""
        facts = {key: fact["value"] for key, fact in service.product_facts(state["product_id"]).items()}
        return {**facts, **state.get("seller_answers", {})}

    def photos(product_id: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Content blocks and refs of the product photos (none without vision; failures are skipped)."""
        if not see_photos:
            return [], []
        urls = service.product_images(product_id)
        loaded = images.load(urls[1:] or urls, limit=IMAGE_LIMIT)   # full photos before the 300 px thumbnail
        return [to_content_block(item) for item in loaded], [asdict(item.ref) for item in loaded]

    def draft_caption(state: ApprovalState) -> dict[str, Any]:
        """The only LLM call: a first draft or a revision, or questions for the seller. No side effects."""
        facts = known_facts(state)
        blocks, refs = photos(state["product_id"])
        can_ask = not state.get("asked_this_version")
        proposal_id = state.get("proposal_id")
        messages = draft_messages(
            product=service.analytics().product(state["product_id"]), fields=state["fields"], facts=facts,
            rejected=rejected_versions(proposal_id) if proposal_id else [],
            instruction=state.get("instruction", ""), draft_error=state.get("draft_error"),
            images=blocks, can_ask=can_ask)
        reply = (draft_or_ask if can_ask else draft_only).invoke(messages)
        attempts = state.get("draft_attempts", 0) + 1
        inputs = {key: value for key, value in (("images", refs), ("seller_facts", facts)) if value}
        update: dict[str, Any] = {"draft": None, "questions": None, "draft_inputs": inputs or None,
                                  "draft_attempts": attempts}
        tool_call = (reply.tool_calls or [None])[0]
        try:
            if tool_call and tool_call["name"] == "AskSeller" and can_ask:
                ask = AskSeller.model_validate(tool_call["args"])
                new = [q for q in ask.questions if q.key not in facts]   # never ask what is known (or unknown)
                if new:
                    return {**update, "questions": [q.model_dump() for q in new], "draft_error": None,
                            "draft_attempts": attempts - 1}   # asking is not a failed draft
                return {**update, "asked_this_version": True,
                        "draft_error": "every fact you asked about is already in the seller facts; draft now"}
            if tool_call and tool_call["name"] == "CaptionDraft":
                return {**update, "draft": CaptionDraft.model_validate(tool_call["args"]).model_dump(),
                        "draft_error": None}
        except ValidationError as error:
            return {**update, "draft_error": f"invalid {tool_call['name']}: "
                                             f"{error.errors(include_url=False, include_context=False)}"}
        return {**update, "draft_error": "answer with exactly one CaptionDraft tool call"}

    def ask_seller(state: ApprovalState) -> dict[str, Any]:
        """Wait for the seller's answers. Runs again from the top on every resume: reads only."""
        request = question_request([SellerQuestion(**q) for q in state["questions"]],
                                   product_id=state["product_id"], request_id=state.get("request_id"),
                                   proposal_id=state.get("proposal_id"))
        asked = {q["key"] for q in state["questions"]}
        while True:
            raw = interrupt(request)
            try:
                answer = Answer.model_validate(raw)
                if set(answer.answers) != asked:
                    raise ValueError(f"answer exactly {sorted(asked)}")
                break
            except (ValidationError, ValueError) as error:
                detail = error.errors(include_url=False, include_context=False) \
                    if isinstance(error, ValidationError) else str(error)
                request = {**request, "error": detail}
        return {"seller_answers": {**state.get("seller_answers", {}), **answer.answers},
                "questions": None, "asked_this_version": True}

    def existing_proposal(state: ApprovalState) -> dict[str, Any] | None:
        """The version a previous run of ``submit_draft`` created before a crash, if any."""
        if state.get("proposal_id"):
            for proposal in service.list_proposals(product_id=state["product_id"], limit=200):
                if proposal["revision_of"] == state["proposal_id"]:
                    return proposal
            return None
        created = service.get_request(state["request_id"])["proposal_id"]
        return service.get_proposal(created) if created else None

    def submit_draft(state: ApprovalState) -> dict[str, Any]:
        """Create the first version (for a request) or the revision; a refused draft goes back once."""
        draft = state.get("draft")
        first = not state.get("proposal_id")
        try:
            if draft is None:
                raise SellerError("no_draft", state.get("draft_error") or "no draft")
            outside = [f for f in CAPTION_FIELDS if draft.get(f) is not None and f not in state["fields"]]
            if outside:
                raise SellerError("field_not_allowed",
                                  f"you may only change {state['fields']}; leave {outside} null")
            link = {"request_id": state["request_id"]} if first else {"revision_of": state["proposal_id"]}
            proposal = service.propose_caption(
                state["product_id"], draft["reason"], actor=REVISION_ACTOR, inputs=state.get("draft_inputs"),
                **link, **{f: draft.get(f) for f in state["fields"]})
        except SellerError as error:
            if first and service.get_request(state["request_id"])["status"] == "cancelled":
                return {"stop_reason": "cancelled"}   # the seller cancelled the request meanwhile
            already = (error.code == "invalid_revision" and "already revised" in error.message) or \
                      (first and error.code == "invalid_transition")
            proposal = existing_proposal(state) if already else None   # ran before a crash
            if proposal is None:
                if already:
                    raise
                if state.get("draft_attempts", 0) >= MAX_DRAFT_ATTEMPTS:
                    return {"stop_reason": "revision_failed", "draft_error": error.message}
                return {"draft_error": error.message}
        return {"proposal_id": proposal["proposal_id"], "rejections": proposal["rejections_in_chain"],
                "decision": None, "feedback": "", "draft": None, "draft_error": None, "draft_attempts": 0,
                "draft_inputs": None, "seller_answers": {}}

    # Routing: read the state, return the next node.

    def route_decision(state: ApprovalState) -> Literal["apply", "record_rejection", "cancel_flow"]:
        return {"approve": "apply", "reject": "record_rejection", "cancel": "cancel_flow"}[state["decision"]]

    def after_load(state: ApprovalState) -> str:
        return "await_decision" if state.get("proposal_id") else "draft_caption"

    def after_rejection(state: ApprovalState) -> str:
        return END if state.get("stop_reason") else "draft_caption"

    def after_draft(state: ApprovalState) -> str:
        return "ask_seller" if state.get("questions") else "submit_draft"

    def after_submit(state: ApprovalState) -> str:
        if state.get("stop_reason"):
            return END
        return "draft_caption" if state.get("draft_error") else "await_decision"

    builder = StateGraph(ApprovalState)
    builder.add_node("load", load)
    builder.add_node("await_decision", await_decision)
    builder.add_node("apply", apply)
    builder.add_node("cancel_flow", cancel_flow)
    builder.add_node("record_rejection", record_rejection)
    builder.add_node("draft_caption", draft_caption, retry_policy=DEFAULT_MODEL_RETRY)
    builder.add_node("ask_seller", ask_seller)
    builder.add_node("submit_draft", submit_draft)

    builder.add_edge(START, "load")
    builder.add_conditional_edges("load", after_load, ["await_decision", "draft_caption"])
    builder.add_conditional_edges("await_decision", route_decision, ["apply", "record_rejection", "cancel_flow"])
    builder.add_edge("apply", END)
    builder.add_edge("cancel_flow", END)
    builder.add_conditional_edges("record_rejection", after_rejection, ["draft_caption", END])
    builder.add_conditional_edges("draft_caption", after_draft, ["ask_seller", "submit_draft"])
    builder.add_edge("ask_seller", "draft_caption")
    builder.add_conditional_edges("submit_draft", after_submit, ["await_decision", "draft_caption", END])
    return builder.compile(checkpointer=checkpointer)
