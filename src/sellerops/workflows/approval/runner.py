"""Drive the caption approval graph for the API: start, read, decide, answer, retry.

The API knows ids; the graph knows threads. ``ApprovalRunner`` maps one onto
the other and turns graph snapshots into a small view:

- **Two kinds of thread.** A proposal chain that began outside the graph
  (the chat agent or the seller proposed) is one thread named after its
  first proposal (``caption-approval-CP-0001``). A caption request
  (``CR-0001``, C3b: "write a caption from scratch") is one thread named
  after the request (``caption-request-CR-0001``); the proposal the graph
  writes for it carries ``request_id``, so every proposal of that chain
  resolves to the request's thread. Any id of either kind works
  (``thread_of``).
- **Started on first use.** The first ``status`` / ``decide`` on a pending
  proposal, or the creation of a request, starts its thread. Graph input:
  ``{"proposal_id": ...}`` or ``{"request_id": ...}``; ``load`` reads the
  rest from the service.
- **Two kinds of wait.** The interrupt value's ``kind`` says what the graph
  waits for: ``"decision"`` (default; ``decide``) or ``"question"``
  (``answer``, see ``questions.py``). The view reports it as ``waiting_for``.
- **Decide only what the seller saw.** A decision names a proposal id; it must
  be the version the graph is waiting on, or it is refused (``stale_decision``)
  instead of approving a revision nobody read.
- **Answers are memory first.** ``answer`` stores the answers as product
  facts in the service, then resumes the graph with them.
- **One step at a time per thread.** A lock per thread serialises requests
  within this process. Several API processes would need a lock in the
  database instead.
- **Failures leave the thread resumable.** If a node raises (the LLM is down
  after its retries), the checkpoint before that node stays, the view says
  ``state="error"`` and ``retry`` runs the failed node again.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, StateSnapshot

from sellerops.seller import SellerError, SellerService
from sellerops.workflows.approval.approval_graph import DecisionIn, build_approval_graph
from sellerops.workflows.approval.approval_state import thread_config
from sellerops.workflows.approval.questions import AnswerIn

logger = logging.getLogger(__name__)

ViewState = Literal["waiting", "done", "error"]


@dataclass(frozen=True)
class Thread:
    """Which graph thread an id belongs to."""

    name: str                         # "caption-approval-CP-0001" | "caption-request-CR-0001"
    root_proposal_id: str | None      # first proposal of the chain (None: a request with no proposal yet)
    request_id: str | None = None


class ApprovalRunner:
    """The approval graph of one store, behind proposal and request ids."""

    def __init__(self, service: SellerService, model: BaseChatModel, checkpointer: BaseCheckpointSaver, *,
                 graph: CompiledStateGraph | None = None) -> None:
        self.service = service
        self.graph = graph if graph is not None else build_approval_graph(service, model, checkpointer=checkpointer)
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    # ------------------------------------------------------------------ public

    def status(self, any_id: str) -> dict[str, Any]:
        """The thread of a proposal (its chain) or request; starts it when it has none yet."""
        thread = self.thread_of(any_id)
        with self._lock(thread):
            self._ensure_started(thread, any_id)
            return self._view(thread)

    def start_request(self, product_id: str, *, user: str, fields: list[str] | None = None,
                      instruction: str = "") -> dict[str, Any]:
        """Create a caption request and run the graph until it waits (question or first proposal)."""
        request = self.service.request_caption(product_id, actor=f"seller:{user}", fields=fields,
                                               instruction=instruction)
        return self.status(request["request_id"])

    def decide(self, proposal_id: str, decision: DecisionIn, *, user: str) -> dict[str, Any]:
        """Resume the graph with the seller's decision (``Command(resume=...)``)."""
        thread = self.thread_of(proposal_id)
        with self._lock(thread):
            self._ensure_started(thread, proposal_id)
            view = self._waiting(thread, proposal_id, "decision")
            waiting_on = view["request"]["proposal"]["proposal_id"]
            if waiting_on != proposal_id:
                raise SellerError("stale_decision",
                                  f"{proposal_id} is not the version waiting for a decision; decide on {waiting_on}")
            self._run(thread, Command(resume={**decision.model_dump(), "user": user}))
            return self._view(thread)

    def answer(self, any_id: str, answer: AnswerIn, *, user: str) -> dict[str, Any]:
        """Store the seller's answers as product facts, then resume the graph with them."""
        thread = self.thread_of(any_id)
        with self._lock(thread):
            view = self._waiting(thread, any_id, "answer")
            asked = [q["key"] for q in view["request"]["questions"]]
            if set(answer.answers) != set(asked):
                raise SellerError("invalid_answer", f"answer exactly the questions asked: {asked}")
            product_id = view["request"].get("product_id") or view["product_id"]
            source = view["request"].get("request_id") or view["request"].get("proposal_id") or any_id
            self.service.remember_facts(product_id, answer.answers, actor=f"seller:{user}", source=source)
            self._run(thread, Command(resume={"answers": answer.answers, "user": user}))
            return self._view(thread)

    def retry(self, any_id: str) -> dict[str, Any]:
        """Run the node that failed again (``invoke(None)`` continues from the last checkpoint)."""
        thread = self.thread_of(any_id)
        with self._lock(thread):
            if self._view(thread)["state"] != "error":
                raise SellerError("approval_not_failed", f"the approval of {any_id} has nothing to retry")
            self._run(thread, None)
            return self._view(thread)

    def thread_of(self, any_id: str) -> Thread:
        """``CR-…`` → its request's thread; ``CP-…`` → its chain's thread (raises ``*_not_found``)."""
        if any_id.startswith("CR-"):
            request = self.service.get_request(any_id)
            return Thread(f"caption-request-{any_id}", request["proposal_id"], any_id)
        root = self.root_of(any_id)
        request_id = self.service.get_proposal(root)["request_id"]
        if request_id:
            return Thread(f"caption-request-{request_id}", root, request_id)
        return Thread(f"caption-approval-{root}", root)

    def root_of(self, proposal_id: str) -> str:
        """The first proposal of the chain (raises ``proposal_not_found``)."""
        proposal = self.service.get_proposal(proposal_id)
        while proposal["revision_of"]:
            proposal = self.service.get_proposal(proposal["revision_of"])
        return proposal["proposal_id"]

    def config(self, thread: Thread | str) -> dict[str, Any]:
        """Graph config of a thread (a ``str`` is a root proposal id, as in C3)."""
        if isinstance(thread, str):
            return thread_config(thread, store_id=self.service.store_id)
        return {"configurable": {"thread_id": f"{self.service.store_id}:{thread.name}"}}

    # ----------------------------------------------------------------- private

    def _lock(self, thread: Thread) -> threading.Lock:
        with self._locks_guard:
            return self._locks.setdefault(thread.name, threading.Lock())

    def _snapshot(self, thread: Thread) -> StateSnapshot:
        return self.graph.get_state(self.config(thread))

    def _ensure_started(self, thread: Thread, any_id: str) -> None:
        snapshot = self._snapshot(thread)
        if snapshot.values or snapshot.next:
            return
        if any_id.startswith("CR-"):
            if self.service.get_request(any_id)["status"] != "open":
                raise SellerError("approval_not_found", f"{any_id} has no approval workflow and is closed")
            self._run(thread, {"request_id": any_id})
            return
        # No thread yet. Start it on the pending proposal asked about (usually the
        # root; a revision when the chain began outside the graph): ``load`` reads
        # everything else (fields, rejections) from the service.
        if self.service.get_proposal(any_id)["status"] != "pending":
            raise SellerError("approval_not_found",
                              f"{any_id} has no approval workflow and is no longer pending")
        self._run(thread, {"proposal_id": any_id})

    def _waiting(self, thread: Thread, any_id: str, step: Literal["decision", "answer"]) -> dict[str, Any]:
        view = self._view(thread)
        if view["state"] != "waiting":
            raise SellerError("approval_closed", f"the approval of {any_id} is not waiting ({view['state']})")
        if view["waiting_for"] != step:
            raise SellerError(f"not_waiting_for_{step}",
                              f"the approval of {any_id} is waiting for {'an answer' if step == 'decision' else 'a decision'}")
        return view

    def _run(self, thread: Thread, payload: Any) -> None:
        try:
            self.graph.invoke(payload, self.config(thread))
        except Exception:
            # The checkpoint before the failing node stays; the view reports "error".
            logger.exception("caption approval step failed", extra={"thread_name": thread.name})

    def _view(self, thread: Thread) -> dict[str, Any]:
        snapshot = self._snapshot(thread)
        values = snapshot.values
        interrupts = [item for task in snapshot.tasks for item in task.interrupts]
        request = interrupts[0].value if interrupts else None
        state: ViewState = "waiting" if interrupts else "error" if snapshot.next else "done"
        stop_reason = values.get("stop_reason")
        if thread.request_id and state == "waiting" and request.get("kind") == "question" \
                and self.service.get_request(thread.request_id)["status"] == "cancelled":
            state, request, stop_reason = "done", None, "cancelled"   # the seller cancelled the request
        waiting_for = None
        if state == "waiting":
            waiting_for = "answer" if request.get("kind") == "question" else "decision"
        return {
            "thread_id": self.config(thread)["configurable"]["thread_id"],
            "root_proposal_id": thread.root_proposal_id or self._root_proposal(thread),
            "request_id": thread.request_id,
            "product_id": values.get("product_id"),
            "state": state,
            "waiting_for": waiting_for,
            "proposal_id": values.get("proposal_id"),
            "request": request,
            "stop_reason": stop_reason,
            "rejections": values.get("rejections", 0),
            "fields": values.get("fields", []),
            "draft_error": values.get("draft_error"),
            # The exception itself is only logged (it can carry provider details).
            "error": f"step {snapshot.next[0]!r} failed; retry it" if state == "error" else None,
        }

    def _root_proposal(self, thread: Thread) -> str | None:
        """A request's first proposal may appear after ``thread_of`` ran (the graph just wrote it)."""
        return self.service.get_request(thread.request_id)["proposal_id"] if thread.request_id else None
