"""Drive the caption approval graph for the API: start, read, decide, retry.

The API knows proposal ids; the graph knows threads. ``ApprovalRunner`` maps
one onto the other and turns graph snapshots into a small view:

- **Thread = proposal chain.** Any proposal id of a chain (the first one or a
  revision) resolves to the chain's first proposal, whose id names the thread
  (``thread_config(root, store_id=...)``).
- **Started on first use.** The graph does not need to be started when the
  proposal is created (the chat agent proposes through the service and knows
  nothing about workflows): the first ``status`` or ``decide`` on a pending
  proposal starts the thread, which runs ``load`` and stops at the interrupt.
- **Decide only what the seller saw.** A decision names a proposal id; it must
  be the version the graph is waiting on, or it is refused (``stale_decision``)
  instead of approving a revision nobody read.
- **One decision at a time per thread.** A lock per thread serialises
  requests within this process (two clicks on "approve" resume the graph
  once; the second sees the closed thread). Several API processes would need
  a lock in the database instead.
- **Failures leave the thread resumable.** If a node raises (the LLM is down
  after its retries), the checkpoint before that node stays, the view says
  ``state="error"`` and ``retry`` runs the failed node again.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.types import Command, StateSnapshot

from sellerops.seller import SellerError, SellerService
from sellerops.workflows.approval.approval_graph import DecisionIn, build_approval_graph
from sellerops.workflows.approval.approval_state import thread_config

logger = logging.getLogger(__name__)

ViewState = Literal["waiting", "done", "error"]


class ApprovalRunner:
    """The approval graph of one store, behind proposal ids."""

    def __init__(self, service: SellerService, model: BaseChatModel, checkpointer: BaseCheckpointSaver) -> None:
        self.service = service
        self.graph = build_approval_graph(service, model, checkpointer=checkpointer)
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    # ------------------------------------------------------------------ public

    def status(self, proposal_id: str) -> dict[str, Any]:
        """The thread of this proposal's chain; starts it if the proposal is pending and has none."""
        root = self.root_of(proposal_id)
        with self._lock(root):
            self._ensure_started(root, proposal_id)
            return self._view(root)

    def decide(self, proposal_id: str, decision: DecisionIn, *, user: str) -> dict[str, Any]:
        """Resume the graph with the seller's answer (``Command(resume=...)``)."""
        root = self.root_of(proposal_id)
        with self._lock(root):
            self._ensure_started(root, proposal_id)
            view = self._view(root)
            if view["state"] != "waiting":
                raise SellerError("approval_closed",
                                  f"the approval of {proposal_id} is not waiting for a decision ({view['state']})")
            waiting_on = view["request"]["proposal"]["proposal_id"]
            if waiting_on != proposal_id:
                raise SellerError("stale_decision",
                                  f"{proposal_id} is not the version waiting for a decision; decide on {waiting_on}")
            self._run(root, Command(resume={**decision.model_dump(), "user": user}))
            return self._view(root)

    def retry(self, proposal_id: str) -> dict[str, Any]:
        """Run the node that failed again (``invoke(None)`` continues from the last checkpoint)."""
        root = self.root_of(proposal_id)
        with self._lock(root):
            if self._view(root)["state"] != "error":
                raise SellerError("approval_not_failed", f"the approval of {proposal_id} has nothing to retry")
            self._run(root, None)
            return self._view(root)

    def root_of(self, proposal_id: str) -> str:
        """The first proposal of the chain (raises ``proposal_not_found``)."""
        proposal = self.service.get_proposal(proposal_id)
        while proposal["revision_of"]:
            proposal = self.service.get_proposal(proposal["revision_of"])
        return proposal["proposal_id"]

    def config(self, root: str) -> dict[str, Any]:
        return thread_config(root, store_id=self.service.store_id)

    # ----------------------------------------------------------------- private

    def _lock(self, root: str) -> threading.Lock:
        with self._locks_guard:
            return self._locks.setdefault(root, threading.Lock())

    def _snapshot(self, root: str) -> StateSnapshot:
        return self.graph.get_state(self.config(root))

    def _ensure_started(self, root: str, proposal_id: str) -> None:
        snapshot = self._snapshot(root)
        if snapshot.values or snapshot.next:
            return
        # No thread yet. Start it on the pending proposal asked about (usually the
        # root; a revision when the chain began outside the graph): ``load`` reads
        # everything else (fields, rejections) from the service.
        if self.service.get_proposal(proposal_id)["status"] != "pending":
            raise SellerError("approval_not_found",
                              f"{proposal_id} has no approval workflow and is no longer pending")
        self._run(root, {"proposal_id": proposal_id})

    def _run(self, root: str, payload: Any) -> None:
        try:
            self.graph.invoke(payload, self.config(root))
        except Exception:
            # The checkpoint before the failing node stays; the view reports "error".
            logger.exception("caption approval step failed", extra={"thread_root": root})

    def _view(self, root: str) -> dict[str, Any]:
        snapshot = self._snapshot(root)
        values = snapshot.values
        interrupts = [item for task in snapshot.tasks for item in task.interrupts]
        state: ViewState = "waiting" if interrupts else "error" if snapshot.next else "done"
        return {
            "thread_id": self.config(root)["configurable"]["thread_id"],
            "root_proposal_id": root,
            "state": state,
            "proposal_id": values.get("proposal_id"),
            "request": interrupts[0].value if interrupts else None,
            "stop_reason": values.get("stop_reason"),
            "rejections": values.get("rejections", 0),
            "fields": values.get("fields", []),
            "draft_error": values.get("draft_error"),
            # The exception itself is only logged (it can carry provider details).
            "error": f"step {snapshot.next[0]!r} failed; retry it" if state == "error" else None,
        }
