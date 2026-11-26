"""Caption approval workflow (C3): propose → interrupt → approve / reject → revise.

Design: docs/c3-approval-graph-design.md
Tutorial: docs/tutorial-c3-approval-graph.md
"""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict


class ApprovalState(TypedDict, total=False):
    product_id: str
    fields: list[str]          # field yang boleh diubah: awalnya field usulan pertama, + add_fields saat reject
    proposal_id: str           # proposal pending saat ini
    decision: Literal["approve", "reject", "cancel"] | None
    feedback: str              # alasan reject terakhir ("" = seller tidak memberi alasan)
    decided_by: str            # "seller:<user>" dari resume
    feedback_history: Annotated[list[str], operator.add]  # semua alasan reject di rantai ini, untuk prompt revisi
    rejections: int            # salinan rejections_in_chain dari service (service = kebenaran)
    draft: dict | None         # hasil draft_revision sebelum disubmit
    draft_error: str | None    # error validasi dari service, untuk percobaan berikutnya
    draft_attempts: int
    stop_reason: Literal["applied", "stale", "cancelled", "max_rejections", "revision_failed"] | None


def thread_config(root_proposal_id: str, *, store_id: str | None = None) -> dict:
    """One thread per proposal chain, named after its first proposal.

    The API passes ``store_id`` so threads of different stores can never share
    a name (proposal ids are only unique within one store's database).
    """
    thread_id = f"caption-approval-{root_proposal_id}"
    return {"configurable": {"thread_id": f"{store_id}:{thread_id}" if store_id else thread_id}}
