"""Play the seller in the caption approval graph (C3) from the terminal.

    python scripts/try_caption_approval.py                     # real LLM (costs money), new session
    python scripts/try_caption_approval.py --fake              # free: a fake model writes the revisions
    python scripts/try_caption_approval.py --product FS101 --description "..."
    python scripts/try_caption_approval.py --resume CP-0001    # continue a session after quitting

What happens: the script plays the chat agent and creates a caption proposal,
starts the approval graph, and stops at ``interrupt``. You answer as the
seller (approve / reject with feedback / cancel). Each answer is sent as
``Command(resume=...)``, exactly what the API will do.

Everything lives in ``--workdir`` (default ``var/try-approval/``, gitignored):
``seller.sqlite`` (store data, proposals, audit log) and ``approvals.sqlite``
(graph checkpoints). Quit with ``q`` and continue later with ``--resume``;
delete the folder to start over with the original captions.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from langgraph.types import Command

from sellerops.harness.trajectory import open_sqlite_checkpointer
from sellerops.seller import SellerService
from sellerops.settings import get_settings
from sellerops.workflows.approval.approval_graph import build_approval_graph
from sellerops.workflows.approval.approval_state import thread_config
from sellerops.workflows.approval.studio import FakeDraftModel

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ("name", "description", "tags")


def show_proposal(request: dict[str, Any]) -> None:
    proposal = request["proposal"]
    print(f"\n── {proposal['proposal_id']} · {proposal['product_id']} · "
          f"rejections {request.get('rejections', 0)}/5 · by {proposal['created_by']}")
    print(f"   reason: {proposal['reason']}")
    for field, change in proposal["changes"].items():
        print(f"   {field}:\n     old: {change['old']!r}\n     new: {change['new']!r}")
    print(f"   fields the revision may change: {', '.join(request['fields'])}")
    if request.get("error"):
        print(f"   ! your last answer was refused: {request['error']}")


def ask_seller(request: dict[str, Any]) -> dict[str, Any] | None:
    while True:
        choice = input("\n[a]pprove / [r]eject / [c]ancel / [q]uit > ").strip().lower()
        if choice in ("a", "approve"):
            return {"decision": "approve"}
        if choice in ("c", "cancel"):
            return {"decision": "cancel"}
        if choice in ("q", "quit"):
            return None
        if choice in ("r", "reject"):
            feedback = input("feedback (enter = no reason) > ").strip()
            others = [f for f in FIELDS if f not in request["fields"]]
            extra = input(f"also allow changing ({', '.join(others)}; comma-separated, enter = none) > ") \
                if others else ""
            return {"decision": "reject", "feedback": feedback,
                    "add_fields": [f.strip() for f in extra.split(",") if f.strip()]}


def show_outcome(service: SellerService, state: dict[str, Any]) -> None:
    product = service.analytics().product(state["product_id"])
    print(f"\n== stop_reason: {state.get('stop_reason')}"
          + (f" ({state['draft_error']})" if state.get("draft_error") else ""))
    print(f"   caption now: name={product['name']!r}\n                description={product['description']!r}\n"
          f"                tags={product['tags']}")
    print("   audit log (this product):")
    for entry in reversed(service.audit_log(product_id=state["product_id"], limit=20)):
        print(f"     {entry['at']}  {entry['actor']:<24} {entry['action']:<10} {entry.get('proposal_id') or ''}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--product", default="FS180")
    parser.add_argument("--description", default=None, help="the agent's first proposal (default: a long one)")
    parser.add_argument("--resume", metavar="CP-XXXX", help="continue the session started by this proposal")
    parser.add_argument("--workdir", type=Path, default=ROOT / "var" / "try-approval")
    parser.add_argument("--user", default="nigka", help="seller name recorded in the audit log")
    parser.add_argument("--fake", action="store_true", help="no LLM calls (free)")
    args = parser.parse_args()

    settings = get_settings()
    service = SellerService.open(args.workdir / "seller.sqlite", catalog_path=settings.catalog_path,
                                 sim_dir=settings.data_dir / "seller_sim")
    if args.fake:
        model: Any = FakeDraftModel()
    else:
        from sellerops.harness.models import get_chat_model

        model = get_chat_model(settings).with_config(tags=["try-approval"])
        print(f"model: {settings.llm_provider} / {settings.llm_model} (real calls, costs money)")
    graph = build_approval_graph(service, model, checkpointer=open_sqlite_checkpointer(
        args.workdir / "approvals.sqlite"))

    if args.resume:
        config = thread_config(args.resume)
        if not graph.get_state(config).next:
            print(f"no waiting session for {args.resume}")
            return 1
        out = {"__interrupt__": graph.get_state(config).tasks[0].interrupts}
    else:
        product = service.analytics().product(args.product)
        description = args.description or (product["description"] + " It is perfect for casual outings, "
                                           "weekend brunches, park walks and relaxed days in the city.")
        proposal = service.propose_caption(args.product, "make it more engaging", actor="agent",
                                           description=description)
        print(f"agent proposed {proposal['proposal_id']}; session: --resume {proposal['proposal_id']}")
        config = thread_config(proposal["proposal_id"])
        out = graph.invoke({"proposal_id": proposal["proposal_id"]}, config)

    while "__interrupt__" in out:
        request = out["__interrupt__"][0].value
        show_proposal(request)
        answer = ask_seller(request)
        if answer is None:
            print(f"paused. continue with: python scripts/try_caption_approval.py --resume "
                  f"{config['configurable']['thread_id'].removeprefix('caption-approval-')}"
                  + (" --fake" if args.fake else ""))
            return 0
        if answer["decision"] == "reject":
            print("drafting a revision…")
        out = graph.invoke(Command(resume={"user": args.user, **answer}), config)

    show_outcome(service, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
