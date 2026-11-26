"""Replay one eval conversation from a checkpoint (Session A4: reproduce a failure).

    python scripts/evaluate_seller.py --task caption --k 1 --save-threads   # keep threads
    python scripts/replay_eval_thread.py --task caption --list              # show checkpoints
    python scripts/replay_eval_thread.py --task caption                     # replay last decision
    python scripts/replay_eval_thread.py --task caption --checkpoint <id> --times 3

What happens:
1. The thread is loaded from ``<run>/threads.sqlite`` (written with --save-threads).
2. A fresh simulator episode is rebuilt to the checkpoint: every successful
   seller tool call in the checkpoint's messages is run again on the episode,
   in order (the simulator is deterministic, so results and proposals match).
3. ``graph.invoke(None, {thread_id, checkpoint_id})`` continues the thread
   from that checkpoint with the current model: same state, new sample. With
   ``--times N`` the same point is replayed N times (each is its own fork) to
   see how often the failure repeats.
4. The turn's outcome is printed; a ``final`` is graded by the verifier.

Calls the Claude API (one turn per replay).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sellerops.evaluation.prompts import SELLER_SYSTEM_PROMPT  # noqa: E402
from sellerops.evaluation.seller_tasks import load_tasks  # noqa: E402
from sellerops.evaluation.seller_tools import SELLER_TOOL_NAMES, make_seller_tools  # noqa: E402
from sellerops.evaluation.seller_verifier import verify_conversation  # noqa: E402
from sellerops.evaluation.terminal_tools import TERMINAL_TOOLS  # noqa: E402
from sellerops.harness.graph import build_agent_graph  # noqa: E402
from sellerops.harness.models import get_chat_model  # noqa: E402
from sellerops.harness.trajectory import (  # noqa: E402
    checkpoint_before_last_model_call,
    export_trajectory,
    open_sqlite_checkpointer,
    replay,
)
from sellerops.simulator import SellerEpisode  # noqa: E402

RESULTS = ROOT / "results" / "evaluation"


def rehydrate(episode: SellerEpisode, messages: list) -> int:
    """Re-run the successful seller tool calls found in ``messages``; returns how many."""
    calls = {}
    count = 0
    for message in messages:
        for call in getattr(message, "tool_calls", None) or []:
            calls[call["id"]] = call
        if message.type == "tool" and message.status != "error" and message.name in SELLER_TOOL_NAMES:
            call = calls.get(message.tool_call_id)
            if call is not None:
                episode.call(call["name"], call["args"])
                count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, default=None, help="run directory (default: latest with threads)")
    parser.add_argument("--task", required=True, help="substring of task_id")
    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument("--checkpoint", default=None, help="checkpoint id (default: before the last model call)")
    parser.add_argument("--list", action="store_true", help="list the thread's checkpoints and exit")
    parser.add_argument("--times", type=int, default=1, help="replay the same checkpoint N times")
    args = parser.parse_args()

    run = args.run or max((p for p in RESULTS.glob("*") if (p / "threads.sqlite").exists()), default=None)
    if run is None or not (run / "threads.sqlite").exists():
        sys.exit("no run with threads.sqlite; run evaluate_seller.py with --save-threads first")
    records = [json.loads(line) for line in (run / "records.jsonl").open()]
    record = next((r for r in records if args.task in r["task_id"] and r["trial"] == args.trial), None)
    if record is None:
        sys.exit(f"no record for task~{args.task!r} trial {args.trial} in {run.name}")
    task = next(t for split in ("dev", "test") for t in load_tasks(split) if t["task_id"] == record["task_id"])
    thread = {"configurable": {"thread_id": record["thread_id"]}}
    saver = open_sqlite_checkpointer(run / "threads.sqlite")

    if args.list:
        viewer = build_agent_graph(get_chat_model(), make_seller_tools(SellerEpisode()),
                                   system_prompt=SELLER_SYSTEM_PROMPT, terminal_tools=TERMINAL_TOOLS,
                                   checkpointer=saver)
        for c in export_trajectory(viewer, thread)["checkpoints"]:
            print(f"{c['checkpoint_id']}  step {c['step']:>3}  next {c['next']}  "
                  f"messages {c['messages']}  stop {c['stop_reason']}")
        return 0

    print(f"{run.name} / {record['thread_id']} (original verdict: score={record['score']}, "
          f"failed={record.get('failed')})")
    model = get_chat_model()
    # Resolve the checkpoint once: each replay adds a fork to the thread, and "the latest
    # checkpoint before a model call" would otherwise point into the previous replay.
    probe = build_agent_graph(model, make_seller_tools(SellerEpisode()), system_prompt=SELLER_SYSTEM_PROMPT,
                              terminal_tools=TERMINAL_TOOLS, checkpointer=saver)
    checkpoint = args.checkpoint or checkpoint_before_last_model_call(probe, thread)
    if checkpoint is None:
        sys.exit("no checkpoint before a model call in this thread")
    for attempt in range(args.times):
        episode = SellerEpisode()
        graph = build_agent_graph(model, make_seller_tools(episode), system_prompt=SELLER_SYSTEM_PROMPT,
                                  terminal_tools=TERMINAL_TOOLS, checkpointer=saver)
        at = graph.get_state({"configurable": {**thread["configurable"], "checkpoint_id": checkpoint}})
        restored = rehydrate(episode, at.values.get("messages", []))
        state = replay(graph, record["thread_id"], checkpoint,
                       tags=["replay"], metadata={"replay_of": record["thread_id"], "attempt": attempt})
        print(f"\n--- replay {attempt + 1}/{args.times} from {checkpoint} "
              f"({len(at.values.get('messages', []))} messages, {restored} tool calls restored)")
        new = state["messages"][len(at.values.get("messages", [])):]
        for message in new:
            for call in getattr(message, "tool_calls", None) or []:
                print(f"  call {call['name']}({json.dumps(call['args'], ensure_ascii=False)[:300]})")
            if message.type == "tool":
                print(f"    -> {message.text[:300]}")
        print(f"  stop_reason: {state.get('stop_reason')}")
        if state.get("stop_reason") == "final":
            result = {"final_action": state["final_action"], "stop_reason": "final", "turns": []}
            report = verify_conversation(task, episode.snapshot(), result, store=episode.store)
            print(f"  verifier: {report['decision']} failed={report['failed']}")
            for name in report["failed"]:
                print(f"    {name}: {report['checks'][name]['detail']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
