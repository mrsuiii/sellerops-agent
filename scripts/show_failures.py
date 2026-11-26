"""Print failed eval conversations as readable trajectories (for error analysis).

    python scripts/show_failures.py                      # latest run, verifier failures
    python scripts/show_failures.py --judge              # verifier passes the judge failed
    python scripts/show_failures.py --all --task caption # every trial of matching tasks
    python scripts/show_failures.py --run results/evaluation/<run_id>

Reads ``records.jsonl`` written by ``scripts/evaluate_seller.py``; no API calls.
For each conversation: verdict and failed checks, then the turns in order
(seller message, the agent's tool calls with arguments and a short result,
``ask_user`` questions, the final message and data), then the judge verdict.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "evaluation"


def latest_run() -> Path:
    runs = sorted(p for p in RESULTS.glob("*") if (p / "records.jsonl").exists())
    if not runs:
        sys.exit(f"no runs in {RESULTS}")
    return runs[-1]


def short(value: Any, limit: int) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return text if len(text) <= limit else text[:limit] + " …"


def show(record: dict[str, Any], width: int) -> None:
    verdict = {1: "PASS", 0: "FAIL", None: "ERROR"}[record["score"]]
    print("=" * 100)
    print(f"{verdict}  {record['task_id']}  trial {record['trial']}  ({record.get('latency_s')}s)")
    if record.get("error"):
        print(f"  error: {record['error']}")
    for name, check in (record.get("checks") or {}).items():
        mark = "ok  " if check["pass"] else "FAIL"
        print(f"  [{mark}] {name}: {short(check['detail'], width)}")
    metrics = record.get("metrics") or {}
    if metrics:
        print(f"  turns {metrics.get('agent_turns')}, tool calls {metrics.get('tool_calls')}, "
              f"asked {metrics.get('asked_clarification')} (needed {metrics.get('needs_clarification')}), "
              f"ended_by {record.get('ended_by')}, stop {record.get('stop_reason')}")

    print("  --- trajectory")
    if record.get("messages"):  # exact interleaving (runs since A4 store the thread's messages)
        for message in record["messages"]:
            if message["role"] == "human":
                print(f"  seller: {message['text']}")
            elif message["role"] == "ai":
                if message.get("text"):
                    print(f"  agent says: {short(message['text'], width)}")
                for call in message.get("tool_calls", []):
                    if call["name"] == "ask_user":
                        print(f"  agent asks: {call['args'].get('question')}")
                    elif call["name"] == "final":
                        print(f"  agent final message: {call['args'].get('message')}")
                        print(f"  agent final data: {short(call['args'].get('data'), width)}")
                    else:
                        print(f"    tool {call['name']}({short(call['args'], 120)})")
            elif message["role"] == "tool":
                label = "seller" if message.get("name") == "ask_user" else "      ->"
                print(f"  {label}: {short(message['text'], width)}" if label == "seller"
                      else f"{label} {short(message['text'], width)}")
        history = []
        turns = []
    else:
        history = list(record.get("history") or [])
        turns = record.get("turns") or []
    transcript = record.get("transcript") or []
    # Older records do not split tool calls per turn: all of them are shown in the last turn,
    # after the clarification questions. Use the LangSmith trace for the exact interleaving.
    seller_lines = [t for t in transcript if t["role"] == "user"]
    if turns:
        print(f"  seller: {seller_lines[0]['text'] if seller_lines else '?'}")
    for index, turn in enumerate(turns):
        action = turn.get("final_action") or {}
        if index == len(turns) - 1:
            for step in history:
                print(f"    tool {step['tool']}({short(step['arguments'], 120)})")
                print(f"      -> {short(step['observation'], width)}")
            history = []
        if turn.get("stop_reason") == "ask_user":
            print(f"  agent asks: {action.get('question')}")
            if turn.get("user_reply"):
                print(f"  seller: {turn['user_reply']}")
        elif turn.get("stop_reason") == "final":
            print(f"  agent final message: {action.get('message')}")
            print(f"  agent final data: {short(action.get('data'), width)}")
        else:
            print(f"  turn ended: {turn.get('stop_reason')}")
    if record.get("proposals"):
        print(f"  proposals: {short(record['proposals'], width)}")
    judge = record.get("judge")
    if judge:
        print(f"  --- judge: {judge.get('status')} pass={judge.get('pass')} "
              f"faithful={judge.get('faithful')} answers_request={judge.get('answers_request')}")
        print(f"      {judge.get('reason')}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, default=None, help="run directory (default: latest)")
    parser.add_argument("--task", action="append", default=[], help="substring of task_id")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--all", action="store_true", help="every trial, passes included")
    group.add_argument("--judge", action="store_true", help="verifier passes the judge failed")
    parser.add_argument("--width", type=int, default=300, help="max characters per tool result")
    args = parser.parse_args()

    run = args.run or latest_run()
    records = [json.loads(line) for line in (run / "records.jsonl").open()]
    if args.task:
        records = [r for r in records if any(s in r["task_id"] for s in args.task)]
    if args.judge:
        records = [r for r in records if (r.get("judge") or {}).get("pass") is False]
    elif not args.all:
        records = [r for r in records if r["score"] != 1]
    print(f"{run.name}: {len(records)} conversation(s)")
    for record in records:
        show(record, args.width)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
