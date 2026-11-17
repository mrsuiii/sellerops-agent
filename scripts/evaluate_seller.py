"""Run the seller agent on the eval tasks and save the results (Session A3).

    python scripts/evaluate_seller.py --user scripted --k 1          # cheap smoke run
    python scripts/evaluate_seller.py --user llm --k 3 --judge       # full dev eval
    python scripts/evaluate_seller.py --task omzet --k 1             # one task

Calls the Claude API (agent, and the user simulator with ``--user llm``) and,
with ``--judge``, the NVIDIA endpoint. Output: ``results/evaluation/<run_id>/``
with ``config.json``, ``records.jsonl`` (one graded conversation per line, with
transcript and tool history for error analysis), ``summary.json`` and
``summary.md``. The frozen test split needs ``--allow-test``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import uuid
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sellerops.evaluation.judge import MessageJudge, nvidia_judge_model  # noqa: E402
from sellerops.evaluation.seller_eval import run_eval, summarize, summary_markdown  # noqa: E402
from sellerops.evaluation.seller_tasks import load_tasks  # noqa: E402
from sellerops.evaluation.user_sim import LLMUser, ScriptedUser  # noqa: E402
from sellerops.harness.models import get_chat_model, get_fallback_models  # noqa: E402
from sellerops.harness.trajectory import open_sqlite_checkpointer  # noqa: E402
from sellerops.schemas import MAX_AGENT_STEPS  # noqa: E402
from sellerops.settings import get_settings  # noqa: E402
from sellerops.simulator.store import default_sim_dir  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--allow-test", action="store_true", help="the test split is frozen; final reports only")
    parser.add_argument("--k", type=int, default=3, help="trials per task")
    parser.add_argument("--user", default="scripted", choices=["scripted", "llm"])
    parser.add_argument("--task", action="append", default=[], help="substring of task_id (repeatable)")
    parser.add_argument("--family", action="append", default=[])
    parser.add_argument("--max-turns", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=MAX_AGENT_STEPS)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--judge", action="store_true", help="NVIDIA message judge on verifier passes")
    parser.add_argument("--judge-model", default=None, help="default: SELLEROPS_JUDGE_MODEL / LLM_MODEL")
    parser.add_argument("--user-effort", default="low", help="effort for the LLM user simulator")
    parser.add_argument("--tag", action="append", default=[], help="extra LangSmith tag (repeatable)")
    parser.add_argument("--agent-provider", choices=["anthropic", "openai_compatible"], default=None,
                        help="agent only (the LLM user stays on SELLEROPS_LLM_*), e.g. openai_compatible = NVIDIA")
    parser.add_argument("--agent-model", default=None, help="agent model id for --agent-provider")
    parser.add_argument("--save-threads", action="store_true",
                        help="keep every thread in <out>/threads.sqlite (for scripts/replay_eval_thread.py)")
    parser.add_argument("--max-turn-tokens", type=int, default=None, help="per-turn token budget")
    parser.add_argument("--max-turn-seconds", type=float, default=None, help="per-turn time budget")
    parser.add_argument("--max-context-tokens", type=int, default=None,
                        help="summarize older messages when the prompt exceeds this")
    parser.add_argument("--out", type=Path, default=None, help="default: results/evaluation/<run_id>")
    return parser.parse_args()


def git_commit() -> str | None:
    if os.environ.get("SELLEROPS_GIT_COMMIT"):  # set by the eval Docker image (no .git inside)
        return os.environ["SELLEROPS_GIT_COMMIT"]
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> int:
    args = parse_args()
    if args.split == "test" and not args.allow_test:
        sys.exit("The test split is frozen. Use --allow-test only for a final report, never for tuning.")
    settings = get_settings()
    if settings.llm_provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set (Docker: put it in .env.eval).")
    tasks = load_tasks(args.split)
    if args.task:
        tasks = [t for t in tasks if any(s in t["task_id"] for s in args.task)]
    if args.family:
        tasks = [t for t in tasks if t["family"] in args.family]
    if not tasks:
        sys.exit("no task matches the filters")

    # The agent may run on another provider/model than the user simulator (comparison runs).
    agent_settings = settings
    if args.agent_provider or args.agent_model:
        agent_settings = replace(settings, llm_provider=args.agent_provider or settings.llm_provider,
                                 llm_model=args.agent_model or settings.llm_model, llm_fallback_models=())
    agent_model = get_chat_model(agent_settings)
    fallback_models = get_fallback_models(agent_settings)  # SELLEROPS_LLM_FALLBACK_MODELS
    if args.user == "llm":
        user_kwargs = {"output_config": {"effort": args.user_effort}} if settings.llm_provider == "anthropic" else {}
        user_model = get_chat_model(settings, max_tokens=4000, **user_kwargs)

        def make_user():
            return LLMUser(user_model)
    else:
        make_user = ScriptedUser  # fresh per trial: it remembers revealed facts
    judge = MessageJudge(nvidia_judge_model(args.judge_model)) if args.judge else None

    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_id = f"{stamp}-{args.split}-{args.user}-{agent_settings.llm_model.replace('/', '_')}-{uuid.uuid4().hex[:4]}"
    out = args.out or settings.results_dir / "evaluation" / run_id
    manifest = default_sim_dir() / "tasks.manifest.json"
    config = {
        "run_id": run_id, "split": args.split, "k": args.k, "user": args.user,
        "user_effort": args.user_effort if args.user == "llm" else None,
        "tasks": [t["task_id"] for t in tasks], "max_turns": args.max_turns, "max_steps": args.max_steps,
        "agent_provider": agent_settings.llm_provider, "agent_model": agent_settings.llm_model,
        "agent_effort": agent_settings.llm_effort if agent_settings.llm_provider == "anthropic" else None,
        "user_model": settings.llm_model if args.user == "llm" else None,
        "fallback_models": list(agent_settings.llm_fallback_models),
        "max_turn_tokens": args.max_turn_tokens, "max_turn_seconds": args.max_turn_seconds,
        "max_context_tokens": args.max_context_tokens, "save_threads": args.save_threads,
        "judge_model": judge.metadata["model"] if judge else None,
        "judge_rubric": judge.metadata["rubric_version"] if judge else None,
        "data_version": tasks[0].get("data_version"),
        "tasks_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "git_commit": git_commit(), "langsmith_project": os.environ.get("LANGSMITH_PROJECT"),
        "started_at": stamp,
    }
    print(f"{run_id}: {len(tasks)} tasks × k={args.k} → {out}", flush=True)

    def progress(record, done, total):
        verdict = {1: "PASS", 0: "fail", None: "ERROR"}[record["score"]]
        why = record.get("error") or ", ".join(record.get("failed", [])) or record.get("stop_reason")
        print(f"[{done}/{total}] {verdict:5} {record['task_id']} t{record['trial']} "
              f"({record.get('latency_s')}s) {why}", flush=True)

    out.mkdir(parents=True, exist_ok=True)
    checkpointer = open_sqlite_checkpointer(out / "threads.sqlite") if args.save_threads else None
    graph_options = {"fallback_models": fallback_models, "max_turn_tokens": args.max_turn_tokens,
                     "max_turn_seconds": args.max_turn_seconds, "max_context_tokens": args.max_context_tokens}
    records = run_eval(tasks, k=args.k, concurrency=args.concurrency, progress=progress,
                       checkpointer=checkpointer, graph_options=graph_options,
                       agent_model=agent_model, make_user=make_user, judge=judge, max_turns=args.max_turns,
                       max_steps=args.max_steps, run_id=run_id, tags=[args.split, args.user, *args.tag],
                       metadata={"model": agent_settings.llm_model, "provider": agent_settings.llm_provider,
                                 "user": args.user})
    summary = summarize(records, k=args.k)
    markdown = summary_markdown(summary, config)

    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=1) + "\n")
    with (out / "records.jsonl").open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n")
    (out / "summary.md").write_text(markdown)
    print("\n" + markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
