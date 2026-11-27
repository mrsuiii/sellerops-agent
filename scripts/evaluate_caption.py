"""Run the caption workflow eval (C3/C3b) and save the results.

    python scripts/evaluate_caption.py --k 3 --judge                       # Claude agent, NVIDIA judge
    python scripts/evaluate_caption.py --k 1 --judge --agent-provider openai_compatible   # Nemotron agent, Claude judge
    python scripts/evaluate_caption.py --scenario photo --k 1              # one scenario, no judge

Calls the LLM (costs money). Product photos are fetched from the catalog CDN.
Output: ``results/evaluation/caption-<run_id>/`` with ``config.json``,
``records.jsonl`` (one trial per line: questions, answers, proposal, checks,
judge), ``summary.json`` and ``summary.md``. Scenarios and checks:
``src/sellerops/evaluation/caption_eval.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sellerops.evaluation.caption_eval import (  # noqa: E402
    SCENARIOS,
    CaptionJudge,
    JudgeError,
    run_trial,
)
from sellerops.evaluation.judge import nvidia_judge_model  # noqa: E402
from sellerops.evaluation.metrics import pass_hat_metrics, pass_metrics  # noqa: E402
from sellerops.harness.models import get_chat_model, supports_vision  # noqa: E402
from sellerops.seller import SellerService  # noqa: E402
from sellerops.seller.images import LocalImageStore, ProductImages  # noqa: E402
from sellerops.settings import get_settings  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--k", type=int, default=1, help="trials per scenario")
    parser.add_argument("--scenario", action="append", help="substring of a scenario id (repeatable)")
    parser.add_argument("--agent-provider", choices=["anthropic", "openai_compatible"], default=None)
    parser.add_argument("--agent-model", default=None)
    parser.add_argument("--judge", action="store_true",
                        help="LLM judge on code passes: NVIDIA for a Claude agent, Claude for any other agent")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> int:
    args = parse_args()
    settings = get_settings()
    agent_settings = settings
    if args.agent_provider or args.agent_model:
        agent_settings = replace(settings, llm_provider=args.agent_provider or settings.llm_provider,
                                 llm_model=args.agent_model or settings.llm_model, llm_fallback_models=())
        if agent_settings.llm_provider == "openai_compatible" and not args.agent_model:
            agent_settings = replace(agent_settings, llm_model=os.environ.get("LLM_MODEL", "").strip())
    if agent_settings.llm_provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set")
    model = get_chat_model(agent_settings).with_config(tags=["caption-eval"])
    scenarios = [s for s in SCENARIOS if not args.scenario or any(f in s.id for f in args.scenario)]
    if not scenarios:
        sys.exit("no scenario matches")

    judge = None
    if args.judge:   # another model family than the agent
        if agent_settings.llm_provider == "anthropic":
            judge = CaptionJudge(nvidia_judge_model(), os.environ.get("SELLEROPS_JUDGE_MODEL")
                                 or os.environ.get("LLM_MODEL", "nvidia"))
        else:
            claude = replace(settings, llm_provider="anthropic", llm_model=settings.llm_model
                             if settings.llm_provider == "anthropic" else "claude-haiku-5-5")
            judge = CaptionJudge(get_chat_model(claude, max_tokens=2000), claude.llm_model)

    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_id = f"caption-{stamp}-{agent_settings.llm_model.replace('/', '_')}-{uuid.uuid4().hex[:4]}"
    out = args.out or settings.results_dir / "evaluation" / run_id
    out.mkdir(parents=True, exist_ok=True)
    config = {
        "run_id": run_id, "eval": "caption-workflow", "k": args.k, "scenarios": [s.id for s in scenarios],
        "agent_provider": agent_settings.llm_provider, "agent_model": agent_settings.llm_model,
        "agent_effort": agent_settings.llm_effort if agent_settings.llm_provider == "anthropic" else None,
        "agent_vision": supports_vision(model),
        "judge": judge.metadata if judge else None, "git_commit": git_commit(), "started_at": stamp,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2))
    print(f"[caption-eval] {run_id}: {len(scenarios)} scenarios x {args.k}, agent {agent_settings.llm_model}, "
          f"vision {config['agent_vision']}, judge {config['judge']['model'] if judge else None}", flush=True)

    with tempfile.TemporaryDirectory(prefix="caption-eval-") as tmp:
        work = Path(tmp)
        image_store = LocalImageStore(work / "images", store_id="kala-studio")

        def one(job):
            scenario, trial = job
            record = run_trial(scenario, trial, model=model, workdir=work, catalog_path=settings.catalog_path,
                               sim_dir=settings.data_dir / "seller_sim",
                               images_factory=lambda: ProductImages(image_store))
            if judge and record.code_pass:
                service = SellerService(work / f"{scenario.id}-{trial}.sqlite")
                facts = {k: v["value"] for k, v in service.product_facts(scenario.product_id).items()}
                product = service.analytics().product(scenario.product_id)
                try:
                    record.judge = judge.assess(scenario, record, product, facts)
                except JudgeError as error:
                    record.judge = {"status": "error", "pass": None, "error": str(error)}
            # a judge error is no verdict (None), never a failure of the agent
            record.passed = record.judge.get("pass") if record.judge else record.code_pass
            mark = "PASS" if record.passed else ("ERR " if record.passed is None else "fail")
            failed = [k for k, v in record.checks.items() if not v]
            print(f"  {mark} {scenario.id}#{trial} asked={len(record.asked)} {record.seconds}s "
                  f"{failed or ''} {record.error or ''} "
                  f"{'' if not record.judge else 'judge=' + str(record.judge.get('pass'))}", flush=True)
            return record

        jobs = [(s, t) for s in scenarios for t in range(args.k)]
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            records = list(pool.map(one, jobs))

    with (out / "records.jsonl").open("w") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
    summary = summarise(records, scenarios, args.k, config)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    (out / "summary.md").write_text(markdown(summary))
    print(markdown(summary))
    print(f"[caption-eval] saved to {out}")
    return 0


def summarise(records, scenarios, k, config) -> dict:
    ids = [s.id for s in scenarios]

    def samples(attr):
        return [(r.scenario, None if getattr(r, attr) is None else int(bool(getattr(r, attr)))) for r in records]

    per = {}
    for s in scenarios:
        rs = [r for r in records if r.scenario == s.id]
        per[s.id] = {"passed": sum(bool(r.passed) for r in rs), "code_passed": sum(r.code_pass for r in rs),
                     "trials": len(rs), "failed_checks": sorted({c for r in rs for c, v in r.checks.items() if not v}),
                     "judge_fail": [r.judge.get("reason") for r in rs if r.judge and r.judge.get("pass") is False],
                     "asked": [len(r.asked) for r in rs]}
    tokens_in = [sum(u.get("input_tokens", 0) for u in r.usage.values()) for r in records]
    tokens_out = [sum(u.get("output_tokens", 0) for u in r.usage.values()) for r in records]
    judged = [r.judge for r in records if r.judge]
    return {
        "config": config,
        "overall": {**pass_metrics(samples("passed"), repetitions=k, k=k, task_ids=ids),
                    **pass_hat_metrics(samples("passed"), repetitions=k, k=k, task_ids=ids)},
        "code_only": {**pass_metrics(samples("code_pass"), repetitions=k, k=k, task_ids=ids),
                      **pass_hat_metrics(samples("code_pass"), repetitions=k, k=k, task_ids=ids)},
        "judge": {"scored": sum(j.get("status") == "scored" for j in judged),
                  "passed": sum(j.get("pass") is True for j in judged),
                  "errors": sum(j.get("status") == "error" for j in judged)},
        "crashed": sum(r.error is not None for r in records),
        "ask_accuracy": f"{sum('ask_behaviour' in r.checks and r.checks['ask_behaviour'] for r in records)}/"
                        f"{sum('ask_behaviour' in r.checks for r in records)}",
        "median_seconds": median(r.seconds for r in records),
        "avg_tokens": {"input": round(sum(tokens_in) / len(records)), "output": round(sum(tokens_out) / len(records))},
        "scenarios": per,
    }


def markdown(summary: dict) -> str:
    c = summary["config"]
    lines = [f"# Caption workflow eval {c['run_id']}", "",
             f"agent `{c['agent_model']}` (vision {c['agent_vision']}), k={c['k']}, "
             f"judge `{c['judge']['model'] if c['judge'] else None}`, commit `{c['git_commit']}`", "",
             "| metric | " + " | ".join(summary["overall"]) + " |",
             "|---|" + "---|" * len(summary["overall"]),
             "| **code + judge** | " + " | ".join(f"{v:.2f}" if v is not None else "n/a"
                                                for v in summary["overall"].values()) + " |",
             "| code checks only | " + " | ".join(f"{v:.2f}" if v is not None else "n/a"
                                                 for v in summary["code_only"].values()) + " |", "",
             "| scenario | pass | code | asked | failed checks | judge fails |", "|---|---|---|---|---|---|"]
    for sid, s in summary["scenarios"].items():
        lines.append(f"| {sid} | {s['passed']}/{s['trials']} | {s['code_passed']}/{s['trials']} | {s['asked']} | "
                     f"{', '.join(s['failed_checks'])} | {' / '.join(r[:120] for r in s['judge_fail'])} |")
    j = summary["judge"]
    lines += ["", f"- ask behaviour correct: {summary['ask_accuracy']}",
              f"- judge: {j['passed']}/{j['scored']} pass, {j['errors']} errors (only on code passes)",
              f"- crashed: {summary['crashed']}, median {summary['median_seconds']}s per trial, "
              f"avg tokens in {summary['avg_tokens']['input']} / out {summary['avg_tokens']['output']}"]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
