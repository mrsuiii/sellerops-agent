#!/usr/bin/env python
"""Run the SellerOps dev evaluation through LangSmith SDK."""

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sellerops.agent.config import MODEL_CONFIGS
from sellerops.schemas import MAX_AGENT_RETRIES
from sellerops.evaluation.runner import run_evaluation
from sellerops.evaluation.judge import DEFAULT_MODEL


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--offline", action="store_true", help="SDK evaluation without cloud uploads")
    mode.add_argument("--cloud", action="store_true", help="Publish dev dataset, traces and scores to LangSmith")
    parser.add_argument("--limit", type=int, help="Evaluate the first N dev tasks (cloud dataset contains all dev tasks)")
    parser.add_argument("--action-model", choices=sorted(MODEL_CONFIGS), default="sellerops-sft")
    parser.add_argument("--router-model", choices=sorted(MODEL_CONFIGS), default="qwen-base")
    parser.add_argument("--max-model-calls", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-retries", type=int, default=MAX_AGENT_RETRIES,
                        help="Shared schema-repair retry budget per episode")
    parser.add_argument("--repetitions", type=int, default=1, help="Fresh episodes per task")
    parser.add_argument("--pass-k", type=int, default=1, help="Report pass@1 and pass@K; K <= repetitions")
    parser.add_argument("--temperature", type=float, default=0.0, help="0 = greedy; >0 = sampling")
    parser.add_argument("--seed", type=int, default=42, help="Base seed for distinct per-task/trial CPU RNG streams")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--judge", choices=["none", "gemini"], default="none",
                        help="Optional external prose judge; gemini sends data even with --offline")
    parser.add_argument("--judge-model", default=DEFAULT_MODEL)
    args = parser.parse_args()
    try:
        result = run_evaluation(cloud=args.cloud, limit=args.limit, action_model=args.action_model,
                                router_model=args.router_model, max_model_calls=args.max_model_calls,
                                max_new_tokens=args.max_new_tokens, max_retries=args.max_retries,
                                repetitions=args.repetitions, pass_k=args.pass_k,
                                temperature=args.temperature, seed=args.seed, output_root=args.output_root,
                                judge=args.judge, judge_model=args.judge_model)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    # Agent failures are legitimate experiment results. Harness errors need attention.
    return 2 if (result["summary"]["evaluation_errors"] or result["summary"].get("judge_errors")
                 or not result["summary"]["complete"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
