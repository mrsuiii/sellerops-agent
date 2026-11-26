"""Validate the message judge against human labels (A3: trust the judge only after this).

    python scripts/judge_labels.py export --out results/evaluation/judge_labels.jsonl   # latest run
    #   → open the file, fill "human_faithful" and "human_answers_request" with true/false
    python scripts/judge_labels.py score --labels results/evaluation/judge_labels.jsonl

``export`` writes one item per judged conversation from one or more runs: the
seller's conversation, the final message and data, and the tool evidence —
*without* the judge's verdict, so labelling is blind. Items already in the
output file keep their labels (re-export after a new run adds new items).

``score`` joins the labels with the judge verdicts and reports, per
criterion: n, raw agreement, TPR (judge says pass when the human says pass),
TNR (judge says fail when the human says fail) and Cohen's kappa. Fails are
rare, so look at TNR first: a judge that passes everything has high agreement
and TNR 0. Aim for ~30–50 labelled items with a few fails before relying on it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "evaluation"
CRITERIA = ("faithful", "answers_request")


def load_records(runs: list[Path]) -> list[tuple[str, dict[str, Any]]]:
    out = []
    for run in runs:
        for line in (run / "records.jsonl").open():
            record = json.loads(line)
            if (record.get("judge") or {}).get("status") == "scored":
                out.append((f"{run.name}/{record['task_id']}#t{record['trial']}", record))
    return out


def compact_evidence(record: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"tool": s["tool"], "arguments": s["arguments"], "result": s["observation"]}
            for s in record.get("history", [])]


def export(args: argparse.Namespace) -> None:
    runs = args.run or [max(p for p in RESULTS.glob("*") if (p / "records.jsonl").exists())]
    existing = {}
    if args.out.exists():
        existing = {item["key"]: item for item in map(json.loads, args.out.open())}
    added = 0
    for key, record in load_records(runs):
        if key in existing:
            continue
        final = record.get("final_action") or {}
        existing[key] = {"key": key, "task_id": record["task_id"], "conversation": record.get("transcript", []),
                         "final_message": final.get("message"), "final_data": final.get("data"),
                         "tool_evidence": compact_evidence(record),
                         "human_faithful": None, "human_answers_request": None, "note": ""}
        added += 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as handle:
        for item in existing.values():
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"{args.out}: {len(existing)} items ({added} new); fill human_faithful / human_answers_request")


def kappa(pairs: list[tuple[bool, bool]]) -> float | None:
    n = len(pairs)
    if n == 0:
        return None
    observed = sum(h == j for h, j in pairs) / n
    p_human, p_judge = sum(h for h, _ in pairs) / n, sum(j for _, j in pairs) / n
    expected = p_human * p_judge + (1 - p_human) * (1 - p_judge)
    return None if expected == 1 else round((observed - expected) / (1 - expected), 3)


def score(args: argparse.Namespace) -> None:
    labels = [json.loads(line) for line in args.labels.open()]
    runs = sorted({RESULTS / item["key"].split("/")[0] for item in labels})
    verdicts = {key: record["judge"] for key, record in load_records([r for r in runs if r.exists()])}
    for criterion in CRITERIA:
        pairs = [(item[f"human_{criterion}"], verdicts[item["key"]][criterion]) for item in labels
                 if isinstance(item.get(f"human_{criterion}"), bool) and item["key"] in verdicts]
        human_pass = [j for h, j in pairs if h]
        human_fail = [j for h, j in pairs if not h]
        tpr = round(sum(human_pass) / len(human_pass), 3) if human_pass else None
        tnr = round(sum(not j for j in human_fail) / len(human_fail), 3) if human_fail else None
        agreement = round(sum(h == j for h, j in pairs) / len(pairs), 3) if pairs else None
        print(f"{criterion:16} n={len(pairs):3}  human fails={len(human_fail):3}  agreement={agreement}  "
              f"TPR={tpr}  TNR={tnr}  kappa={kappa(pairs)}")
        for item in labels:
            human = item.get(f"human_{criterion}")
            if isinstance(human, bool) and item["key"] in verdicts and verdicts[item["key"]][criterion] != human:
                print(f"    disagree {item['key']}: human={human} judge={verdicts[item['key']][criterion]} "
                      f"— {verdicts[item['key']]['reason'][:160]}")
    if not any(isinstance(i.get("human_faithful"), bool) for i in labels):
        sys.exit("no labels yet: fill human_faithful / human_answers_request in the file")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p_export = sub.add_parser("export")
    p_export.add_argument("--run", type=Path, action="append", help="run directory (repeatable; default latest)")
    p_export.add_argument("--out", type=Path, default=RESULTS / "judge_labels.jsonl")
    p_score = sub.add_parser("score")
    p_score.add_argument("--labels", type=Path, default=RESULTS / "judge_labels.jsonl")
    args = parser.parse_args()
    (export if args.command == "export" else score)(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
