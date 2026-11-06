"""LangSmith evaluation loop plus a commented ordinary-Python comparison."""

from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from statistics import mean
from uuid import uuid4

from langsmith import Client, evaluate
from langsmith.run_helpers import tracing_context

from ..agent.config import MODEL_CONFIGS
from ..schemas import MAX_AGENT_RETRIES
from .dataset import PROJECT_ROOT, as_examples, load_dev_records, prepare_cloud_dataset
from .evaluators import episode_evaluator
from .target import make_target
from .metrics import make_pass_summary, pass_metrics, validate_repetitions
from .judge import DEFAULT_MODEL, GeminiJudge, with_judge


def _percentile(values: list[float], quantile: float):
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * quantile
    low = int(index)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def summarize(records: list[dict], *, repetitions=1, pass_k=1, task_ids=None, include_judge=False) -> dict:
    """Keep evaluator failures visible rather than converting them to agent fails."""
    scored = [r for r in records if r["evaluation_error"] is None]
    passed = sum(r["scores"]["structured_outcome_pass"] for r in scored)
    outputs = [r["output"] for r in records if r["output"]]
    summary = {
        "episodes": len(records), "scored_episodes": len(scored),
        "evaluation_errors": len(records) - len(scored), "passed": passed,
        "structured_outcome_pass_rate": passed / len(records) if records and len(scored) == len(records) else None,
        "runtime_failures": sum(o["status"] == "failed" for o in outputs),
        "reason_counts": dict(Counter(r["reason_code"] for r in records)),
        "mean_model_calls": mean(o["model_calls"] for o in outputs) if outputs else None,
        "mean_tool_calls": mean(o["tool_calls"] for o in outputs) if outputs else None,
        "mean_retry_count": mean(o["retry_count"] for o in outputs) if outputs else None,
        "latency_p50_seconds": _percentile([o["latency_seconds"] for o in outputs], .5),
        "latency_p95_seconds": _percentile([o["latency_seconds"] for o in outputs], .95),
        "answer_groundedness": "not_evaluated",
    }
    summary.update(pass_metrics(
        [(r["task_id"], None if r["evaluation_error"] else r["scores"].get("structured_outcome_pass"))
         for r in records], repetitions=repetitions, k=pass_k,
        task_ids=task_ids if task_ids is not None else {r["task_id"] for r in records},
    ))
    if include_judge:
        summary["answer_groundedness"] = "judge_enabled"
        summary["judge_errors"] = sum(o.get("judge", {}).get("status") == "error" for o in outputs)
        summary["judged_episodes"] = sum(o.get("judge", {}).get("status") == "scored" for o in outputs)
        summary["end_to_end"] = pass_metrics(
            [(r["task_id"], None if r["evaluation_error"] else r["scores"].get("end_to_end_pass")) for r in records],
            repetitions=repetitions, k=pass_k,
            task_ids=task_ids if task_ids is not None else {r["task_id"] for r in records})
    for node in ("router", "action"):
        checked = [a["schema_valid"] for o in outputs for a in o["attempts"]
                   if a["node"] == node and a["schema_valid"] is not None]
        summary[f"{node}_schema_validity"] = sum(checked) / len(checked) if checked else None
    summary["by_family"] = {
        family: {"episodes": len(group), "passed": sum(
            r["scores"].get("structured_outcome_pass") == 1 and not r["evaluation_error"] for r in group
        ), "evaluation_errors": sum(r["evaluation_error"] is not None for r in group)}
        for family in sorted({r["family"] for r in records})
        for group in [[r for r in records if r["family"] == family]]
    }
    return summary


def _export_row(row) -> dict:
    run, example = row["run"], row["example"]
    feedback = row["evaluation_results"]["results"]
    scores = {f.key: f.score for f in feedback}
    failure = run.error or next((f.comment for f in feedback if (f.extra or {}).get("error")), None)
    outcome = next((f for f in feedback if f.key == "structured_outcome_pass"), None)
    if outcome is None or outcome.score is None:
        failure = failure or "Evaluator did not produce structured_outcome_pass"
    report = json.loads(outcome.comment) if outcome and outcome.score is not None else None
    return {"task_id": example.inputs["task_id"], "family": example.metadata["family"],
            "example_id": str(example.id), "run_id": str(run.id),
            "output": run.outputs, "scores": scores, "verifier_report": report,
            "reference_outputs": example.outputs,
            "reason_code": report["reason_code"] if report else "evaluation_error",
            "evaluation_error": failure, "task_metadata": example.metadata}


def run_evaluation(*, cloud=False, limit=None, action_model="sellerops-sft",
                   router_model="qwen-base", max_model_calls=8, max_new_tokens=256,
                   max_retries=MAX_AGENT_RETRIES, repetitions=1, pass_k=1,
                   temperature=0.0, seed=42,
                   judge="none", judge_model=DEFAULT_MODEL,
                   output_root: Path | None = None) -> dict:
    """Run dev episodes sequentially; cloud publishing is an explicit choice."""
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("limit must be a positive integer")
    validate_repetitions(repetitions, pass_k)
    if judge not in {"none", "gemini"}:
        raise ValueError("judge must be none or gemini")
    judge_client = GeminiJudge(judge_model) if judge == "gemini" else None
    target = make_target(action_model=action_model, router_model=router_model,
                         max_model_calls=max_model_calls, max_new_tokens=max_new_tokens,
                         max_retries=max_retries, temperature=temperature, seed=seed,
                         trace_graph=cloud)
    if cloud and not (os.environ.get("LANGSMITH_API_KEY") or os.environ.get("LANGCHAIN_API_KEY")):
        raise ValueError("Set LANGSMITH_API_KEY in your terminal for --cloud, or use --offline")
    examples = as_examples(load_dev_records())
    if judge_client:
        target = with_judge(target, judge_client, {e.inputs["task_id"]: e.outputs for e in examples})
    # Offline has no server calls even when cloud tracing is enabled in the shell.
    client = Client() if cloud else Client(api_url="http://localhost:1984", api_key="offline",
                                          auto_batch_tracing=False, info={})
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    destination = (output_root or PROJECT_ROOT / "results/evaluation") / f"{run_id}-{action_model}"
    records = []
    try:
        if cloud:
            examples = prepare_cloud_dataset(client, examples)
        selected = examples[:limit] if limit else examples
        task_ids = [e.inputs["task_id"] for e in selected]
        destination.mkdir(parents=True, exist_ok=False)
        metadata = {
            "action_model": action_model, "router_model": router_model,
            "model_sources": {name: MODEL_CONFIGS[name] for name in {action_model, router_model}},
            "source_sha256": selected[0].metadata["source_sha256"],
            **{key: value for key, value in selected[0].metadata.items()
               if key.startswith("answer_key_") or key in {"task_source_sha256", "environment_sha256"}},
            "dataset_version": selected[0].metadata["dataset_version"],
            "environment_version": selected[0].metadata["environment_version"],
            "max_model_calls": max_model_calls, "max_new_tokens": max_new_tokens,
            "max_retries": max_retries, "repetitions": repetitions, "pass_k": pass_k,
            "temperature": temperature, "generation_base_seed": seed,
            "split": "dev", "selected_task_ids": task_ids,
            "cloud": cloud, "do_sample": temperature > 0, "max_concurrency": 0,
            "judge": judge_client.metadata if judge_client else {"provider": "none"},
        }
        (destination / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        # Ini pengganti loop `for task in tasks` pada contoh di bawah.
        # evaluate() memanggil target, lalu evaluator, untuk setiap example.
        with tracing_context(enabled=True if cloud else "local", client=client):
            results = evaluate(
                target, data=selected, evaluators=[episode_evaluator], client=client,
                num_repetitions=repetitions,
                summary_evaluators=[make_pass_summary(repetitions=repetitions, k=pass_k, task_ids=task_ids,
                                                      include_judge=judge_client is not None)],
                experiment_prefix=f"sellerops-{action_model}", metadata=metadata,
                max_concurrency=0, blocking=False, upload_results=cloud,
            )
            with (destination / "episodes.jsonl").open("x", encoding="utf-8") as stream:
                for row in results:
                    record = _export_row(row)
                    records.append(record)
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stream.flush()
            results.wait()
        summary = summarize(records, repetitions=repetitions, pass_k=pass_k, task_ids=task_ids,
                            include_judge=judge_client is not None)
        summary.update(experiment_name=results.experiment_name,
                       expected_episodes=len(selected) * repetitions,
                       repetitions=repetitions, pass_k=pass_k,
                       complete=Counter(r["task_id"] for r in records) ==
                       Counter({task_id: repetitions for task_id in task_ids}))
        (destination / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        if cloud:
            client.flush()
        return {"output_dir": str(destination), "summary": summary}
    finally:
        client.close()


# PERBANDINGAN BELAJAR: loop biasa tanpa runner/dashboard LangSmith.
# Ini komentar yang bisa dibaca berdampingan dengan evaluate() di atas.
# Fungsi episode dan aturan penilaian identik; yang berubah penggerak loopnya.
# Matikan tracing saat mencoba contoh ini agar tidak mengirim trace ke cloud.
#
# import os
# os.environ["LANGSMITH_TRACING"] = "false"
# from sellerops.evaluation.target import make_target
# from sellerops.evaluation.dataset import load_dev_records
# from sellerops.evaluation.evaluators import grade_episode
#
# target = make_target(action_model="sellerops-sft", temperature=0.7, seed=42)
# repetitions, k = 3, 3
# local_results = []
# for example in load_dev_records():
#     for trial in range(repetitions):        # LangSmith: num_repetitions
#         output = target(example["inputs"])  # episode baru, retry di dalam graph
#         try:
#             grade = grade_episode(example["outputs"]["task"], output)
#             score = int(grade["decision"] == "pass")
#         except Exception:
#             score = None                   # error grader, bukan agent fail
#         local_results.append((example["inputs"]["task_id"], score))
# from sellerops.evaluation.metrics import pass_metrics
# metrics = pass_metrics(local_results, repetitions=repetitions, k=k,
#                        task_ids=[e["inputs"]["task_id"] for e in load_dev_records()])
#
# # Di sini kita sendiri yang menyimpan JSONL, menghitung metrik, dan membuat
# # tampilan perbandingan eksperimen. LangSmith menyediakan pengelolaan itu.
