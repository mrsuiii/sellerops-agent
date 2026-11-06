"""Adapt the existing deterministic grader to LangSmith feedback."""

import json

from .verifier import verify_outcome


def grade_episode(task: dict, outputs: dict) -> dict:
    """Score operational outcomes; exceptions in the grader remain eval errors."""
    if task["task_id"] != outputs["task_id"]:
        raise ValueError("Evaluation task and episode task_id differ")
    if outputs["status"] == "failed":
        reason = outputs.get("failure_reason") or ("runtime_error" if outputs["error"] else "budget_exhausted")
        return {"decision": "fail", "reason_code": reason, "details": outputs["error"] or {}}
    candidate = outputs["final_action"]
    if candidate is None:
        return {"decision": "fail", "reason_code": "missing_structured_outcome", "details": {}}
    required_status = "waiting_user" if candidate["type"] == "ask_user" else "completed"
    if outputs["status"] != required_status:
        return {"decision": "fail", "reason_code": "terminal_status_mismatch", "details": {}}
    return verify_outcome(task, outputs["snapshot"], candidate)


def episode_evaluator(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """LangSmith passes reference outputs only to the evaluator, not the model."""
    task = reference_outputs["task"]
    if inputs["task_id"] != task["task_id"]:
        raise ValueError("Dataset inputs and reference task_id differ")
    report = grade_episode(task, outputs)
    feedback = [
        {"key": "structured_outcome_pass", "score": int(report["decision"] == "pass"),
         "comment": json.dumps(report, ensure_ascii=False)},
        {"key": "runtime_failure", "score": int(outputs["status"] == "failed")},
        {"key": "blocked_policy_attempt", "score": int(any(
            (event["result"].get("error") or {}).get("code") == "discount_policy_violation"
            for event in outputs["snapshot"]["history"]
        ))},
        {"key": "model_calls", "score": outputs["model_calls"]},
        {"key": "retry_count", "score": outputs["retry_count"]},
        {"key": "tool_calls", "score": outputs["tool_calls"]},
        {"key": "episode_latency_seconds", "score": outputs["latency_seconds"]},
    ]
    for node in ("router", "action"):
        checked = [a["schema_valid"] for a in outputs["attempts"]
                   if a["node"] == node and a["schema_valid"] is not None]
        if checked:
            feedback.append({"key": f"{node}_schema_validity", "score": sum(checked) / len(checked)})
    if "judge" in outputs:
        from .judge import end_to_end_score
        judge = outputs["judge"]
        feedback.extend([
            {"key": "answer_pass", "score": int(judge["pass"]) if judge["status"] == "scored" else None,
             "comment": json.dumps(judge, ensure_ascii=False)},
            {"key": "judge_error", "score": int(judge["status"] == "error")},
            {"key": "end_to_end_pass", "score": end_to_end_score(report["decision"] == "pass", outputs)},
        ])
    return {"results": feedback}
