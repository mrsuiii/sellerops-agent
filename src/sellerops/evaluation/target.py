"""One task -> one isolated episode. Usable with or without LangSmith's runner."""

import math
from collections import Counter
from copy import deepcopy
from hashlib import sha256
from time import perf_counter

import torch
from langchain_core.messages import HumanMessage, message_to_dict
from langsmith.run_helpers import tracing_context

from ..agent.config import MODEL_CONFIGS
from ..agent.graph import build_graph
from ..schemas import MAX_AGENT_RETRIES
from ..services.episode_service.episode_service import EpisodeService
from .telemetry import capture_attempts


def make_target(*, action_model="sellerops-sft", router_model="qwen-base",
                max_model_calls=8, max_new_tokens=256, max_retries=MAX_AGENT_RETRIES,
                temperature=0.0, seed=42, trace_graph=False):
    """Configure the agent once; each call creates a fresh service and graph."""
    for model in (action_model, router_model):
        if model not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model alias: {model}")
    for value in (max_model_calls, max_new_tokens):
        if type(value) is not int or value < 1:
            raise ValueError("Model budgets must be positive integers")
    if type(max_retries) is not int or max_retries < 0:
        raise ValueError("max_retries must be a non-negative integer")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if type(temperature) not in (int, float) or not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be a finite non-negative number")
    config = {"configurable": {
        "action_model_name": action_model, "router_model_name": router_model,
        "max_model_calls": max_model_calls, "max_new_tokens": max_new_tokens,
        "max_retries": max_retries, "temperature": temperature,
    }, "recursion_limit": 4 * max_model_calls + 10}
    trials = Counter()

    def run_episode(inputs: dict) -> dict:
        # Only the prompt enters model history. Reference context/oracle answers
        # belong to the evaluator and are deliberately absent from this interface.
        if set(inputs) != {"task_id", "prompt"} or any(
            not isinstance(inputs[k], str) or not inputs[k].strip() for k in inputs
        ):
            raise ValueError("Target inputs must contain only task_id and prompt")
        trial_index = trials[inputs["task_id"]]
        trials[inputs["task_id"]] += 1
        # Stable per-task, per-trial streams, independent of task iteration order.
        trial_seed = int.from_bytes(sha256(
            f"{seed}:{inputs['task_id']}:{trial_index}".encode()
        ).digest()[:4], "big")
        trial_config = deepcopy(config)
        trial_config["configurable"]["generation_seed"] = trial_seed
        service = EpisodeService()
        episode = service.create_episode(inputs["task_id"])
        state = {"messages": [HumanMessage(content=inputs["prompt"])],
                 "session_id": episode.session_id, "model_calls": 0,
                 "retry_count": 0, "status": "running", "response": "",
                 "final_action": None}
        error = None
        started = perf_counter()
        # LangChain callbacks in the pinned version treat SDK mode "local" as
        # enabled and would post child traces. Explicit False keeps offline runs
        # local; the outer SDK evaluation still records the returned episode.
        # The current ModelAdapter runs on CPU. Restore its RNG after each trial.
        # Nodes derive distinct attempt seeds from this trial seed + call index.
        with (capture_attempts() as attempts, tracing_context(enabled=trace_graph),
              torch.random.fork_rng(devices=[])):
            torch.random.default_generator.manual_seed(trial_seed)
            try:
                # Values streaming preserves the last committed state if a later
                # node raises. Observations are also retained in EpisodeService.
                for value in build_graph(service).stream(
                    state, trial_config, stream_mode="values"
                ):
                    state = value
            except Exception as exc:
                error = {"type": type(exc).__name__, "message": str(exc)}
                state = {**state, "status": "failed"}
        return {
            "task_id": inputs["task_id"], "status": state["status"],
            "trial_index": trial_index, "generation_seed": trial_seed,
            "response": state.get("response", ""),
            "final_action": state.get("final_action"),
            "snapshot": service.snapshot(episode.session_id),
            "messages": [message_to_dict(m) for m in state["messages"]],
            "attempts": attempts, "model_calls": len(attempts),
            "committed_model_calls": state["model_calls"],
            "tool_calls": episode.step_count, "retry_count": state["retry_count"],
            "latency_seconds": perf_counter() - started, "error": error or state.get("error"),
            "failure_reason": "runtime_error" if error else state.get("failure_reason"),
            "config": trial_config, "answer_groundedness": "not_evaluated",
        }

    return run_episode
