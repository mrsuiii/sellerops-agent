"""Per-episode observations; collection never changes agent decisions."""

from contextlib import contextmanager
from contextvars import ContextVar
from time import perf_counter

from langsmith import get_current_run_tree

from ..schemas import SchemaError

_attempts: ContextVar[list | None] = ContextVar("sellerops_attempts", default=None)
_current: ContextVar[dict | None] = ContextVar("sellerops_attempt", default=None)


@contextmanager
def capture_attempts():
    """Give each episode an independent collector, including failed parsing."""
    events = []
    token = _attempts.set(events)
    try:
        yield events
    finally:
        _attempts.reset(token)


@contextmanager
def model_attempt(node: str, model: str):
    event = {"node": node, "model": model, "raw_output": None,
             "schema_valid": None, "error": None, "usage": None}
    collector = _attempts.get()
    if collector is not None:
        collector.append(event)
    token = _current.set(event)
    started = perf_counter()
    try:
        yield event
        event["schema_valid"] = True
    except Exception as error:
        if isinstance(error, SchemaError):
            event["schema_valid"] = False
        event["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        event["latency_seconds"] = perf_counter() - started
        _current.reset(token)


def note_attempt(**fields) -> None:
    """Annotate the current attempt (for example, a normalized output)."""
    event = _current.get()
    if event is not None:
        event.update(fields)


def record_generation(source: str, revision: str | None, usage: dict) -> None:
    """Record actual tokenizer counts, both locally and on the LLM trace."""
    event = _current.get()
    if event is not None:
        event.update(source=source, revision=revision, usage=usage)
    run = get_current_run_tree()
    if run is not None and run.run_type == "llm":
        run.add_metadata({"ls_provider": "huggingface", "ls_model_name": source,
                          "model_revision": revision})
        run.set(usage_metadata=usage)


def llm_trace_inputs(inputs: dict) -> dict:
    # Log messages, not RunnableConfig (which can contain callbacks or secrets).
    return {"messages": inputs["messages"]}


def llm_trace_outputs(output: str) -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": output}}]}
