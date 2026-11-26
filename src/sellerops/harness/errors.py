"""Which model-call failures are worth retrying (Session A2).

Error map of the agent graph — every failure has one explicit behaviour:

| Failure                                        | Behaviour                     | Where                         |
|------------------------------------------------|-------------------------------|-------------------------------|
| 429 / 408 / 409 / 5xx / 529, timeout, network  | retry with backoff            | SDK ``max_retries`` (honours  |
|                                                |                               | ``retry-after``), then the    |
|                                                |                               | ``model`` node ``RetryPolicy``|
| provider still failing after its retries       | switch to a fallback model    | ``with_fallbacks`` in graph   |
| 400 / 401 / 403 / 404, bugs                    | fail fast (no retry)          | raised from ``graph.invoke``  |
| bad tool arguments, unknown tool, tool raised  | error result to the model     | ``tools`` node                |
| business error (unknown product, caption limit)| ``{"ok": false}`` observation | the tool's client             |
| loop / budget exhausted                        | end the turn with a reason    | ``stop_reason`` (max_steps,   |
|                                                |                               | token_budget, time_budget)    |

Retries multiply: SDK ``max_retries=2`` (3 tries) × node ``max_attempts=2`` = up
to 6 calls for one model step. The SDK layer handles short blips quickly; the
node layer waits longer (seconds) for overloads that outlast the SDK backoff.

Classification is by duck typing (``status_code`` / class name), so the
harness does not import ``anthropic`` or ``openai``: both SDKs expose
``status_code`` on ``APIStatusError`` and name their network errors
``APIConnectionError`` / ``APITimeoutError``.
"""

from __future__ import annotations

TRANSIENT_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504, 529})
_TRANSIENT_NAMES = ("ConnectionError", "TimeoutError", "Timeout", "RateLimitError", "OverloadedError",
                    "InternalServerError", "ServiceUnavailableError")


def is_transient_error(error: BaseException) -> bool:
    """True for rate limits, overloads, server errors, timeouts and network errors."""
    status = getattr(error, "status_code", None)
    if isinstance(status, int):
        return status in TRANSIENT_STATUS
    if isinstance(error, (TimeoutError, ConnectionError)):
        return True
    return type(error).__name__.endswith(_TRANSIENT_NAMES)
