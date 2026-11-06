from langchain.messages import AnyMessage
from typing_extensions import TypedDict, Annotated, NotRequired, Literal
import operator


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    session_id: str
    model_calls: int
    retry_count: int
    # These keys are produced by nodes, so the initial input can omit them.
    route: NotRequired[Literal["act", "reply", "ask"]]
    status: NotRequired[Literal["running", "completed", "waiting_user", "failed"]]
    response: NotRequired[str]
    final_action: NotRequired[dict | None]
    pending_retry: NotRequired[Literal["router", "action"] | None]
    retry_feedback: NotRequired[str | None]
    failure_reason: NotRequired[str | None]
    error: NotRequired[dict | None]
