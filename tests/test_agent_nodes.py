"""Exercise generation boundaries and real graph config injection, without weights."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from sellerops.agent import nodes
from sellerops.agent.state import AgentState
from sellerops.dataset import SYSTEM_PROMPT


@pytest.fixture
def generation(monkeypatch):
    tokenizer = Mock()
    tokenizer.apply_chat_template.return_value = {
        "input_ids": torch.tensor([[10, 11]]),
        "attention_mask": torch.tensor([[1, 1]]),
    }
    tokenizer.decode.return_value = (
        '{"type":"tool_call","tool":"get_inventory","arguments":{"product_id":"P1001"}}'
    )
    model = Mock(device=torch.device("cpu"))
    model.generate.return_value = torch.tensor([[10, 11, 12, 13]])
    loader = Mock(return_value=SimpleNamespace(tokenizer=tokenizer, model=model))
    monkeypatch.setattr(nodes, "_load_model", loader)
    monkeypatch.setattr(nodes, "MODEL_CONFIGS", {"demo/model": {"local": False, "model_name": "demo/model"}})
    return loader, tokenizer, model


def initial_state():
    return {"messages": [HumanMessage(content="Cek stok P1001")],
            "model_calls": 0, "retry_count": 0, "session_id": "demo"}


def test_graph_injects_config_and_appends_only_new_message(generation):
    loader, tokenizer, model = generation
    builder = StateGraph(AgentState)
    builder.add_node("call_model", nodes.call_model)
    builder.add_edge(START, "call_model")
    builder.add_edge("call_model", END)
    state = initial_state()
    result = builder.compile().invoke(
        state, config={"configurable": {"model_name": "demo/model", "max_new_tokens": 64}}
    )
    loader.assert_called_once_with(local=False, model_name="demo/model")
    assert len(result["messages"]) == 2
    assert result["messages"][-1].tool_calls[0]["name"] == "get_inventory"
    assert result["model_calls"] == 1
    assert len(state["messages"]) == 1
    assert tokenizer.apply_chat_template.call_args.args[0] == [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Cek stok P1001"},
    ]
    assert tokenizer.decode.call_args.args[0].tolist() == [12, 13]
    assert model.generate.call_args.kwargs["max_new_tokens"] == 64
    assert model.generate.call_args.kwargs["do_sample"] is False


def test_local_folder_and_existing_system_prompt(generation, tmp_path):
    loader, tokenizer, _ = generation
    state = initial_state()
    state["messages"].insert(0, SystemMessage(content="Existing instructions"))
    source = str(tmp_path.resolve())
    nodes.MODEL_CONFIGS["local-sft"] = {"local": True, "model_path": source, "tokenizer_path": source}
    nodes.call_model(state, {"configurable": {"model_name": "local-sft"}})
    loader.assert_called_once_with(local=True, model_path=source, tokenizer_path=source)
    messages = tokenizer.apply_chat_template.call_args.args[0]
    assert len(messages) == 2
    assert messages[0]["content"] == "Existing instructions"


@pytest.mark.parametrize("options", [
    {}, {"model_name": " "}, {"model_name": 3}, {"model_name": "unknown"},
    {"model_name": "demo/model", "max_new_tokens": 0},
    {"model_name": "demo/model", "max_new_tokens": True},
])
def test_invalid_config_fails_before_loading(generation, options):
    loader, _, _ = generation
    with pytest.raises(ValueError):
        nodes.call_model(initial_state(), {"configurable": options})
    loader.assert_not_called()


def test_invalid_model_output_schedules_retry_without_mutating_input(generation):
    _, tokenizer, _ = generation
    tokenizer.decode.return_value = "not an action"
    state = initial_state()
    result = nodes.call_model(state, {"configurable": {"model_name": "demo/model"}})
    assert result["model_calls"] == 1 and result["retry_count"] == 0
    assert result["pending_retry"] == "action" and result["messages"] == []
    assert state["model_calls"] == 0
    assert len(state["messages"]) == 1


def test_temperature_enables_sampling(generation):
    _, _, model = generation
    nodes.call_model(initial_state(), {"configurable": {"model_name": "demo/model", "temperature": .7}})
    options = model.generate.call_args.kwargs
    assert options["do_sample"] is True and options["temperature"] == .7
    assert options["top_p"] == 1.0 and options["top_k"] == 0


def test_sampling_seed_survives_cold_loader_and_changes_per_attempt(generation):
    loader, _, model = generation
    draws = []
    adapter = loader.return_value

    def load(**kwargs):
        torch.rand(20)  # model initialization can consume random numbers
        return adapter

    def generate(**kwargs):
        draws.append(torch.rand(3).tolist())
        return torch.tensor([[10, 11, 12, 13]])

    loader.side_effect = load
    model.generate.side_effect = generate
    config = {"configurable": {"model_name": "demo/model", "temperature": .7, "generation_seed": 42}}
    nodes.call_model(initial_state(), config)
    nodes.call_model(initial_state(), config)
    nodes.call_model({**initial_state(), "model_calls": 1}, config)
    assert draws[0] == draws[1] and draws[0] != draws[2]
