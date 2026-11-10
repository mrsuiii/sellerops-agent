"""Regression checks for episode isolation, reset, and tool bookkeeping."""

import json
from copy import deepcopy
from typing import Any

import pytest

from sellerops.environment import SellerEnvironment
from sellerops.schemas import SchemaError
from sellerops.services.episode_service.episode_service import EpisodeService
from sellerops.services.episode_service.episode_state import EpisodeState


def inventory_action(product_id: str = "P1001") -> dict[str, Any]:
    """Build an inventory call for the supplied product ID.

    Args:
        product_id: Product to query.

    Returns:
        A fresh action dictionary accepted by the environment.
    """
    return {
        "type": "tool_call",
        "tool": "get_inventory",
        "arguments": {"product_id": product_id},
    }


def test_episodes_have_independent_data_and_history() -> None:
    """Changing one episode must not affect another episode of the same task."""
    service = EpisodeService()
    first = service.create_episode("task-a")
    second = service.create_episode("task-a")

    assert first.session_id != second.session_id
    first.environment.inventory["P1001"]["stock"] = 99
    first.environment.catalog["P1001"]["tags"].append("changed")
    first.environment.policies["P1001"]["max_discount_percent"] = 0
    service.execute_tool(first.session_id, inventory_action())

    assert second.environment.inventory["P1001"]["stock"] == 7
    assert "changed" not in second.environment.catalog["P1001"]["tags"]
    assert second.environment.policies["P1001"]["max_discount_percent"] == 10
    assert second.history == []
    assert second.step_count == 0


def test_tool_execution_preserves_history_when_caller_mutates_objects() -> None:
    """Tool calls work and their history survives edits to inputs and outputs."""
    service = EpisodeService()
    episode = service.create_episode("task-a")
    action = inventory_action()
    result = service.execute_tool(episode.session_id, action)
    recorded = deepcopy(episode.history)

    assert result["ok"] is True
    assert result["result"] == {"product_id": "P1001", "stock": 7}
    assert episode.step_count == 1
    assert recorded == [{"action": action, "result": result}]

    action["arguments"]["product_id"] = "P1002"
    result["result"]["stock"] = 999
    assert episode.history == recorded


def test_reset_restores_fixtures_and_allows_same_observation_again() -> None:
    """Reset restores data and execution state while preserving episode identity."""
    service = EpisodeService()
    episode = service.create_episode("task-a")
    episode.max_steps = 1
    initial_data = deepcopy(vars(episode.environment))
    first_result = service.execute_tool(episode.session_id, inventory_action())
    assert episode.terminal is True

    episode.environment.inventory["P1001"]["stock"] = 999
    episode.environment.catalog["P1001"]["tags"].append("changed")
    episode.environment.policies["P1001"]["max_discount_percent"] = 0
    service.reset(episode.session_id)

    assert service.episodes[episode.session_id] is episode
    assert episode.task_id == "task-a"
    assert episode.max_steps == 1
    assert vars(episode.environment) == initial_data
    assert episode.history == []
    assert episode.step_count == 0
    assert episode.terminal is False
    assert service.execute_tool(episode.session_id, inventory_action()) == first_result


@pytest.mark.parametrize("operation", ["reset", "execute_tool", "snapshot"])
def test_unknown_episode_is_reported(operation: str) -> None:
    """Missing IDs must produce a clear error without creating an episode.

    Args:
        operation: Service method under test.
    """
    service = EpisodeService()
    arguments = ("missing", inventory_action()) if operation == "execute_tool" else ("missing",)
    with pytest.raises(ValueError, match="No episode found"):
        getattr(service, operation)(*arguments)
    assert service.episodes == {}


@pytest.mark.parametrize("terminal,step_count,max_steps", [(True, 0, 8), (False, 8, 8), (False, 0, 0)])
def test_blocked_episode_never_dispatches_tool(
    monkeypatch: pytest.MonkeyPatch, terminal: bool, step_count: int, max_steps: int
) -> None:
    """Terminal or exhausted episodes must reject calls before reaching tools.

    Args:
        monkeypatch: Pytest fixture for observing whether dispatch occurs.
        terminal: Existing terminal flag.
        step_count: Existing execution count.
        max_steps: Configured limit, including a zero-call budget.
    """
    service = EpisodeService()
    episode = service.create_episode("task-a")
    episode.terminal = terminal
    episode.step_count = step_count
    episode.max_steps = max_steps
    calls = []
    monkeypatch.setattr(episode.environment, "execute", calls.append)

    with pytest.raises(ValueError, match="terminal|step limit"):
        service.execute_tool(episode.session_id, inventory_action())
    assert calls == []
    assert episode.history == []
    assert episode.step_count == step_count


def test_final_allowed_call_terminates_episode() -> None:
    """The last allowed call succeeds; any subsequent call is rejected."""
    service = EpisodeService()
    episode = service.create_episode("task-a")
    episode.max_steps = 1
    assert service.execute_tool(episode.session_id, inventory_action())["ok"] is True
    assert episode.terminal is True
    with pytest.raises(ValueError, match="terminal"):
        service.execute_tool(episode.session_id, inventory_action())
    assert episode.step_count == len(episode.history) == 1


def test_business_failure_is_recorded_and_counts_as_a_step() -> None:
    """A valid tool request with a missing product still consumes one step."""
    service = EpisodeService()
    episode = service.create_episode("task-a")
    result = service.execute_tool(episode.session_id, inventory_action("missing"))
    assert result["ok"] is False
    assert result["error"]["code"] == "product_not_found"
    assert episode.history[0]["result"] == result
    assert episode.step_count == 1


def test_invalid_action_preserves_schema_error_and_does_not_count() -> None:
    """Schema rejection remains distinguishable from a tool observation."""
    service = EpisodeService()
    episode = service.create_episode("task-a")
    with pytest.raises(SchemaError):
        service.execute_tool(episode.session_id, {"type": "tool_call"})
    assert episode.history == []
    assert episode.step_count == 0


def test_snapshot_round_trips_through_json_with_execution_data() -> None:
    """Exported data retains identity, history, and terminal status through JSON."""
    service = EpisodeService()
    episode = service.create_episode("task-a")
    episode.max_steps = 1
    action = inventory_action()
    result = service.execute_tool(episode.session_id, action)

    snapshot = service.snapshot(episode.session_id)
    restored = json.loads(json.dumps(snapshot, allow_nan=False))

    assert restored == snapshot
    assert restored["snapshot_version"] == 1
    assert restored["session_id"] == episode.session_id
    assert restored["task_id"] == "task-a"
    assert restored["history"] == [{"action": action, "result": result}]
    assert restored["step_count"] == restored["max_steps"] == 1
    assert restored["terminal"] is True
    assert restored["initial_environment"] == restored["environment"]


def test_snapshot_edits_do_not_change_episode_or_initial_data() -> None:
    """Nested edits to the export cannot mutate live data or the reset baseline."""
    service = EpisodeService()
    episode = service.create_episode("task-a")
    service.execute_tool(episode.session_id, inventory_action())
    before = service.snapshot(episode.session_id)
    edited = service.snapshot(episode.session_id)

    edited["initial_environment"]["inventory"]["P1001"]["stock"] = 999
    edited["environment"]["catalog"]["P1001"]["tags"].append("changed")
    edited["environment"]["policies"]["P1001"]["max_discount_percent"] = 0
    edited["history"][0]["action"]["arguments"]["product_id"] = "P1002"
    edited["history"][0]["result"]["result"]["stock"] = 999
    edited["step_count"] = 99

    assert service.snapshot(episode.session_id) == before
    service.reset(episode.session_id)
    assert episode.environment.inventory["P1001"]["stock"] == 7


def test_saved_snapshot_stays_unchanged_when_episode_progresses() -> None:
    """A prior export remains a point-in-time record after calls and reset."""
    service = EpisodeService()
    episode = service.create_episode("task-a")
    saved = service.snapshot(episode.session_id)
    original = deepcopy(saved)

    episode.environment.inventory["P1001"]["stock"] = 99
    service.execute_tool(episode.session_id, inventory_action())
    current = service.snapshot(episode.session_id)
    assert current["initial_environment"]["inventory"]["P1001"]["stock"] == 7
    assert current["environment"]["inventory"]["P1001"]["stock"] == 99
    assert saved == original

    service.reset(episode.session_id)
    assert saved == original
    assert current["history"][0]["result"]["result"]["stock"] == 99


def test_reset_uses_captured_initial_environment_not_global_defaults() -> None:
    """Custom starting data survives repeated resets and copies cannot alter it."""
    environment = SellerEnvironment()
    environment.inventory["P1001"]["stock"] = 42
    environment.catalog["P1001"]["tags"].append("custom")
    environment.policies["P1001"]["max_discount_percent"] = 3
    episode = EpisodeState("custom-session", "task-a", environment=environment)
    service = EpisodeService()
    service.episodes[episode.session_id] = episode
    expected = deepcopy(vars(environment))

    exposed_initial = episode.initial_environment
    exposed_initial["inventory"]["P1001"]["stock"] = -1
    for _ in range(2):
        episode.environment.inventory["P1001"]["stock"] = 999
        episode.environment.catalog["P1001"]["tags"].clear()
        episode.environment.policies["P1001"]["max_discount_percent"] = 0
        service.reset(episode.session_id)
        assert vars(episode.environment) == expected
        assert service.execute_tool(episode.session_id, inventory_action())["result"]["stock"] == 42


def test_environment_key_error_is_not_reported_as_missing_episode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Environment bugs retain their exception instead of becoming lookup errors.

    Args:
        monkeypatch: Pytest fixture for injecting an environment failure.
    """
    service = EpisodeService()
    episode = service.create_episode("task-a")

    def broken_execute(action: dict[str, Any]) -> dict[str, Any]:
        """Simulate an internal environment error.

        Args:
            action: Unused request passed by the service.

        Raises:
            KeyError: Always, to simulate an environment implementation bug.
        """
        raise KeyError("broken environment")

    monkeypatch.setattr(episode.environment, "execute", broken_execute)
    with pytest.raises(KeyError, match="broken environment"):
        service.execute_tool(episode.session_id, inventory_action())
    assert episode.history == []
    assert episode.step_count == 0
