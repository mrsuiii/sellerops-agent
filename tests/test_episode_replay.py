"""Replay observations from saved episodes without modifying their source."""

from copy import deepcopy
import json
from typing import Any

import pytest

from sellerops.environment import SellerEnvironment
from sellerops.services.episode_service.episode_service import EpisodeService
from sellerops.services.episode_service.episode_state import EpisodeState


@pytest.fixture
def recorded_episode() -> tuple[EpisodeService, str, dict[str, Any]]:
    """Record all four tools in an episode ending exactly at its step limit.

    Returns:
        The source service, episode ID, and its exported snapshot.
    """
    service = EpisodeService()
    episode = service.create_episode("demo-offer")
    episode.max_steps = 4
    calls = [
        ("search_products", {"query": "mouse", "filters": {}}),
        ("get_inventory", {"product_id": "P1001"}),
        ("get_discount_policy", {"product_id": "P1001"}),
        ("calculate_offer", {"product_id": "P1001", "quantity": 2, "discount_percent": 10}),
    ]
    for tool, arguments in calls:
        service.execute_tool(
            episode.session_id,
            {"type": "tool_call", "tool": tool, "arguments": arguments},
        )
    return service, episode.session_id, service.snapshot(episode.session_id)


def test_replay_json_round_trip_is_repeatable_and_isolated(recorded_episode) -> None:
    """Replay a terminal episode repeatedly, even after its live state changes.

    Args:
        recorded_episode: Source service and a completed four-tool episode.
    """
    service, session_id, original = recorded_episode
    snapshot = json.loads(json.dumps(original))
    service.episodes[session_id].environment.inventory["P1001"]["stock"] = 999
    live_before = service.snapshot(session_id)
    registry_before = dict(service.episodes)

    expected = {"matched": True, "steps_checked": 4, "first_mismatch": None}
    assert service.replay(snapshot) == expected
    assert service.replay(snapshot) == expected
    assert snapshot == original
    assert service.snapshot(session_id) == live_before
    assert service.episodes == registry_before

    # A saved export is sufficient; the original registry is not required.
    fresh_service = EpisodeService()
    assert fresh_service.replay(snapshot) == expected
    assert fresh_service.episodes == {}


def test_replay_reports_first_mismatch_and_stops(recorded_episode, monkeypatch) -> None:
    """A tampered observation identifies its one-based step and stops dispatch.

    Args:
        recorded_episode: Source service and snapshot to tamper with.
        monkeypatch: Pytest fixture for counting actual tool calls.
    """
    service, session_id, snapshot = recorded_episode
    original_episode = service.snapshot(session_id)
    snapshot["history"][1]["result"]["result"]["stock"] = 999
    input_before = deepcopy(snapshot)
    calls = []
    original_execute = SellerEnvironment.execute

    def tracked_execute(environment, action):
        """Record dispatch while retaining real environment behavior.

        Args:
            environment: Temporary replay environment.
            action: Tagged tool request.

        Returns:
            The real tool observation.
        """
        calls.append(action["tool"])
        return original_execute(environment, action)

    monkeypatch.setattr(SellerEnvironment, "execute", tracked_execute)
    report = service.replay(snapshot)
    assert report["matched"] is False
    assert report["steps_checked"] == 2
    assert report["first_mismatch"]["step"] == 2
    assert report["first_mismatch"]["expected"]["result"]["stock"] == 999
    assert report["first_mismatch"]["actual"]["result"]["stock"] == 7
    assert calls == ["search_products", "get_inventory"]
    report["first_mismatch"]["action"]["arguments"]["product_id"] = "edited"
    report["first_mismatch"]["expected"]["result"]["stock"] = -1
    assert snapshot == input_before
    assert service.snapshot(session_id) == original_episode


def test_replay_uses_custom_initial_data_and_matches_business_failures() -> None:
    """Replay preserves custom starting stock and valid tool failure results."""
    environment = SellerEnvironment()
    environment.inventory["P1001"]["stock"] = 42
    episode = EpisodeState("custom", "custom-task", environment=environment)
    service = EpisodeService()
    service.episodes[episode.session_id] = episode
    for product_id in ("P1001", "missing"):
        service.execute_tool(episode.session_id, {
            "type": "tool_call", "tool": "get_inventory",
            "arguments": {"product_id": product_id},
        })
    snapshot = service.snapshot(episode.session_id)
    assert snapshot["history"][0]["result"]["result"]["stock"] == 42
    assert snapshot["history"][1]["result"]["ok"] is False
    assert service.replay(snapshot) == {
        "matched": True, "steps_checked": 2, "first_mismatch": None,
    }


def test_replay_empty_zero_budget_episode() -> None:
    """An empty history needs no tool calls, including with a zero step budget."""
    service = EpisodeService()
    episode = service.create_episode("empty")
    episode.max_steps = 0
    assert service.replay(service.snapshot(episode.session_id)) == {
        "matched": True, "steps_checked": 0, "first_mismatch": None,
    }


@pytest.mark.parametrize("damage", [
    "version", "boolean_version", "missing_initial", "invalid_table",
    "counter", "budget", "invalid_action", "invalid_observation", "not_json",
])
def test_invalid_snapshots_are_rejected_before_dispatch(
    recorded_episode, monkeypatch, damage: str,
) -> None:
    """Invalid exports must not execute even a valid prefix of their history.

    Args:
        recorded_episode: Valid source export to damage.
        monkeypatch: Pytest fixture for detecting tool dispatch.
        damage: Invalid input variant.
    """
    service, session_id, snapshot = recorded_episode
    original_episode = service.snapshot(session_id)
    if damage == "version":
        snapshot["snapshot_version"] = 2
    elif damage == "boolean_version":
        snapshot["snapshot_version"] = True
    elif damage == "missing_initial":
        del snapshot["initial_environment"]
    elif damage == "invalid_table":
        snapshot["initial_environment"]["inventory"] = []
    elif damage == "counter":
        snapshot["step_count"] = 1
    elif damage == "budget":
        snapshot["max_steps"] = 3
    elif damage == "invalid_action":
        snapshot["history"][-1]["action"]["tool"] = "unknown_tool"
    elif damage == "invalid_observation":
        snapshot["history"][-1]["result"]["ok"] = "true"
    else:
        snapshot["initial_environment"]["catalog"]["P1001"]["price_idr"] = float("nan")
    calls = []
    monkeypatch.setattr(SellerEnvironment, "execute", lambda self, action: calls.append(action))
    with pytest.raises(ValueError):
        service.replay(snapshot)
    assert calls == []
    assert service.snapshot(session_id) == original_episode


def test_replay_does_not_mask_unexpected_environment_errors(recorded_episode, monkeypatch) -> None:
    """A tool implementation bug must not be mislabeled as a result mismatch.

    Args:
        recorded_episode: Source service and valid export.
        monkeypatch: Pytest fixture for injecting a tool failure.
    """
    service, session_id, snapshot = recorded_episode
    original = deepcopy(snapshot)

    def broken_execute(environment, action):
        """Simulate an implementation error in the temporary environment.

        Args:
            environment: Unused environment instance.
            action: Unused tool action.

        Raises:
            RuntimeError: Always, as the injected failure.
        """
        raise RuntimeError("tool bug")

    monkeypatch.setattr(SellerEnvironment, "execute", broken_execute)
    with pytest.raises(RuntimeError, match="tool bug"):
        service.replay(snapshot)
    assert snapshot == original
    assert service.snapshot(session_id) == original
    assert len(service.episodes) == 1
