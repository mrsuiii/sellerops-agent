"""Manage isolated episodes and record calls to the local SellerOps tools."""

from copy import deepcopy
import json
from typing import Any
from uuid import uuid4

from sellerops.environment import SellerEnvironment
from sellerops.schemas import canonical_json, validate_action, validate_tool_observation
from sellerops.services.episode_service.episode_state import EpisodeState


class EpisodeService:
    """Own episode lifecycle, tool limits, and independent history records."""

    def __init__(self) -> None:
        """Initialize an empty episode registry."""
        self.episodes: dict[str, EpisodeState] = {}

    def create_episode(self, task_id: str) -> EpisodeState:
        """Create an isolated episode with a new session ID.

        Args:
            task_id: Identifier of the task assigned to this episode.

        Returns:
            The registered episode, with its own environment and history.
        """
        session_id = str(uuid4())
        episode_state = EpisodeState(session_id=session_id, task_id=task_id)
        self.episodes[session_id] = episode_state
        return episode_state

    def _get_episode(self, session_id: str) -> EpisodeState:
        """Look up an episode without masking errors from tool execution.

        Args:
            session_id: Identifier returned when the episode was created.

        Returns:
            The existing episode state.

        Raises:
            ValueError: If the session ID is not registered.
        """
        try:
            return self.episodes[session_id]
        except KeyError as error:
            raise ValueError(f"No episode found with session_id {session_id}") from error

    def reset(self, session_id: str) -> None:
        """Restore an existing episode to its captured starting environment.

        The session ID, task ID, and configured step limit are preserved.

        Args:
            session_id: Identifier of the episode to restart.

        Raises:
            ValueError: If the session ID is not registered.
        """
        episode_state = self._get_episode(session_id)
        initial = episode_state.initial_environment
        environment = SellerEnvironment()
        environment.catalog = initial["catalog"]
        environment.inventory = initial["inventory"]
        environment.policies = initial["policies"]
        episode_state.environment = environment
        episode_state.history.clear()
        episode_state.step_count = 0
        episode_state.terminal = False

    def snapshot(self, session_id: str) -> dict[str, Any]:
        """Export an independent, JSON-serializable snapshot of one episode.

        Initial environment data records the starting point for future replay;
        environment records the current data. Only explicit data fields are
        exported, never the live SellerEnvironment object. This method neither
        writes a file nor restores or replays the episode.

        Args:
            session_id: Identifier of the episode to export.

        Returns:
            A versioned dictionary containing episode identity, initial/current
            environment data, tool history, execution count, limit, and status.

        Raises:
            ValueError: If the session ID is not registered.
        """
        episode_state = self._get_episode(session_id)
        environment = episode_state.environment
        return deepcopy(
            {
                "snapshot_version": 1,
                "session_id": episode_state.session_id,
                "task_id": episode_state.task_id,
                "initial_environment": episode_state.initial_environment,
                "environment": {
                    "catalog": environment.catalog,
                    "inventory": environment.inventory,
                    "policies": environment.policies,
                },
                "history": episode_state.history,
                "step_count": episode_state.step_count,
                "max_steps": episode_state.max_steps,
                "terminal": episode_state.terminal,
            }
        )

    def execute_tool(self, session_id: str, action: dict[str, Any]) -> dict[str, Any]:
        """Execute one tool call and record an independent copy of its result.

        Each returned observation counts as a step, including business-rule
        failures. Actions rejected by schema validation do not count as tool
        executions; the future agent loop must separately limit invalid outputs.

        Args:
            session_id: Identifier of the episode whose tools should run.
            action: Tagged tool call accepted by SellerEnvironment.execute.

        Returns:
            The tool's structured success or failure observation.

        Raises:
            ValueError: If the episode is missing, terminal, out of steps, or
                the action is not a tool call.
            SchemaError: If the action fails environment schema validation.
        """
        episode_state = self._get_episode(session_id)
        if episode_state.terminal:
            raise ValueError(f"Episode with session_id {session_id} is already terminal.")
        if episode_state.step_count >= episode_state.max_steps:
            episode_state.terminal = True
            raise ValueError(f"Episode with session_id {session_id} has reached its step limit.")

        recorded_action = deepcopy(action)
        result = episode_state.environment.execute(action)
        episode_state.history.append(
            {"action": recorded_action, "result": deepcopy(result)}
        )
        episode_state.step_count += 1
        if episode_state.step_count >= episode_state.max_steps:
            episode_state.terminal = True
        return result

    def replay(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        """Replay recorded tool calls from initial data in a temporary episode.

        The supplied snapshot and this service's episode registry are unchanged.
        Replay checks recorded tool observations only, not final task success,
        the current environment export, or externally changed terminal flags.
        No model is called. The temporary episode is discarded after replay.

        Args:
            snapshot: A version-1 snapshot, optionally loaded back from JSON.

        Returns:
            matched, steps_checked (including a mismatching step), and
            first_mismatch. A mismatch contains a one-based step number, action,
            expected observation, and actual observation. Replay stops there.

        Raises:
            ValueError: If the snapshot is malformed or unsupported. SchemaError
                from invalid actions/observations is also a ValueError subtype.
            Exception: Unexpected tool implementation errors propagate to the
                caller instead of being reported as observation mismatches.
        """
        self._validate_replay_snapshot(snapshot)
        recorded = deepcopy(snapshot)
        initial = recorded["initial_environment"]
        environment = SellerEnvironment()
        environment.catalog = initial["catalog"]
        environment.inventory = initial["inventory"]
        environment.policies = initial["policies"]

        replay_service = EpisodeService()
        episode = EpisodeState(
            session_id=str(uuid4()),
            task_id=recorded["task_id"],
            environment=environment,
            max_steps=recorded["max_steps"],
        )
        replay_service.episodes[episode.session_id] = episode

        for step, event in enumerate(recorded["history"], start=1):
            actual = replay_service.execute_tool(episode.session_id, event["action"])
            if canonical_json(actual) != canonical_json(event["result"]):
                return {
                    "matched": False,
                    "steps_checked": step,
                    "first_mismatch": {
                        "step": step,
                        "action": deepcopy(event["action"]),
                        "expected": deepcopy(event["result"]),
                        "actual": actual,
                    },
                }

        return {
            "matched": True,
            "steps_checked": len(recorded["history"]),
            "first_mismatch": None,
        }

    @staticmethod
    def _validate_replay_snapshot(snapshot: dict[str, Any]) -> None:
        """Validate replay input before executing any recorded tool call.

        Args:
            snapshot: Candidate versioned episode export.

        Raises:
            ValueError: If required data, version, counters, or JSON values are
                invalid. Existing schema validators check every recorded action
                and observation. Environment business data is interpreted by
                SellerEnvironment during replay.
        """
        required = {
            "snapshot_version", "session_id", "task_id", "initial_environment",
            "environment", "history", "step_count", "max_steps", "terminal",
        }
        if not isinstance(snapshot, dict) or not required <= snapshot.keys():
            raise ValueError("Replay requires a complete episode snapshot")
        version = snapshot["snapshot_version"]
        if type(version) is not int or version != 1:
            raise ValueError("Unsupported snapshot_version; expected 1")
        for key in ("session_id", "task_id"):
            if not isinstance(snapshot[key], str) or not snapshot[key].strip():
                raise ValueError(f"Snapshot {key} must be a non-empty string")
        for key in ("step_count", "max_steps"):
            if type(snapshot[key]) is not int or snapshot[key] < 0:
                raise ValueError(f"Snapshot {key} must be a non-negative integer")
        if not isinstance(snapshot["terminal"], bool):
            raise ValueError("Snapshot terminal must be boolean")
        history = snapshot["history"]
        if not isinstance(history, list) or len(history) != snapshot["step_count"]:
            raise ValueError("Snapshot history length must equal step_count")
        if snapshot["step_count"] > snapshot["max_steps"]:
            raise ValueError("Snapshot history exceeds max_steps")

        for key in ("initial_environment", "environment"):
            data = snapshot[key]
            if not isinstance(data, dict) or set(data) != {"catalog", "inventory", "policies"}:
                raise ValueError(f"Snapshot {key} must contain catalog, inventory, and policies")
            for table in data.values():
                if not isinstance(table, dict):
                    raise ValueError(f"Snapshot {key} tables must be dictionaries")
                for product_id, row in table.items():
                    if not isinstance(product_id, str) or not isinstance(row, dict):
                        raise ValueError(f"Snapshot {key} contains an invalid product row")
                    if row.get("product_id") != product_id:
                        raise ValueError(f"Snapshot {key} product_id does not match its key")

        for event in history:
            if not isinstance(event, dict) or set(event) != {"action", "result"}:
                raise ValueError("Snapshot history entries must contain action and result")
            validate_action(event["action"])
            if event["action"]["type"] != "tool_call":
                raise ValueError("Replay history must contain only tool_call actions")
            validate_tool_observation(event["result"])
            if event["action"]["tool"] != event["result"]["tool"]:
                raise ValueError("Recorded observation tool does not match the action")
        try:
            json.dumps(snapshot, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError("Snapshot must contain JSON-serializable data") from error
