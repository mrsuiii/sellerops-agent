"""Mutable state for one isolated SellerOps episode."""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from sellerops.environment import SellerEnvironment
from sellerops.schemas import MAX_AGENT_STEPS


@dataclass
class EpisodeState:
    """Store episode identity, environment, tool history, and execution limits."""

    session_id: str
    task_id: str
    environment: SellerEnvironment = field(default_factory=SellerEnvironment)
    history: list[dict[str, Any]] = field(default_factory=list)

    step_count: int = 0
    max_steps: int = MAX_AGENT_STEPS
    terminal: bool = False

    _initial_environment: dict[str, Any] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Capture independent environment data at episode creation time."""
        self._initial_environment = deepcopy(
            {
                "catalog": self.environment.catalog,
                "inventory": self.environment.inventory,
                "policies": self.environment.policies,
            }
        )

    @property
    def initial_environment(self) -> dict[str, Any]:
        """Return a copy of the starting data without exposing the stored baseline.

        Returns:
            Independent catalog, inventory, and policy mappings for reset and
            future replay. Editing the returned data does not change the baseline.
        """
        return deepcopy(self._initial_environment)
