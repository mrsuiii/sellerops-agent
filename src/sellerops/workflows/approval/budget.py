"""LLM budget for the public demo: how many agent steps a visitor (and the whole demo) may start.

Only routes that run the LLM take from it: a new caption request, an answer
to the agent's questions, a retry, and a decision that rejects (it drafts a
revision). Approve and cancel are free. Two limits:

- **per client per hour** (sliding window), so one visitor cannot use it all;
- **per day for everyone** (UTC day), so a shared link cannot run up the bill.

The budget lives in the process, like the approval locks: the demo runs one
machine. With several, move it to a shared store (Redis, the database).
Disabled unless ``SELLEROPS_DEMO_MODE=1``.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from sellerops.seller import SellerError

HOUR = 3600.0


class LLMBudget:
    def __init__(self, *, per_client_hour: int, per_day: int, enabled: bool = True,
                 clock: Callable[[], float] = time.time) -> None:
        self.per_client_hour, self.per_day, self.enabled = per_client_hour, per_day, enabled
        self._clock = clock
        self._lock = threading.Lock()
        self._recent: dict[str, deque[float]] = {}
        self._day, self._used_today = "", 0

    def take(self, client: str) -> None:
        """Use one step, or raise ``SellerError("llm_budget_exhausted")`` (HTTP 429)."""
        if not self.enabled:
            return
        with self._lock:
            now = self._clock()
            self._roll_day(now)
            recent = self._window(client, now)
            if self._used_today >= self.per_day:
                raise SellerError("llm_budget_exhausted",
                                  "the demo's agent budget for today is used up; try again tomorrow (UTC)")
            if len(recent) >= self.per_client_hour:
                wait = int(recent[0] + HOUR - now) // 60 + 1
                raise SellerError("llm_budget_exhausted",
                                  f"demo limit: {self.per_client_hour} agent steps per hour; try again in {wait} min")
            recent.append(now)
            self._used_today += 1

    def remaining(self, client: str) -> dict[str, Any]:
        if not self.enabled:
            return {"enabled": False}
        with self._lock:
            now = self._clock()
            self._roll_day(now)
            return {"enabled": True,
                    "per_client_hour": self.per_client_hour,
                    "remaining_hour": max(0, self.per_client_hour - len(self._window(client, now))),
                    "remaining_today": max(0, self.per_day - self._used_today)}

    def _window(self, client: str, now: float) -> deque[float]:
        recent = self._recent.setdefault(client, deque())
        while recent and recent[0] <= now - HOUR:
            recent.popleft()
        if not recent:                       # forget idle clients so the map does not grow forever
            self._recent.pop(client, None)
            recent = self._recent.setdefault(client, deque())
        return recent

    def _roll_day(self, now: float) -> None:
        day = datetime.fromtimestamp(now, UTC).date().isoformat()
        if day != self._day:
            self._day, self._used_today = day, 0
            self._recent = {k: v for k, v in self._recent.items() if v}
