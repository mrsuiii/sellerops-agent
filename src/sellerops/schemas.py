"""Shared constants and helpers used across the agent, simulator and evaluation."""

from __future__ import annotations

import json
from typing import Any

MAX_AGENT_STEPS = 8


class SchemaError(ValueError):
    """Raised when generated or model-produced data violates a contract."""


def canonical_json(value: Any) -> str:
    """Stable compact JSON (sorted keys, no spaces) for tool outputs and checksums."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
