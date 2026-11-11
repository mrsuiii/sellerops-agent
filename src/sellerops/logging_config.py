"""One place to configure logging for scripts and the store API.

``SELLEROPS_LOG_FORMAT=json`` emits one JSON object per line (for containers
and log collectors); ``text`` (default) is human-readable. Libraries in this
package only call ``logging.getLogger(__name__)``; entry points call
``configure_logging()`` once.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

from .settings import get_settings

# Attributes every LogRecord has; anything else was passed via ``extra=``.
_RESERVED = set(vars(logging.makeLogRecord({}))) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update({k: v for k, v in vars(record).items() if k not in _RESERVED})
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(level: str | None = None, fmt: str | None = None) -> None:
    """Configure the root logger. Safe to call more than once."""

    settings = get_settings()
    handler = logging.StreamHandler()
    if (fmt or settings.log_format) == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level or settings.log_level)
