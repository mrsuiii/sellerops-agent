"""Runtime settings read from environment variables (and an optional ``.env``).

Stdlib only, so the core package stays dependency-free. All variables use the
``SELLEROPS_`` prefix; third-party keys (``LANGSMITH_API_KEY``,
``GEMINI_API_KEY``) keep their own names and are read by their libraries.

    from sellerops.settings import get_settings
    settings = get_settings()
    settings.catalog_path

``get_settings()`` is cached; call ``get_settings.cache_clear()`` in tests after
changing the environment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

# src/sellerops/settings.py -> repository root (valid for source checkouts and
# editable installs). Installed wheels should set SELLEROPS_PROJECT_ROOT.
_SOURCE_ROOT = Path(__file__).resolve().parents[2]

_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
_LOG_FORMATS = {"text", "json"}


def load_dotenv(path: Path) -> None:
    """Load ``KEY=VALUE`` lines into ``os.environ`` without overriding existing values.

    Supports blank lines, ``#`` comments, an optional ``export`` prefix and
    single/double-quoted values and `` # `` inline comments on unquoted values. Missing files are ignored.
    """

    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.removeprefix("export ").split("=", 1)
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        else:
            value = value.split(" #", 1)[0].strip()  # inline comment
        if key and value and key not in os.environ:
            os.environ[key] = value


def _env_path(name: str, default: Path, root: Path) -> Path:
    value = os.environ.get(name)
    if not value:
        return default
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def _env_choice(name: str, default: str, choices: set[str], *, upper: bool = False) -> str:
    value = os.environ.get(name, default).strip()
    value = value.upper() if upper else value.lower()
    if value not in choices:
        raise ValueError(f"{name}={value!r} is invalid; expected one of {sorted(choices)}")
    return value


def _env_list(name: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in os.environ.get(name, "").split(",") if item.strip())


@dataclass(frozen=True)
class Settings:
    project_root: Path
    data_dir: Path
    web_dir: Path
    results_dir: Path
    catalog_path: Path
    log_level: str
    log_format: str
    cors_origins: tuple[str, ...]

    @classmethod
    def from_env(cls) -> Settings:
        root = Path(os.environ.get("SELLEROPS_PROJECT_ROOT") or _SOURCE_ROOT).expanduser().resolve()
        load_dotenv(root / ".env")
        data_dir = _env_path("SELLEROPS_DATA_DIR", root / "data", root)
        return cls(
            project_root=root,
            data_dir=data_dir,
            web_dir=_env_path("SELLEROPS_WEB_DIR", root / "web", root),
            results_dir=_env_path("SELLEROPS_RESULTS_DIR", root / "results", root),
            catalog_path=_env_path("SELLEROPS_CATALOG_PATH", data_dir / "store" / "fashion_catalog.json", root),
            log_level=_env_choice("SELLEROPS_LOG_LEVEL", "INFO", _LOG_LEVELS, upper=True),
            log_format=_env_choice("SELLEROPS_LOG_FORMAT", "text", _LOG_FORMATS),
            cors_origins=_env_list("SELLEROPS_CORS_ORIGINS"),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()
