"""Runtime settings read from environment variables (and an optional ``.env``).

``.env`` is loaded with python-dotenv; real environment variables win. All variables use the
``SELLEROPS_`` prefix; third-party keys (``LANGSMITH_API_KEY``,
``ANTHROPIC_API_KEY``, ``LLM_API_KEY``) keep their own names and
are read by their libraries or the model adapter, so they are never stored on
``Settings``. Because ``.env`` is
loaded into ``os.environ``, ``anthropic.Anthropic()`` picks up a key from ``.env``
once ``get_settings()`` has run.

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

from dotenv import load_dotenv

# src/sellerops/settings.py -> repository root (valid for source checkouts and
# editable installs). Installed wheels should set SELLEROPS_PROJECT_ROOT.
_SOURCE_ROOT = Path(__file__).resolve().parents[2]

_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
_LOG_FORMATS = {"text", "json"}
_LLM_EFFORTS = {"low", "medium", "high", "xhigh", "max"}
_LLM_PROVIDERS = {"anthropic", "openai_compatible"}
DEFAULT_LLM_MODEL = "claude-haiku-5-5"
DEFAULT_LLM_BASE_URL = "https://integrate.api.nvidia.com/v1"


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



def _llm_model(provider: str) -> str:
    # Claude uses SELLEROPS_LLM_MODEL; the OpenAI-compatible endpoint keeps LLM_MODEL
    # (no default: the id depends on the provider's catalog).
    if provider == "anthropic":
        return os.environ.get("SELLEROPS_LLM_MODEL", "").strip() or DEFAULT_LLM_MODEL
    return os.environ.get("LLM_MODEL", "").strip()


@dataclass(frozen=True)
class Settings:
    project_root: Path
    data_dir: Path
    web_dir: Path
    results_dir: Path
    catalog_path: Path
    seller_db_path: Path  # SQLite database of the seller backend (captions, audit log)
    approvals_db_path: Path  # LangGraph checkpoints of the caption approval workflow
    images_dir: Path  # prepared product images for the model (seller/images.py), by store and hash
    log_level: str
    log_format: str
    cors_origins: tuple[str, ...]
    llm_provider: str
    llm_model: str
    llm_effort: str
    llm_base_url: str  # only used by the openai_compatible provider
    llm_fallback_models: tuple[str, ...]  # same provider; tried in order when the primary fails
    llm_vision: bool  # openai_compatible only: the configured model reads images (Claude always does)

    @classmethod
    def from_env(cls) -> Settings:
        root = Path(os.environ.get("SELLEROPS_PROJECT_ROOT") or _SOURCE_ROOT).expanduser().resolve()
        load_dotenv(root / ".env", override=False)
        data_dir = _env_path("SELLEROPS_DATA_DIR", root / "data", root)
        provider = _env_choice("SELLEROPS_LLM_PROVIDER", "anthropic", _LLM_PROVIDERS)
        return cls(
            project_root=root,
            data_dir=data_dir,
            web_dir=_env_path("SELLEROPS_WEB_DIR", root / "web", root),
            results_dir=_env_path("SELLEROPS_RESULTS_DIR", root / "results", root),
            catalog_path=_env_path("SELLEROPS_CATALOG_PATH", data_dir / "store" / "fashion_catalog.json", root),
            seller_db_path=_env_path("SELLEROPS_SELLER_DB", root / "var" / "seller.sqlite", root),
            approvals_db_path=_env_path("SELLEROPS_APPROVALS_DB", root / "var" / "approvals.sqlite", root),
            images_dir=_env_path("SELLEROPS_IMAGES_DIR", root / "var" / "images", root),
            log_level=_env_choice("SELLEROPS_LOG_LEVEL", "INFO", _LOG_LEVELS, upper=True),
            log_format=_env_choice("SELLEROPS_LOG_FORMAT", "text", _LOG_FORMATS),
            cors_origins=_env_list("SELLEROPS_CORS_ORIGINS"),
            llm_provider=provider,
            llm_model=_llm_model(provider),
            llm_effort=_env_choice("SELLEROPS_LLM_EFFORT", "medium", _LLM_EFFORTS),
            llm_base_url=os.environ.get("LLM_BASE_URL", "").strip() or DEFAULT_LLM_BASE_URL,
            llm_vision=os.environ.get("SELLEROPS_LLM_VISION", "").strip() == "1",
            llm_fallback_models=_env_list("SELLEROPS_LLM_FALLBACK_MODELS"),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()
