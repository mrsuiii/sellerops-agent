from __future__ import annotations

import json
import logging

import pytest

from sellerops.logging_config import JsonFormatter
from sellerops.settings import Settings, get_settings


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    env = __import__("os").environ
    for key in [k for k in env if k.startswith("SELLEROPS_") or k in {"LLM_BASE_URL", "LLM_MODEL"}]:
        monkeypatch.delenv(key)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_defaults_point_at_repository():
    settings = Settings.from_env()
    assert (settings.project_root / "pyproject.toml").is_file()
    assert settings.catalog_path == settings.project_root / "data" / "store" / "fashion_catalog.json"
    assert settings.catalog_path.is_file()
    assert settings.log_level == "INFO" and settings.log_format == "text"
    assert settings.cors_origins == ()
    assert settings.llm_provider == "anthropic"
    assert settings.llm_model == "claude-haiku-5-5" and settings.llm_effort == "medium"
    assert settings.llm_fallback_models == ()


def test_env_overrides_resolve_relative_to_root(monkeypatch, tmp_path):
    monkeypatch.setenv("SELLEROPS_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("SELLEROPS_CATALOG_PATH", "custom/catalog.json")
    monkeypatch.setenv("SELLEROPS_LOG_LEVEL", "debug")
    monkeypatch.setenv("SELLEROPS_LOG_FORMAT", "JSON")
    monkeypatch.setenv("SELLEROPS_CORS_ORIGINS", "http://a.test, http://b.test ,")
    settings = Settings.from_env()
    assert settings.project_root == tmp_path.resolve()
    assert settings.data_dir == tmp_path.resolve() / "data"
    assert settings.catalog_path == tmp_path.resolve() / "custom" / "catalog.json"
    assert settings.log_level == "DEBUG" and settings.log_format == "json"
    assert settings.cors_origins == ("http://a.test", "http://b.test")


def test_llm_overrides(monkeypatch):
    monkeypatch.setenv("SELLEROPS_LLM_MODEL", " claude-sonnet-5-5 ")
    monkeypatch.setenv("SELLEROPS_LLM_EFFORT", "HIGH")
    monkeypatch.setenv("SELLEROPS_LLM_FALLBACK_MODELS", "claude-sonnet-5-5, claude-opus-5-5")
    settings = Settings.from_env()
    assert settings.llm_model == "claude-sonnet-5-5" and settings.llm_effort == "high"
    assert settings.llm_fallback_models == ("claude-sonnet-5-5", "claude-opus-5-5")
    monkeypatch.setenv("SELLEROPS_LLM_EFFORT", "turbo")
    with pytest.raises(ValueError, match="SELLEROPS_LLM_EFFORT"):
        Settings.from_env()


def test_openai_compatible_provider(monkeypatch, tmp_path):
    monkeypatch.setenv("SELLEROPS_PROJECT_ROOT", str(tmp_path))  # keep a local .env out of the test
    monkeypatch.setenv("SELLEROPS_LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("SELLEROPS_LLM_MODEL", "claude-sonnet-5-5")  # ignored for this provider
    settings = Settings.from_env()
    assert settings.llm_provider == "openai_compatible"
    assert settings.llm_base_url == "https://integrate.api.nvidia.com/v1"
    assert settings.llm_model == ""
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:8001/v1")
    monkeypatch.setenv("LLM_MODEL", "meta/llama-3.3-70b-instruct")
    settings = Settings.from_env()
    assert settings.llm_base_url == "http://localhost:8001/v1"
    assert settings.llm_model == "meta/llama-3.3-70b-instruct"
    monkeypatch.setenv("SELLEROPS_LLM_PROVIDER", "gemini")
    with pytest.raises(ValueError, match="SELLEROPS_LLM_PROVIDER"):
        Settings.from_env()


def test_invalid_choice_fails_fast(monkeypatch):
    monkeypatch.setenv("SELLEROPS_LOG_FORMAT", "xml")
    with pytest.raises(ValueError, match="SELLEROPS_LOG_FORMAT"):
        Settings.from_env()


def test_dotenv_file_is_loaded_without_overriding_env(monkeypatch, tmp_path):
    (tmp_path / ".env").write_text("SELLEROPS_LOG_LEVEL=DEBUG\nSELLEROPS_LOG_FORMAT=json\n")
    monkeypatch.setenv("SELLEROPS_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("SELLEROPS_LOG_FORMAT", "text")
    # Register the key with monkeypatch so the value dotenv writes is removed afterwards.
    monkeypatch.setenv("SELLEROPS_LOG_LEVEL", "unused")
    monkeypatch.delenv("SELLEROPS_LOG_LEVEL")
    settings = Settings.from_env()
    assert settings.log_level == "DEBUG"
    assert settings.log_format == "text"


def test_json_formatter_includes_extra_fields():
    record = logging.makeLogRecord(
        {"name": "x", "levelname": "INFO", "msg": "hello %s", "args": ("world",), "request_id": "abc"}
    )
    payload = json.loads(JsonFormatter().format(record))
    assert payload["message"] == "hello world"
    assert payload["request_id"] == "abc"
    assert payload["level"] == "INFO"
