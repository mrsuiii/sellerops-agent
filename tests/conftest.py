from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

# LangSmith: test runs (fake models and clients) go to their own project so
# they never mix with real agent/eval traces. Load .env first so tracing
# follows LANGSMITH_TRACING there (a shell value wins: override=False), then
# force the project before langsmith reads and caches the variables.
# Opt out for one run: LANGSMITH_TRACING=false pytest
TEST_LANGSMITH_PROJECT = "unit-test-sellerops"
load_dotenv(REPO_ROOT / ".env", override=False)
os.environ["LANGSMITH_PROJECT"] = TEST_LANGSMITH_PROJECT
os.environ.pop("LANGCHAIN_PROJECT", None)

# The seller backend opens a SQLite database on first use; never let a test
# create or change the developer's var/seller.sqlite.
import tempfile

_TEST_VAR = Path(tempfile.mkdtemp(prefix="sellerops-test-"))
os.environ["SELLEROPS_SELLER_DB"] = str(_TEST_VAR / "seller.sqlite")
os.environ["SELLEROPS_APPROVALS_DB"] = str(_TEST_VAR / "approvals.sqlite")


# A seller backend on a fresh SQLite file per test (tests/seller, tests/workflows).
import itertools  # noqa: E402

import pytest  # noqa: E402


@pytest.fixture
def service(tmp_path):
    from sellerops.seller import SellerService
    from sellerops.settings import get_settings

    settings = get_settings()
    ticks = itertools.count()
    return SellerService.open(tmp_path / "seller.sqlite", catalog_path=settings.catalog_path,
                              sim_dir=settings.data_dir / "seller_sim",
                              clock=lambda: f"2026-11-18T00:00:{next(ticks) % 60:02d}+00:00")
