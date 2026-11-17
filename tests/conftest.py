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
