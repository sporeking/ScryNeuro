from __future__ import annotations

import pathlib
import sys

import pytest


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))


@pytest.fixture(autouse=True)
def clear_agent_env(monkeypatch: pytest.MonkeyPatch):
    for key in (
        "SCRYNEURO_AGENT_CONFIG",
        "SCRYNEURO_DOTENV",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_MODEL",
    ):
        monkeypatch.delenv(key, raising=False)
