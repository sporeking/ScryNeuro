from __future__ import annotations

import importlib
import json
import os
import pathlib
import sys
import uuid

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))


def _agent_config_module():
    return importlib.import_module("scryer_agent.config")


def _agent_runtime_module():
    return importlib.import_module("scryer_agent.runtime")


def test_example_config_has_schema_version_and_no_embedded_api_keys() -> None:
    example_path = (
        pathlib.Path(__file__).resolve().parents[1]
        / "python"
        / "scryer_agent"
        / "config"
        / "agent_profiles.example.json"
    )
    raw = json.loads(example_path.read_text(encoding="utf-8"))

    assert raw.get("schema_version") == "1.0"
    profiles = raw.get("profiles")
    assert isinstance(profiles, dict)
    assert "default_mock" in profiles

    for profile in profiles.values():
        assert isinstance(profile, dict)
        assert not str(profile.get("api_key", "")).strip()


def test_list_profiles_falls_back_to_example_when_local_config_is_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    agent_config = _agent_config_module()
    config_dir = tmp_path / "package_config"
    config_dir.mkdir(parents=True)
    example_path = config_dir / "agent_profiles.example.json"
    example_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "profiles": {"default_mock": {"provider": "mock", "model": "mock-v2"}},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(agent_config, "_CONFIG_DIR", config_dir)

    rows = agent_config.list_profiles()

    assert rows == [
        {
            "name": "default_mock",
            "provider": "mock",
            "model": "mock-v2",
            "base_url": "",
        }
    ]


def test_agent_create_from_profile_uses_dotenv_openai_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    agent_runtime = _agent_runtime_module()
    config_path = tmp_path / "agent_profiles.example.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "profiles": {
                    "default_openai": {
                        "provider": "openai",
                        "model": "auto",
                        "temperature": 0.0,
                        "max_tokens": 900,
                    }
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(
        "OPENAI_API_KEY=dotenv-key\n"
        "OPENAI_BASE_URL=https://example.invalid/v1\n"
        "OPENAI_MODEL=gpt-dotenv\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("SCRYNEURO_AGENT_CONFIG", str(config_path))
    monkeypatch.setenv("SCRYNEURO_DOTENV", str(dotenv_path))

    captured: dict[str, object] = {}

    def fake_load(name: str, model_id: str, provider: str = "openai", **kwargs):
        captured["name"] = name
        captured["model_id"] = model_id
        captured["provider"] = provider
        captured["kwargs"] = dict(kwargs)
        return {"ok": True}

    monkeypatch.setattr(agent_runtime._llm_manager, "load", fake_load)
    monkeypatch.setattr(agent_runtime._llm_manager, "unload", lambda name: None)

    setattr(agent_runtime, "_DOTENV_LOADED", False)
    entry = agent_runtime.agent_create_from_profile(
        f"pytest_agent_{uuid.uuid4().hex}", "default_openai"
    )
    try:
        assert entry.provider == "openai"
        assert entry.model_id == "gpt-dotenv"
        assert entry.metadata["api_key"] == "dotenv-key"
        assert entry.metadata["base_url"] == "https://example.invalid/v1"
        assert captured["provider"] == "openai"
        assert captured["model_id"] == "gpt-dotenv"
        assert captured["kwargs"] == {
            "api_key": "dotenv-key",
            "base_url": "https://example.invalid/v1",
            "temperature": 0.0,
            "max_tokens": 900,
        }
    finally:
        agent_runtime.agent_unload(entry)
        setattr(agent_runtime, "_DOTENV_LOADED", False)
        for key in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL"):
            os.environ.pop(key, None)
