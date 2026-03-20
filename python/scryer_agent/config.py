from __future__ import annotations

import json
import os
import pathlib
from typing import Any


_PACKAGE_DIR = pathlib.Path(__file__).resolve().parent
_PYTHON_DIR = _PACKAGE_DIR.parent
_CONFIG_DIR = _PACKAGE_DIR / "config"
_LEGACY_CONFIG_DIR = _PYTHON_DIR / "config"
_DEFAULT_SCHEMA_VERSION = "1.0"


def _default_config_path() -> pathlib.Path:
    explicit = os.getenv("SCRYNEURO_AGENT_CONFIG", "").strip()
    if explicit:
        return pathlib.Path(explicit).expanduser().resolve()
    primary = (_CONFIG_DIR / "agent_profiles.json").resolve()
    if primary.is_file():
        return primary
    legacy_primary = (_LEGACY_CONFIG_DIR / "agent_profiles.json").resolve()
    if legacy_primary.is_file():
        return legacy_primary
    return (_CONFIG_DIR / "agent_profiles.example.json").resolve()


def _default_local_override_path(config_path: pathlib.Path) -> pathlib.Path:
    explicit = os.getenv("SCRYNEURO_AGENT_CONFIG", "").strip()
    if explicit:
        return config_path.with_suffix(".local.json")
    preferred = (_CONFIG_DIR / "agent_profiles.local.json").resolve()
    if preferred.is_file():
        return preferred
    legacy = (_LEGACY_CONFIG_DIR / "agent_profiles.local.json").resolve()
    if legacy.is_file():
        return legacy
    return preferred


def _load_json_object(path: pathlib.Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} file not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{label} must be a JSON object")
    return raw


def _load_config() -> dict[str, Any]:
    path = _default_config_path()
    if not path.is_file():
        primary = (_CONFIG_DIR / "agent_profiles.json").resolve()
        legacy_primary = (_LEGACY_CONFIG_DIR / "agent_profiles.json").resolve()
        example = (_CONFIG_DIR / "agent_profiles.example.json").resolve()
        raise FileNotFoundError(
            "Agent config file not found. "
            f"Checked package config '{primary}', legacy config '{legacy_primary}', and example config '{example}'. "
            "Create a local python/scryer_agent/config/agent_profiles.json or python/config/agent_profiles.json, or set "
            "SCRYNEURO_AGENT_CONFIG=/abs/path/to/agent_profiles.json."
        )
    raw = _load_json_object(path, "Agent config")
    raw.setdefault("schema_version", _DEFAULT_SCHEMA_VERSION)
    profiles = raw.get("profiles", {})
    if not isinstance(profiles, dict):
        raise ValueError("Agent config field 'profiles' must be a JSON object")

    local_path = _default_local_override_path(path)
    if local_path.is_file():
        local_raw = _load_json_object(local_path, "Agent local config")
        raw = _deep_merge(raw, local_raw)
        raw.setdefault("schema_version", _DEFAULT_SCHEMA_VERSION)
        profiles = raw.get("profiles", {})
        if not isinstance(profiles, dict):
            raise ValueError("Agent config field 'profiles' must be a JSON object")
    return raw


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = dict(base)
    for key, value in override.items():
        base_value = out.get(key)
        if isinstance(base_value, dict) and isinstance(value, dict):
            out[key] = _deep_merge(base_value, value)
        else:
            out[key] = value
    return out


def list_profiles() -> list[dict[str, Any]]:
    cfg = _load_config()
    profiles = cfg.get("profiles", {})
    if not isinstance(profiles, dict):
        profiles = {}
    rows: list[dict[str, Any]] = []
    for name in sorted(profiles.keys()):
        p = profiles[name]
        if not isinstance(p, dict):
            continue
        rows.append(
            {
                "name": name,
                "provider": p.get("provider", ""),
                "model": p.get("model", ""),
                "base_url": p.get("base_url", ""),
            }
        )
    return rows


def get_profile(name: str) -> dict[str, Any]:
    cfg = _load_config()
    profiles = cfg.get("profiles", {})
    if not isinstance(profiles, dict):
        profiles = {}
    if name not in profiles:
        raise KeyError(f"Profile '{name}' not found")
    p = profiles[name]
    if not isinstance(p, dict):
        raise ValueError(f"Profile '{name}' must be an object")
    return {
        "name": name,
        **p,
    }


def resolve_profile(
    name: str, overrides: dict[str, Any] | None = None
) -> dict[str, Any]:
    if overrides is None:
        overrides = {}
    p = get_profile(name)

    provider = str(overrides.get("provider", p.get("provider", "openai")))
    model = str(overrides.get("model", p.get("model", "auto")))
    api_key = str(overrides.get("api_key", p.get("api_key", "")))
    base_url = str(overrides.get("base_url", p.get("base_url", "")))

    out: dict[str, Any] = {
        "profile": name,
        "provider": provider,
        "model": model,
    }
    if api_key:
        out["api_key"] = api_key
    if base_url:
        out["base_url"] = base_url

    for k, v in p.items():
        if k in out or k in ("provider", "model", "api_key", "base_url"):
            continue
        out[k] = v
    for k, v in overrides.items():
        if k in ("provider", "model", "api_key", "base_url"):
            continue
        out[k] = v
    return out
