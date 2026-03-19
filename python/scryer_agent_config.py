from __future__ import annotations

import json
import os
import pathlib
from typing import Any


def _default_config_path() -> pathlib.Path:
    explicit = os.getenv("SCRYNEURO_AGENT_CONFIG", "").strip()
    if explicit:
        return pathlib.Path(explicit).expanduser().resolve()
    return pathlib.Path("python/config/agent_profiles.json").resolve()


def _load_config() -> dict[str, Any]:
    path = _default_config_path()
    if not path.is_file():
        raise FileNotFoundError(f"Agent config file not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Agent config must be a JSON object")

    local_path = path.with_suffix(".local.json")
    if local_path.is_file():
        local_raw = json.loads(local_path.read_text(encoding="utf-8"))
        if not isinstance(local_raw, dict):
            raise ValueError("Agent local config must be a JSON object")
        raw = _deep_merge(raw, local_raw)
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
