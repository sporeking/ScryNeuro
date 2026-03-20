from __future__ import annotations

import json
import pathlib
import sys
from typing import Any


def _ensure_python_runtime_path() -> None:
    py_dir = pathlib.Path(__file__).resolve().parents[1]
    py_dir_str = str(py_dir)
    if py_dir_str not in sys.path:
        sys.path.insert(0, py_dir_str)


_ensure_python_runtime_path()

from scryer_agent_runtime import (  # noqa: E402
    _agent_manager,
    agent_create_from_profile,
    agent_discover_skills,
    agent_get_profile,
    agent_list_profiles,
    agent_load_skill,
    agent_register_builtin_tools,
    agent_run,
    agent_trace,
    agent_unload,
)
from scryer_tool_runtime import tool_list_available  # noqa: E402


_PROVIDER_MODELS: dict[str, list[str]] = {
    "openai": ["auto", "gpt-4o-mini", "gpt-4.1", "glm-5"],
    "anthropic": ["claude-3-5-sonnet-latest", "claude-3-7-sonnet-latest"],
    "huggingface": ["Qwen/Qwen2.5-7B-Instruct", "meta-llama/Llama-3.1-8B-Instruct"],
    "ollama": ["qwen2.5:7b", "llama3.1:8b"],
    "mock": ["mock-v2", "mock-v3"],
}


_AGENTS: dict[str, Any] = {}
_AGENT_CONFIGS: dict[str, dict[str, Any]] = {}


def parse_json_object(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        return {}
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("JSON must be an object")
    return value


def to_json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def list_providers() -> list[str]:
    return list(_PROVIDER_MODELS.keys())


def list_models(provider: str) -> list[str]:
    return list(_PROVIDER_MODELS.get(str(provider), []))


def list_profiles() -> list[str]:
    rows = agent_list_profiles()
    out: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name", "")).strip()
        if name:
            out.append(name)
    return sorted(out)


def get_profile(profile_name: str) -> dict[str, Any]:
    return dict(agent_get_profile(profile_name))


def list_tools() -> list[dict[str, Any]]:
    rows = tool_list_available()
    cleaned = [r for r in rows if isinstance(r, dict)]
    cleaned.sort(key=lambda x: str(x.get("name", "")))
    return cleaned


def list_skills(skills_dir: str = "python/skills") -> list[dict[str, Any]]:
    rows = agent_discover_skills(kwargs={"skills_dir": skills_dir})
    cleaned = [r for r in rows if isinstance(r, dict)]
    cleaned.sort(key=lambda x: str(x.get("name", "")))
    return cleaned


def list_agents() -> list[str]:
    return sorted(_AGENTS.keys())


def create_or_replace_agent(
    name: str,
    profile_name: str,
    overrides: dict[str, Any] | None = None,
    tool_names: list[str] | None = None,
    skill_names: list[str] | None = None,
    skills_dir: str = "python/skills",
) -> dict[str, Any]:
    clean_name = str(name).strip()
    if not clean_name:
        raise ValueError("Agent name is required")
    if not str(profile_name).strip():
        raise ValueError("Profile name is required")

    if clean_name in _AGENTS:
        close_agent(clean_name)

    kwargs = dict(overrides or {})
    entry = agent_create_from_profile(clean_name, profile_name, kwargs=kwargs)
    _AGENTS[clean_name] = entry
    _AGENT_CONFIGS[clean_name] = {
        "profile": str(profile_name),
        "overrides": dict(kwargs),
        "tools": list(tool_names or []),
        "skills": list(skill_names or []),
        "skills_dir": str(skills_dir),
    }

    tools_added: list[str] = []
    if tool_names:
        tool_result = agent_register_builtin_tools(entry, list(tool_names))
        tools_added = [str(x) for x in tool_result.get("added", [])]

    skills_loaded: list[str] = []
    if skill_names:
        for skill_name in skill_names:
            info = agent_load_skill(
                entry,
                str(skill_name),
                kwargs={"skills_dir": skills_dir},
            )
            skills_loaded.append(str(info.get("skill", skill_name)))

    return {
        "ok": True,
        "agent": clean_name,
        "profile": profile_name,
        "tools_added": tools_added,
        "skills_loaded": skills_loaded,
        "overrides": kwargs,
    }


def run_agent(
    name: str,
    user_input: str,
    run_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    clean_name = str(name).strip()
    if clean_name not in _AGENTS:
        raise KeyError(f"Agent '{clean_name}' is not active")
    text = str(user_input).strip()
    if not text:
        raise ValueError("Input task is required")
    opts = dict(run_options or {})
    return dict(agent_run(_AGENTS[clean_name], text, kwargs=opts))


def trace_agent(name: str) -> list[dict[str, Any]]:
    clean_name = str(name).strip()
    if clean_name not in _AGENTS:
        raise KeyError(f"Agent '{clean_name}' is not active")
    trace = agent_trace(_AGENTS[clean_name])
    return [item for item in trace if isinstance(item, dict)]


def close_agent(name: str) -> bool:
    clean_name = str(name).strip()
    entry = _AGENTS.pop(clean_name, None)
    _AGENT_CONFIGS.pop(clean_name, None)
    if entry is None:
        return False
    agent_unload(entry)
    return True


def get_agent_messages(name: str) -> list[dict[str, str]]:
    clean_name = str(name).strip()
    if clean_name not in _AGENTS:
        raise KeyError(f"Agent '{clean_name}' is not active")
    state = _agent_manager.get(clean_name)
    out: list[dict[str, str]] = []
    for msg in state.messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role", "")).strip()
        content = str(msg.get("content", ""))
        if role in {"user", "assistant", "system"}:
            out.append({"role": role, "content": content})
    return out


def get_agent_summary(name: str) -> dict[str, Any]:
    clean_name = str(name).strip()
    if clean_name not in _AGENTS:
        raise KeyError(f"Agent '{clean_name}' is not active")
    state = _agent_manager.get(clean_name)
    cfg = _AGENT_CONFIGS.get(clean_name, {})
    return {
        "agent": clean_name,
        "provider": state.provider,
        "model": state.model_id,
        "message_count": len(state.messages),
        "trace_count": len(state.trace),
        "active_tools": sorted(state.tools.keys()),
        "active_skills": sorted(state.active_skill_names),
        "profile": cfg.get("profile", ""),
    }


def reset_agent(name: str) -> dict[str, Any]:
    clean_name = str(name).strip()
    if clean_name not in _AGENT_CONFIGS:
        raise KeyError(
            f"Agent '{clean_name}' has no cached config; recreate from UI first"
        )
    cfg = dict(_AGENT_CONFIGS[clean_name])
    profile = str(cfg.get("profile", ""))
    overrides = dict(cfg.get("overrides", {}))
    tools = [str(x) for x in cfg.get("tools", []) if str(x).strip()]
    skills = [str(x) for x in cfg.get("skills", []) if str(x).strip()]
    skills_dir = str(cfg.get("skills_dir", "python/skills"))

    if clean_name in _AGENTS:
        close_agent(clean_name)

    out = create_or_replace_agent(
        name=clean_name,
        profile_name=profile,
        overrides=overrides,
        tool_names=tools,
        skill_names=skills,
        skills_dir=skills_dir,
    )
    out["reset"] = True
    return out


def to_chatbot_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for msg in messages:
        role = str(msg.get("role", ""))
        content = str(msg.get("content", ""))
        if role not in {"user", "assistant"}:
            continue
        out.append({"role": role, "content": content})
    return out
