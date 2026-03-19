from __future__ import annotations

import importlib
import json
import logging
import os
import pathlib
import re
import secrets
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from scryer_llm_runtime import _llm_manager
from scryer_tool_runtime import _tool_registry
from scryer_agent_config import get_profile, list_profiles, resolve_profile

logger = logging.getLogger("scryneuro.agent")

_DOTENV_LOADED = False


@dataclass
class ExperimentLogger:
    schema_version: str
    run_id: str
    file_path: pathlib.Path
    enabled: bool = True
    seq: int = 0

    def write(self, event: str, payload: dict[str, Any]) -> None:
        if not self.enabled:
            return
        self.seq += 1
        record = {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "seq": self.seq,
            "ts": time.time(),
            "event": event,
            "payload": payload,
        }
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        with self.file_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _iter_dotenv_candidates() -> list[pathlib.Path]:
    candidates: list[pathlib.Path] = []

    explicit = os.getenv("SCRYNEURO_DOTENV", "").strip()
    if explicit:
        candidates.append(pathlib.Path(explicit).expanduser())

    cwd = pathlib.Path.cwd().resolve()
    for parent in [cwd, *cwd.parents]:
        candidates.append(parent / ".env")

    repo_root = pathlib.Path(__file__).resolve().parents[1]
    candidates.append(repo_root / ".env")

    unique: list[pathlib.Path] = []
    seen: set[str] = set()
    for item in candidates:
        key = str(item)
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def _load_dotenv_once() -> None:
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return

    for env_path in _iter_dotenv_candidates():
        if not env_path.is_file():
            continue
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
        break

    _DOTENV_LOADED = True


def _validate_action_schema(action: dict[str, Any]) -> tuple[bool, str]:
    if not isinstance(action, dict):
        return False, "action payload must be an object"
    act = action.get("action")
    if act not in ("respond", "tool_call"):
        return False, "action must be 'respond' or 'tool_call'"
    if act == "respond":
        content = action.get("content")
        if not isinstance(content, str) or content.strip() == "":
            return False, "respond action requires non-empty string 'content'"
        return True, ""
    tool_name = action.get("tool_name")
    args = action.get("args")
    if not isinstance(tool_name, str) or tool_name.strip() == "":
        return False, "tool_call action requires string 'tool_name'"
    if not isinstance(args, dict):
        return False, "tool_call action requires object 'args'"
    return True, ""


def _validate_tool_io_schema(
    tool_name: str, args: dict[str, Any], result: Any
) -> tuple[bool, str]:
    if not isinstance(args, dict):
        return False, f"tool '{tool_name}' args must be object"
    try:
        json.dumps(result, ensure_ascii=False)
    except TypeError as e:
        return False, f"tool '{tool_name}' result must be JSON-serializable: {e}"
    return True, ""


def _parse_skill_markdown(path: pathlib.Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    lines = raw.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError(f"Skill file missing YAML frontmatter: {path}")

    frontmatter: dict[str, str] = {}
    i = 1
    while i < len(lines):
        line = lines[i].strip()
        if line == "---":
            i += 1
            break
        if line and ":" in line:
            key, value = line.split(":", 1)
            frontmatter[key.strip()] = value.strip().strip('"').strip("'")
        i += 1

    name = frontmatter.get("name", "").strip()
    description = frontmatter.get("description", "").strip()
    if not name:
        raise ValueError(f"Skill frontmatter missing required field 'name': {path}")
    if not description:
        raise ValueError(
            f"Skill frontmatter missing required field 'description': {path}"
        )

    body = "\n".join(lines[i:]).strip()

    def parse_list_field(key: str) -> list[str]:
        raw_v = frontmatter.get(key, "").strip()
        if not raw_v:
            return []
        if raw_v.startswith("[") and raw_v.endswith("]"):
            raw_v = raw_v[1:-1]
        parts = [p.strip().strip('"').strip("'") for p in raw_v.split(",")]
        return [p for p in parts if p]

    triggers = parse_list_field("triggers")
    requires_tools = parse_list_field("requires_tools")
    category = frontmatter.get("category", "").strip()
    priority_raw = frontmatter.get("priority", "0").strip()
    max_chars_raw = frontmatter.get("max_injection_chars", "").strip()
    try:
        priority = int(priority_raw)
    except ValueError:
        priority = 0
    try:
        max_injection_chars = int(max_chars_raw) if max_chars_raw else None
    except ValueError:
        max_injection_chars = None

    return {
        "name": name,
        "description": description,
        "body": body,
        "path": str(path.resolve()),
        "metadata": frontmatter,
        "triggers": triggers,
        "requires_tools": requires_tools,
        "category": category,
        "priority": priority,
        "max_injection_chars": max_injection_chars,
    }


def _tokenize_for_match(text: str) -> list[str]:
    return re.findall(r"[a-z0-9\-]+", text.lower())


def _skill_match_score(skill: dict[str, Any], user_input: str) -> int:
    haystack = f"{skill.get('name', '')} {skill.get('description', '')}"
    hay_tokens = set(_tokenize_for_match(haystack))
    in_tokens = set(_tokenize_for_match(user_input))
    overlap = len(hay_tokens & in_tokens)
    trigger_hits = 0
    for trig in skill.get("triggers", []):
        t = str(trig).lower().strip()
        if t and t in user_input.lower():
            trigger_hits += 1
    overlap += trigger_hits * 2
    if skill.get("name", "") in user_input.lower():
        overlap += 3
    overlap += int(skill.get("priority", 0))
    return overlap


def _resolve_model_id(provider: str, model_id: str) -> str:
    raw = str(model_id or "").strip()
    if provider != "openai":
        return raw
    if raw in ("", "auto", "from_env"):
        return os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    return raw


def _build_experiment_logger(
    name: str, kwargs: dict[str, Any]
) -> Optional[ExperimentLogger]:
    enabled_raw = kwargs.get("enable_experiment_log", False)
    if isinstance(enabled_raw, str):
        enabled = enabled_raw.lower() in ("1", "true", "yes", "on")
    else:
        enabled = bool(enabled_raw)
    if not enabled:
        return None

    run_id = str(kwargs.get("experiment_run_id", "")).strip()
    if not run_id:
        run_id = f"run_{int(time.time() * 1000)}_{secrets.token_hex(4)}"

    log_dir = str(kwargs.get("experiment_log_dir", "logs")).strip() or "logs"
    file_name = f"agent_{run_id}.jsonl"
    file_path = pathlib.Path(log_dir) / file_name
    return ExperimentLogger(
        schema_version="1.0",
        run_id=run_id,
        file_path=file_path,
        enabled=True,
    )


@dataclass
class ToolSpec:
    name: str
    entrypoint: str
    fn: Callable[..., Any]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentState:
    name: str
    model_id: str
    provider: str
    llm_entry: Any
    tools: dict[str, ToolSpec] = field(default_factory=dict)
    tool_docs: dict[str, str] = field(default_factory=dict)
    skills: list[dict[str, Any]] = field(default_factory=list)
    skill_index: dict[str, dict[str, Any]] = field(default_factory=dict)
    active_skill_names: list[str] = field(default_factory=list)
    skill_policy: dict[str, Any] = field(
        default_factory=lambda: {
            "mode": "hybrid",
            "max_skills": 3,
            "min_score": 1,
            "skill_total_budget_chars": 2400,
            "skill_max_chars_each": 900,
        }
    )
    messages: list[dict[str, str]] = field(default_factory=list)
    trace: list[dict[str, Any]] = field(default_factory=list)
    plugins: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    experiment_logger: Optional[ExperimentLogger] = None


class AgentManager:
    def __init__(self) -> None:
        self._agents: dict[str, AgentState] = {}

    def create(
        self,
        name: str,
        model_id: str,
        provider: str = "mock",
        **kwargs: Any,
    ) -> AgentState:
        _load_dotenv_once()
        if name in self._agents:
            raise ValueError(f"Agent '{name}' already exists")

        if provider == "openai":
            if "api_key" not in kwargs or not kwargs.get("api_key"):
                env_key = os.getenv("OPENAI_API_KEY", "")
                if env_key:
                    kwargs["api_key"] = env_key
            if "base_url" not in kwargs or not kwargs.get("base_url"):
                env_base = os.getenv("OPENAI_BASE_URL", "")
                if env_base:
                    kwargs["base_url"] = env_base
            if "api_key" not in kwargs or not kwargs.get("api_key"):
                raise ValueError(
                    "OPENAI_API_KEY is missing. Set environment variable or put it in .env"
                )

        resolved_model_id = _resolve_model_id(provider, model_id)
        experiment_logger = _build_experiment_logger(name, kwargs)

        llm_entry: Any
        if provider == "mock":
            llm_entry = {"provider": "mock", "model_id": resolved_model_id}
        else:
            llm_entry = _llm_manager.load(
                name,
                resolved_model_id,
                provider=provider,
                **kwargs,
            )

        state = AgentState(
            name=name,
            model_id=resolved_model_id,
            provider=provider,
            llm_entry=llm_entry,
            metadata=dict(kwargs),
            experiment_logger=experiment_logger,
        )
        self._agents[name] = state
        self._emit(state, "agent_created", {"provider": provider, "model_id": model_id})
        logger.info("Created agent '%s' (%s:%s)", name, provider, model_id)
        return state

    def get(self, name: str) -> AgentState:
        if name not in self._agents:
            raise KeyError(f"Agent '{name}' not found. Available: {list(self._agents)}")
        return self._agents[name]

    def list_agents(self) -> list[str]:
        return sorted(self._agents.keys())

    def unload(self, name: str) -> None:
        state = self.get(name)
        if state.provider != "mock":
            try:
                _llm_manager.unload(name)
            except Exception:
                logger.exception("Failed to unload LLM manager entry for '%s'", name)
        del self._agents[name]
        logger.info("Unloaded agent '%s'", name)

    def register_tool(
        self,
        state: AgentState,
        tool_name: str,
        entrypoint: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        description = str(kwargs.pop("description", "")).strip()
        entry = _tool_registry.register(
            tool_name,
            entrypoint,
            description=description,
            **kwargs,
        )
        spec = ToolSpec(
            name=entry.name,
            entrypoint=entry.entrypoint,
            fn=entry.fn,
            metadata=dict(kwargs),
        )
        state.tools[tool_name] = spec
        state.tool_docs[tool_name] = entry.description
        self._emit(
            state,
            "tool_registered",
            {
                "tool": tool_name,
                "entrypoint": entrypoint,
            },
        )
        return {
            "ok": True,
            "tool": tool_name,
            "entrypoint": entry.entrypoint,
            "description": entry.description,
        }

    def register_builtin_tools(
        self,
        state: AgentState,
        names: list[str],
    ) -> dict[str, Any]:
        added = _tool_registry.register_builtin(names)
        for name in added:
            entry = _tool_registry.get(name)
            spec = ToolSpec(
                name=entry.name,
                entrypoint=entry.entrypoint,
                fn=entry.fn,
                metadata=dict(entry.metadata),
            )
            state.tools[name] = spec
            state.tool_docs[name] = entry.description
            self._emit(
                state,
                "tool_registered",
                {"tool": name, "entrypoint": entry.entrypoint},
            )
        return {"ok": True, "added": added}

    def load_skill(
        self,
        state: AgentState,
        skill_name: str,
        skills_dir: str = "python/skills",
    ) -> dict[str, Any]:
        base = pathlib.Path(skills_dir).resolve()
        skill_dir = base / skill_name
        skill_file = skill_dir / "SKILL.md"
        if not skill_file.is_file():
            legacy_txt = base / f"{skill_name}.txt"
            if not legacy_txt.is_file():
                raise FileNotFoundError(f"Skill file not found: {skill_file}")
            content = legacy_txt.read_text(encoding="utf-8").strip()
            skill = {
                "name": skill_name,
                "description": f"Legacy skill {skill_name}",
                "body": content,
                "path": str(legacy_txt.resolve()),
                "metadata": {
                    "name": skill_name,
                    "description": f"Legacy skill {skill_name}",
                },
            }
        else:
            skill = _parse_skill_markdown(skill_file)

        existing = state.skill_index.get(skill["name"])
        if existing is None:
            state.skills.append(skill)
        else:
            for idx, item in enumerate(state.skills):
                if item.get("name") == skill["name"]:
                    state.skills[idx] = skill
                    break
        state.skill_index[skill["name"]] = skill
        if skill["name"] not in state.active_skill_names:
            state.active_skill_names.append(skill["name"])

        self._emit(
            state, "skill_loaded", {"skill": skill["name"], "path": skill["path"]}
        )
        return {
            "ok": True,
            "skill": skill["name"],
            "path": skill["path"],
            "description": skill["description"],
        }

    def discover_skills(
        self, skills_dir: str = "python/skills"
    ) -> list[dict[str, Any]]:
        base = pathlib.Path(skills_dir).resolve()
        if not base.is_dir():
            return []

        discovered: list[dict[str, Any]] = []
        for child in sorted(base.iterdir(), key=lambda p: p.name.lower()):
            if not child.is_dir():
                continue
            skill_file = child / "SKILL.md"
            if not skill_file.is_file():
                continue
            try:
                spec = _parse_skill_markdown(skill_file)
            except Exception:
                continue
            discovered.append(
                {
                    "name": spec.get("name", ""),
                    "description": spec.get("description", ""),
                    "path": spec.get("path", ""),
                    "category": spec.get("category", ""),
                    "priority": spec.get("priority", 0),
                    "requires_tools": spec.get("requires_tools", []),
                }
            )
        return discovered

    def list_skills(
        self,
        state: AgentState,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for s in state.skills:
            name = str(s.get("name", ""))
            rows.append(
                {
                    "name": name,
                    "description": s.get("description", ""),
                    "active": name in state.active_skill_names,
                    "requires_tools": s.get("requires_tools", []),
                    "category": s.get("category", ""),
                    "priority": s.get("priority", 0),
                    "path": s.get("path", ""),
                }
            )
        return rows

    def enable_skill(self, state: AgentState, skill_name: str) -> dict[str, Any]:
        if skill_name not in state.skill_index:
            raise KeyError(f"Skill '{skill_name}' is not loaded")
        if skill_name not in state.active_skill_names:
            state.active_skill_names.append(skill_name)
        self._emit(state, "skill_enabled", {"skill": skill_name})
        return {"ok": True, "skill": skill_name}

    def disable_skill(self, state: AgentState, skill_name: str) -> dict[str, Any]:
        state.active_skill_names = [
            s for s in state.active_skill_names if s != skill_name
        ]
        self._emit(state, "skill_disabled", {"skill": skill_name})
        return {"ok": True, "skill": skill_name}

    def set_skill_policy(self, state: AgentState, **kwargs: Any) -> dict[str, Any]:
        for key in [
            "mode",
            "max_skills",
            "min_score",
            "skill_total_budget_chars",
            "skill_max_chars_each",
        ]:
            if key in kwargs:
                state.skill_policy[key] = kwargs[key]
        self._emit(state, "skill_policy_updated", {"policy": state.skill_policy})
        return {"ok": True, "policy": state.skill_policy}

    def save_session(
        self,
        state: AgentState,
        path: str,
    ) -> dict[str, Any]:
        payload = {
            "name": state.name,
            "model_id": state.model_id,
            "provider": state.provider,
            "metadata": state.metadata,
            "skills": state.skills,
            "active_skill_names": state.active_skill_names,
            "skill_policy": state.skill_policy,
            "messages": state.messages,
            "trace": state.trace,
            "tools": [
                {
                    "name": t.name,
                    "entrypoint": t.entrypoint,
                    "metadata": t.metadata,
                    "description": state.tool_docs.get(t.name, ""),
                }
                for t in state.tools.values()
            ],
            "plugins": [
                {
                    "name": p["name"],
                    "entrypoint": p["entrypoint"],
                }
                for p in state.plugins
            ],
        }
        out_path = pathlib.Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self._emit(state, "session_saved", {"path": str(out_path.resolve())})
        return {"ok": True, "path": str(out_path.resolve())}

    def load_session(
        self,
        name: str,
        path: str,
        override_provider: str | None = None,
        override_model_id: str | None = None,
    ) -> AgentState:
        payload = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        provider = override_provider or str(payload.get("provider", "mock"))
        model_id = override_model_id or str(payload.get("model_id", "mock-v1"))
        state = self.create(name, model_id, provider=provider)

        for item in payload.get("tools", []):
            tname = str(item.get("name", "")).strip()
            entrypoint = str(item.get("entrypoint", "")).strip()
            if not tname or not entrypoint:
                continue
            metadata = item.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}
            self.register_tool(state, tname, entrypoint, **metadata)

        for skill in payload.get("skills", []):
            if not isinstance(skill, dict):
                continue
            sname = str(skill.get("name", "")).strip()
            if sname:
                state.skills.append(skill)
                state.skill_index[sname] = skill

        active_skill_names = payload.get("active_skill_names", [])
        if isinstance(active_skill_names, list):
            state.active_skill_names = [
                str(x) for x in active_skill_names if isinstance(x, str)
            ]
        if not state.active_skill_names:
            state.active_skill_names = [
                s.get("name", "") for s in state.skills if s.get("name", "")
            ]

        skill_policy = payload.get("skill_policy", {})
        if isinstance(skill_policy, dict):
            state.skill_policy.update(skill_policy)

        for plugin in payload.get("plugins", []):
            if not isinstance(plugin, dict):
                continue
            entrypoint = str(plugin.get("entrypoint", "")).strip()
            if entrypoint:
                self.load_plugin(state, entrypoint)

        messages = payload.get("messages", [])
        trace = payload.get("trace", [])
        if isinstance(messages, list):
            state.messages = [m for m in messages if isinstance(m, dict)]
        if isinstance(trace, list):
            state.trace = [t for t in trace if isinstance(t, dict)]

        self._emit(state, "session_loaded", {"path": str(pathlib.Path(path).resolve())})
        return state

    def load_plugin(
        self,
        state: AgentState,
        plugin_entrypoint: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        plugin_obj: Any = _load_entrypoint(plugin_entrypoint)
        register_fn: Any = getattr(plugin_obj, "register", None)
        if callable(register_fn):
            descriptor = register_fn(state, **kwargs)
        else:
            descriptor = plugin_obj(state, **kwargs)

        if not isinstance(descriptor, dict):
            raise ValueError("Plugin registration must return a dict descriptor")

        hooks: dict[str, Callable[..., Any]] = {}
        for key in [
            "before_plan",
            "after_plan",
            "before_tool",
            "after_tool",
            "after_step",
        ]:
            fn = descriptor.get(key)
            if fn is None:
                continue
            if not callable(fn):
                raise ValueError(f"Plugin hook '{key}' must be callable")
            hooks[key] = fn

        name = str(descriptor.get("name", plugin_entrypoint))
        state.plugins.append(
            {
                "name": name,
                "entrypoint": plugin_entrypoint,
                "hooks": hooks,
            }
        )
        self._emit(
            state, "plugin_loaded", {"plugin": name, "entrypoint": plugin_entrypoint}
        )
        return {
            "ok": True,
            "plugin": name,
            "entrypoint": plugin_entrypoint,
            "hooks": sorted(hooks.keys()),
        }

    def step(self, state: AgentState, user_input: str, **kwargs: Any) -> dict[str, Any]:
        start = time.time()
        self._emit(state, "step_start", {"input": user_input})
        state.messages.append({"role": "user", "content": user_input})

        max_auto_tools = int(kwargs.get("max_auto_tools", 3))
        final_text = ""
        steps: list[dict[str, Any]] = []

        local_input = user_input
        for tool_iter in range(max_auto_tools + 1):
            local_input = self._run_before_plan_plugins(state, local_input)
            action = self._plan_action(state, local_input, **kwargs)
            action = self._run_after_plan_plugins(state, local_input, action)
            valid_action, action_err = _validate_action_schema(action)
            if not valid_action:
                action = {
                    "action": "respond",
                    "content": f"Invalid model action schema: {action_err}",
                }
            steps.append({"phase": "plan", "action": action})

            act = str(action.get("action", "respond"))
            if act == "respond":
                final_text = str(action.get("content", ""))
                if not final_text:
                    final_text = "No response produced by model."
                state.messages.append({"role": "assistant", "content": final_text})
                out = {
                    "done": True,
                    "response": final_text,
                    "steps": steps,
                }
                out = self._run_after_step_plugins(state, out)
                self._emit(
                    state,
                    "step_end",
                    {"ok": True, "tool": None, "ms": _elapsed_ms(start)},
                )
                return out

            if act == "tool_call":
                tool_name = str(action.get("tool_name", "")).strip()
                args = action.get("args", {})
                if not isinstance(args, dict):
                    args = {"value": args}
                args = self._run_before_tool_plugins(state, tool_name, args)
                tool_result = self._invoke_tool(state, tool_name, args)
                tool_result = self._run_after_tool_plugins(
                    state,
                    tool_name,
                    args,
                    tool_result,
                )
                steps.append(
                    {"phase": "tool", "tool_name": tool_name, "result": tool_result}
                )

                if tool_result.get("ok", False):
                    observation = json.dumps(
                        {
                            "tool_name": tool_name,
                            "tool_result": tool_result.get("result"),
                        },
                        ensure_ascii=False,
                    )
                else:
                    observation = json.dumps(
                        {
                            "tool_name": tool_name,
                            "tool_error": tool_result.get("error"),
                        },
                        ensure_ascii=False,
                    )

                state.messages.append(
                    {
                        "role": "assistant",
                        "content": f"TOOL_RESULT {observation}",
                    }
                )
                local_input = (
                    "You have just received the tool result above. "
                    "Now either call another tool or respond to user."
                )
                continue

            final_text = "Invalid action from model. Must be JSON with action=respond or tool_call."
            state.messages.append({"role": "assistant", "content": final_text})
            out = {
                "done": True,
                "response": final_text,
                "steps": steps,
                "error": "invalid_action",
            }
            out = self._run_after_step_plugins(state, out)
            self._emit(
                state,
                "step_end",
                {"ok": False, "reason": "invalid_action", "ms": _elapsed_ms(start)},
            )
            return out

        final_text = "Stopped after reaching max_auto_tools limit."
        state.messages.append({"role": "assistant", "content": final_text})
        out = {
            "done": True,
            "response": final_text,
            "steps": steps,
            "error": "max_auto_tools_reached",
        }
        out = self._run_after_step_plugins(state, out)
        self._emit(
            state,
            "step_end",
            {"ok": True, "reason": "max_auto_tools", "ms": _elapsed_ms(start)},
        )
        return out

    def run(
        self,
        state: AgentState,
        user_input: str,
        max_steps: int = 5,
        **kwargs: Any,
    ) -> dict[str, Any]:
        self._log_event(
            state,
            "run_start",
            {
                "agent_name": state.name,
                "provider": state.provider,
                "model_id": state.model_id,
                "user_input": user_input,
                "options": kwargs,
            },
        )
        all_steps: list[dict[str, Any]] = []
        latest = ""
        for _ in range(max_steps):
            out = self.step(state, user_input, **kwargs)
            all_steps.append(out)
            latest = str(out.get("response", ""))
            final = {
                "done": True,
                "response": latest,
                "steps": all_steps,
            }
            self._log_event(
                state,
                "run_end",
                {
                    "done": bool(final.get("done", False)),
                    "response": str(final.get("response", "")),
                    "steps_count": len(all_steps),
                },
            )
            return final
        final = {
            "done": False,
            "response": "Max run steps reached",
            "steps": all_steps,
        }
        self._log_event(
            state,
            "run_end",
            {
                "done": bool(final.get("done", False)),
                "response": str(final.get("response", "")),
                "steps_count": len(all_steps),
            },
        )
        return final

    def trace(self, state: AgentState) -> list[dict[str, Any]]:
        return list(state.trace)

    def _plan_action(
        self, state: AgentState, user_input: str, **kwargs: Any
    ) -> dict[str, Any]:
        self._emit(
            state,
            "llm_generate",
            {
                "provider": state.provider,
                "model_id": state.model_id,
            },
        )

        if state.provider == "mock":
            return self._mock_plan(state, user_input)

        prompt = self._build_action_prompt(state, user_input)
        llm_text = _llm_manager.generate(state.name, prompt, **kwargs)
        return _extract_action_json(llm_text)

    def _mock_plan(self, state: AgentState, user_input: str) -> dict[str, Any]:
        text = user_input.strip()
        lower = text.lower()

        if lower.startswith("tool:"):
            prefix, _, rest = text.partition(":")
            _ = prefix
            tool_name, _, arg_json = rest.partition(" ")
            try:
                args = json.loads(arg_json.strip()) if arg_json.strip() else {}
            except json.JSONDecodeError as e:
                return {
                    "action": "respond",
                    "content": f"Invalid mock tool args JSON: {e}",
                }
            return {
                "action": "tool_call",
                "tool_name": tool_name.strip(),
                "args": args,
            }

        return {
            "action": "respond",
            "content": (
                f"[mock:{state.model_id}] {text} | tools={sorted(state.tools.keys())}. "
                "To trigger tool in mock mode use: tool:<name> {json}"
            ),
        }

    def _build_action_prompt(self, state: AgentState, user_input: str) -> str:
        tools_block = []
        for name in sorted(state.tools.keys()):
            desc = state.tool_docs.get(name, "")
            tools_block.append(f"- {name}: {desc}")
        tools_text = "\n".join(tools_block) if tools_block else "- (none)"

        skills_block = []
        selected_skills = self._select_skills_for_input(state, user_input)
        per_skill_limit = int(state.skill_policy.get("skill_max_chars_each", 900))
        total_budget = int(state.skill_policy.get("skill_total_budget_chars", 2400))
        used = 0
        selected_names: list[str] = []
        skipped_names: list[str] = []

        for s in selected_skills:
            requires_tools = s.get("requires_tools", [])
            missing_tools = [t for t in requires_tools if t not in state.tools]
            if missing_tools:
                skipped_names.append(
                    f"{s.get('name', '')}:missing_tools={missing_tools}"
                )
                continue

            body = str(s.get("body", ""))
            max_per_skill = s.get("max_injection_chars")
            if isinstance(max_per_skill, int) and max_per_skill > 0:
                cap = min(max_per_skill, per_skill_limit)
            else:
                cap = per_skill_limit
            body = body[:cap]

            block = (
                f"[{s['name']}]\n"
                f"description: {s.get('description', '')}\n"
                f"instructions:\n{body}"
            )
            projected = used + len(block)
            if projected > total_budget:
                skipped_names.append(f"{s.get('name', '')}:budget")
                continue
            skills_block.append(block)
            selected_names.append(str(s.get("name", "")))
            used = projected

        skills_text = "\n\n".join(skills_block) if skills_block else "(none)"
        self._emit(
            state,
            "skills_selected",
            {
                "selected": selected_names,
                "skipped": skipped_names,
                "skills_chars": used,
                "skills_budget": total_budget,
            },
        )

        history_tail = state.messages[-8:]
        history_text = "\n".join(f"{m['role']}: {m['content']}" for m in history_tail)

        return (
            "You are an agent runtime.\n"
            "You MUST reply with valid JSON object only (no markdown).\n"
            "Schema:\n"
            '{"action":"respond","content":"..."}\n'
            "or\n"
            '{"action":"tool_call","tool_name":"...","args":{...}}\n\n'
            "Tool usage policy:\n"
            "- If a tool is required to answer accurately, call tool first.\n"
            "- If enough info already exists, respond directly.\n"
            "- Never invent tool results.\n\n"
            f"Available tools:\n{tools_text}\n\n"
            f"Loaded skills:\n{skills_text}\n\n"
            f"Recent history:\n{history_text}\n\n"
            f"User input:\n{user_input}\n"
        )

    def _select_skills_for_input(
        self,
        state: AgentState,
        user_input: str,
    ) -> list[dict[str, Any]]:
        if not state.skills:
            return []

        mode = str(state.skill_policy.get("mode", "hybrid"))
        max_skills = int(state.skill_policy.get("max_skills", 3))
        min_score = int(state.skill_policy.get("min_score", 1))

        if mode == "manual":
            selected_manual: list[dict[str, Any]] = []
            for name in state.active_skill_names:
                skill = state.skill_index.get(name)
                if skill is not None:
                    selected_manual.append(skill)
            return selected_manual[:max_skills]

        ranked = sorted(
            state.skills,
            key=lambda s: _skill_match_score(s, user_input),
            reverse=True,
        )

        selected: list[dict[str, Any]] = []
        for skill in ranked:
            score = _skill_match_score(skill, user_input)
            if score < min_score:
                continue
            selected.append(skill)

        if mode in ("legacy", "hybrid") and not selected:
            for name in state.active_skill_names:
                skill = state.skill_index.get(name)
                if skill is not None:
                    selected.append(skill)
        return selected[:max_skills]

    def _invoke_tool(
        self,
        state: AgentState,
        tool_name: str,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        if tool_name not in state.tools:
            return {
                "ok": False,
                "error": f"Tool '{tool_name}' is not registered. Available: {sorted(state.tools.keys())}",
            }
        spec = state.tools[tool_name]
        self._emit(state, "tool_call", {"tool": tool_name, "args": args})
        start = time.time()
        try:
            valid_args, args_err = _validate_tool_io_schema(tool_name, args, {})
            if not valid_args:
                return {"ok": False, "error": args_err}
            result = spec.fn(**args)
            valid_result, result_err = _validate_tool_io_schema(tool_name, args, result)
            if not valid_result:
                return {"ok": False, "error": result_err}
            payload = {
                "ok": True,
                "result": result,
            }
            self._emit(
                state,
                "tool_result",
                {
                    "tool": tool_name,
                    "ok": True,
                    "ms": _elapsed_ms(start),
                },
            )
            return payload
        except Exception as e:
            self._emit(
                state,
                "tool_result",
                {
                    "tool": tool_name,
                    "ok": False,
                    "error": str(e),
                    "ms": _elapsed_ms(start),
                },
            )
            return {"ok": False, "error": str(e)}

    def _run_before_plan_plugins(self, state: AgentState, user_input: str) -> str:
        value = user_input
        for plugin in state.plugins:
            fn = plugin["hooks"].get("before_plan")
            if fn is None:
                continue
            updated = fn(state, value)
            if isinstance(updated, str):
                value = updated
        return value

    def _run_after_plan_plugins(
        self,
        state: AgentState,
        user_input: str,
        action: dict[str, Any],
    ) -> dict[str, Any]:
        value = action
        for plugin in state.plugins:
            fn = plugin["hooks"].get("after_plan")
            if fn is None:
                continue
            updated = fn(state, user_input, value)
            if isinstance(updated, dict):
                value = updated
        return value

    def _run_before_tool_plugins(
        self,
        state: AgentState,
        tool_name: str,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        value = args
        for plugin in state.plugins:
            fn = plugin["hooks"].get("before_tool")
            if fn is None:
                continue
            updated = fn(state, tool_name, value)
            if isinstance(updated, dict):
                value = updated
        return value

    def _run_after_tool_plugins(
        self,
        state: AgentState,
        tool_name: str,
        args: dict[str, Any],
        result: dict[str, Any],
    ) -> dict[str, Any]:
        value = result
        for plugin in state.plugins:
            fn = plugin["hooks"].get("after_tool")
            if fn is None:
                continue
            updated = fn(state, tool_name, args, value)
            if isinstance(updated, dict):
                value = updated
        return value

    def _run_after_step_plugins(
        self,
        state: AgentState,
        step_output: dict[str, Any],
    ) -> dict[str, Any]:
        value = step_output
        for plugin in state.plugins:
            fn = plugin["hooks"].get("after_step")
            if fn is None:
                continue
            updated = fn(state, value)
            if isinstance(updated, dict):
                value = updated
        return value

    @staticmethod
    def _log_event(state: AgentState, event: str, payload: dict[str, Any]) -> None:
        logger_inst = state.experiment_logger
        if logger_inst is None:
            return
        logger_inst.write(event, payload)

    @staticmethod
    def _emit(state: AgentState, event_type: str, payload: dict[str, Any]) -> None:
        state.trace.append(
            {
                "type": event_type,
                "time": time.time(),
                **payload,
            }
        )
        AgentManager._log_event(
            state,
            event_type,
            payload,
        )


def _extract_action_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        chunk = stripped[start : end + 1]
        try:
            parsed = json.loads(chunk)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    return {
        "action": "respond",
        "content": stripped,
    }


def _load_entrypoint(entrypoint: str) -> Callable[..., Any]:
    if ":" not in entrypoint:
        raise ValueError(
            f"Invalid tool entrypoint '{entrypoint}'. Expected 'module:function'."
        )
    module_name, func_name = entrypoint.split(":", 1)
    module = importlib.import_module(module_name)
    fn = getattr(module, func_name, None)
    if fn is None or not callable(fn):
        raise ValueError(f"Entrypoint '{entrypoint}' is not a callable")
    return fn


def _elapsed_ms(start_ts: float) -> int:
    return int((time.time() - start_ts) * 1000)


_agent_manager = AgentManager()


def agent_create(name: str, model_id: str, kwargs: Optional[dict] = None) -> Any:
    if kwargs is None:
        kwargs = {}
    provider = kwargs.pop("provider", "mock")
    return _agent_manager.create(name, model_id, provider=provider, **kwargs)


def agent_list_profiles() -> list[dict[str, Any]]:
    return list_profiles()


def agent_get_profile(profile_name: str) -> dict[str, Any]:
    return get_profile(profile_name)


def agent_create_from_profile(
    name: str,
    profile_name: str,
    kwargs: Optional[dict] = None,
) -> Any:
    if kwargs is None:
        kwargs = {}
    merged = resolve_profile(profile_name, kwargs)
    merged.pop("name", None)
    provider = str(merged.pop("provider", "openai"))
    model_id = str(merged.pop("model", "auto"))
    merged.pop("profile", None)
    return _agent_manager.create(name, model_id, provider=provider, **merged)


def agent_register_tool(
    agent_entry: Any,
    tool_name: str,
    entrypoint: str,
    kwargs: Optional[dict] = None,
) -> dict:
    if kwargs is None:
        kwargs = {}
    return _agent_manager.register_tool(agent_entry, tool_name, entrypoint, **kwargs)


def agent_register_builtin_tools(agent_entry: Any, names: list[str]) -> dict:
    return _agent_manager.register_builtin_tools(agent_entry, names)


def agent_discover_skills(kwargs: Optional[dict] = None) -> list[dict[str, Any]]:
    if kwargs is None:
        kwargs = {}
    skills_dir = str(kwargs.get("skills_dir", "python/skills"))
    return _agent_manager.discover_skills(skills_dir=skills_dir)


def agent_load_skill(
    agent_entry: Any,
    skill_name: str,
    kwargs: Optional[dict] = None,
) -> dict:
    if kwargs is None:
        kwargs = {}
    skills_dir = str(kwargs.get("skills_dir", "python/skills"))
    return _agent_manager.load_skill(agent_entry, skill_name, skills_dir=skills_dir)


def agent_load_plugin(
    agent_entry: Any,
    plugin_entrypoint: str,
    kwargs: Optional[dict] = None,
) -> dict:
    if kwargs is None:
        kwargs = {}
    return _agent_manager.load_plugin(agent_entry, plugin_entrypoint, **kwargs)


def agent_list_skills(agent_entry: Any) -> list[dict[str, Any]]:
    return _agent_manager.list_skills(agent_entry)


def agent_enable_skill(agent_entry: Any, skill_name: str) -> dict:
    return _agent_manager.enable_skill(agent_entry, skill_name)


def agent_disable_skill(agent_entry: Any, skill_name: str) -> dict:
    return _agent_manager.disable_skill(agent_entry, skill_name)


def agent_set_skill_policy(agent_entry: Any, kwargs: Optional[dict] = None) -> dict:
    if kwargs is None:
        kwargs = {}
    return _agent_manager.set_skill_policy(agent_entry, **kwargs)


def agent_save_session(agent_entry: Any, path: str) -> dict:
    return _agent_manager.save_session(agent_entry, path)


def agent_load_session(name: str, path: str, kwargs: Optional[dict] = None) -> Any:
    if kwargs is None:
        kwargs = {}
    override_provider = kwargs.get("provider")
    override_model_id = kwargs.get("model_id")
    return _agent_manager.load_session(
        name,
        path,
        override_provider=override_provider,
        override_model_id=override_model_id,
    )


def agent_step(
    agent_entry: Any,
    user_input: str,
    kwargs: Optional[dict] = None,
) -> dict:
    if kwargs is None:
        kwargs = {}
    return _agent_manager.step(agent_entry, user_input, **kwargs)


def agent_run(
    agent_entry: Any,
    user_input: str,
    kwargs: Optional[dict] = None,
) -> dict:
    if kwargs is None:
        kwargs = {}
    max_steps = int(kwargs.pop("max_steps", 5))
    return _agent_manager.run(agent_entry, user_input, max_steps=max_steps, **kwargs)


def agent_trace(agent_entry: Any) -> list[dict[str, Any]]:
    return _agent_manager.trace(agent_entry)


def agent_list() -> list[str]:
    return _agent_manager.list_agents()


def agent_unload(agent_entry: Any) -> bool:
    _agent_manager.unload(agent_entry.name)
    return True
