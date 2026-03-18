from __future__ import annotations

import importlib
import json
import logging
import os
import pathlib
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from scryer_llm_runtime import _llm_manager

logger = logging.getLogger("scryneuro.agent")

_DOTENV_LOADED = False


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


def _resolve_model_id(provider: str, model_id: str) -> str:
    raw = str(model_id or "").strip()
    if provider != "openai":
        return raw
    if raw in ("", "auto", "from_env"):
        return os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    return raw


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
    skills: list[dict[str, str]] = field(default_factory=list)
    messages: list[dict[str, str]] = field(default_factory=list)
    trace: list[dict[str, Any]] = field(default_factory=list)
    plugins: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


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
        fn = _load_entrypoint(entrypoint)
        description = str(kwargs.get("description", "")).strip()
        if not description:
            description = _extract_doc(fn)
        spec = ToolSpec(
            name=tool_name,
            entrypoint=entrypoint,
            fn=fn,
            metadata=dict(kwargs),
        )
        state.tools[tool_name] = spec
        state.tool_docs[tool_name] = description
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
            "entrypoint": entrypoint,
            "description": description,
        }

    def register_builtin_tools(
        self,
        state: AgentState,
        names: list[str],
    ) -> dict[str, Any]:
        imported = importlib.import_module("scryer_agent_tools")
        mapping = {
            "web_fetch": "tool_web_fetch",
            "shell_exec": "tool_shell_exec",
            "read_file": "tool_read_file",
            "write_file": "tool_write_file",
            "list_dir": "tool_list_dir",
            "grep_text": "tool_grep_text",
            "add": "add_numbers",
            "multiply": "multiply_numbers",
            "reverse": "reverse_text",
        }

        added: list[str] = []
        for name in names:
            if name not in mapping:
                raise ValueError(f"Unknown built-in tool '{name}'")
            fn_name = mapping[name]
            fn = getattr(imported, fn_name)
            entrypoint = f"scryer_agent_tools:{fn_name}"
            description = _extract_doc(fn)
            spec = ToolSpec(
                name=name, entrypoint=entrypoint, fn=fn, metadata={"builtin": True}
            )
            state.tools[name] = spec
            state.tool_docs[name] = description
            added.append(name)
            self._emit(
                state, "tool_registered", {"tool": name, "entrypoint": entrypoint}
            )
        return {"ok": True, "added": added}

    def load_skill(
        self,
        state: AgentState,
        skill_name: str,
        skills_dir: str = "python/skills",
    ) -> dict[str, Any]:
        base = os.path.abspath(skills_dir)
        path = os.path.join(base, f"{skill_name}.txt")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Skill file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        state.skills.append({"name": skill_name, "content": content})
        self._emit(state, "skill_loaded", {"skill": skill_name, "path": path})
        return {"ok": True, "skill": skill_name, "path": path}

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
            scontent = str(skill.get("content", ""))
            if sname:
                state.skills.append({"name": sname, "content": scontent})

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
        all_steps: list[dict[str, Any]] = []
        latest = ""
        for _ in range(max_steps):
            out = self.step(state, user_input, **kwargs)
            all_steps.append(out)
            latest = str(out.get("response", ""))
            return {
                "done": True,
                "response": latest,
                "steps": all_steps,
            }
        return {
            "done": False,
            "response": "Max run steps reached",
            "steps": all_steps,
        }

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
        for s in state.skills:
            skills_block.append(f"[{s['name']}]\n{s['content']}")
        skills_text = "\n\n".join(skills_block) if skills_block else "(none)"

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
    def _emit(state: AgentState, event_type: str, payload: dict[str, Any]) -> None:
        state.trace.append(
            {
                "type": event_type,
                "time": time.time(),
                **payload,
            }
        )


def _extract_doc(fn: Callable[..., Any]) -> str:
    raw = (fn.__doc__ or "").strip()
    if not raw:
        return ""
    first = raw.splitlines()[0].strip()
    return first


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


def _elapsed_ms(start_ts: float) -> int:
    return int((time.time() - start_ts) * 1000)


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


_agent_manager = AgentManager()


def agent_create(name: str, model_id: str, kwargs: Optional[dict] = None) -> Any:
    if kwargs is None:
        kwargs = {}
    provider = kwargs.pop("provider", "mock")
    return _agent_manager.create(name, model_id, provider=provider, **kwargs)


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
