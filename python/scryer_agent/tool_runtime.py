from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Callable


_BUILTIN_ENTRYPOINTS: dict[str, str] = {
    "web_fetch": "scryer_agent.tools:tool_web_fetch",
    "shell_exec": "scryer_agent.tools:tool_shell_exec",
    "read_file": "scryer_agent.tools:tool_read_file",
    "write_file": "scryer_agent.tools:tool_write_file",
    "list_dir": "scryer_agent.tools:tool_list_dir",
    "grep_text": "scryer_agent.tools:tool_grep_text",
}

_BUILTIN_DESCRIPTIONS: dict[str, str] = {
    "web_fetch": "Fetch a URL and return content preview metadata.",
    "shell_exec": "Execute a shell command within the tool root and capture stdout/stderr. Requires allow_unsafe_shell=true.",
    "read_file": "Read UTF-8 file content from a sandboxed path.",
    "write_file": "Write UTF-8 content to a sandboxed file path.",
    "list_dir": "List files/directories under a sandboxed path.",
    "grep_text": "Run regex search against one text file under the tool root.",
}

_BUILTIN_METADATA: dict[str, dict[str, Any]] = {
    "web_fetch": {
        "builtin": True,
        "risk_level": "low",
        "default_enabled": True,
        "requires_confirmation": False,
        "category": "network-read",
    },
    "read_file": {
        "builtin": True,
        "risk_level": "low",
        "default_enabled": True,
        "requires_confirmation": False,
        "category": "filesystem-read",
    },
    "list_dir": {
        "builtin": True,
        "risk_level": "low",
        "default_enabled": True,
        "requires_confirmation": False,
        "category": "filesystem-read",
    },
    "grep_text": {
        "builtin": True,
        "risk_level": "low",
        "default_enabled": True,
        "requires_confirmation": False,
        "category": "filesystem-read",
    },
    "write_file": {
        "builtin": True,
        "risk_level": "high",
        "default_enabled": False,
        "requires_confirmation": True,
        "category": "filesystem-write",
    },
    "shell_exec": {
        "builtin": True,
        "risk_level": "high",
        "default_enabled": False,
        "requires_confirmation": True,
        "category": "system-exec",
    },
}


@dataclass
class ToolEntry:
    name: str
    entrypoint: str
    fn: Callable[..., Any]
    description: str
    metadata: dict[str, Any] = field(default_factory=dict)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolEntry] = {}

    def register(
        self, name: str, entrypoint: str, description: str = "", **metadata: Any
    ) -> ToolEntry:
        fn = _load_entrypoint(entrypoint)
        desc = description.strip() or _extract_doc(fn)
        entry = ToolEntry(
            name=name,
            entrypoint=entrypoint,
            fn=fn,
            description=desc,
            metadata=dict(metadata),
        )
        self._tools[name] = entry
        return entry

    def register_builtin(self, names: list[str]) -> list[str]:
        added: list[str] = []
        for name in names:
            if name not in _BUILTIN_ENTRYPOINTS:
                raise ValueError(f"Unknown built-in tool '{name}'")
            self.register(
                name,
                _BUILTIN_ENTRYPOINTS[name],
                description=_BUILTIN_DESCRIPTIONS.get(name, ""),
                **dict(_BUILTIN_METADATA.get(name, {"builtin": True})),
            )
            added.append(name)
        return added

    def list_catalog(self) -> list[dict[str, Any]]:
        rows_by_name: dict[str, dict[str, Any]] = {}

        for name, entrypoint in sorted(
            _BUILTIN_ENTRYPOINTS.items(), key=lambda kv: kv[0]
        ):
            fn = _load_entrypoint(entrypoint)
            description = _BUILTIN_DESCRIPTIONS.get(name, "") or _extract_doc(fn)
            rows_by_name[name] = {
                "name": name,
                "entrypoint": entrypoint,
                "description": description,
                "metadata": dict(_BUILTIN_METADATA.get(name, {"builtin": True})),
            }

        for name in sorted(self._tools.keys()):
            t = self._tools[name]
            rows_by_name[name] = {
                "name": t.name,
                "entrypoint": t.entrypoint,
                "description": t.description,
                "metadata": t.metadata,
            }

        return [rows_by_name[name] for name in sorted(rows_by_name.keys())]

    def get(self, name: str) -> ToolEntry:
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not found")
        return self._tools[name]

    def call(self, name: str, kwargs: dict[str, Any]) -> dict[str, Any]:
        tool = self.get(name)
        result = tool.fn(**kwargs)
        return {
            "ok": True,
            "tool": name,
            "result": result,
        }


def _load_entrypoint(entrypoint: str) -> Callable[..., Any]:
    if ":" not in entrypoint:
        raise ValueError(
            f"Invalid tool entrypoint '{entrypoint}', expected module:function"
        )
    module_name, func_name = entrypoint.split(":", 1)
    module = importlib.import_module(module_name)
    fn = getattr(module, func_name, None)
    if fn is None or not callable(fn):
        raise ValueError(f"Entrypoint '{entrypoint}' is not callable")
    return fn


def _extract_doc(fn: Callable[..., Any]) -> str:
    raw = (fn.__doc__ or "").strip()
    if not raw:
        return ""
    return raw.splitlines()[0].strip()


_tool_registry = ToolRegistry()


def tool_list_available() -> list[dict[str, Any]]:
    return _tool_registry.list_catalog()


def tool_register(
    name: str, entrypoint: str, kwargs: dict | None = None
) -> dict[str, Any]:
    if kwargs is None:
        kwargs = {}
    description = str(kwargs.pop("description", ""))
    entry = _tool_registry.register(name, entrypoint, description=description, **kwargs)
    return {
        "ok": True,
        "name": entry.name,
        "entrypoint": entry.entrypoint,
        "description": entry.description,
    }


def tool_register_builtin(names: list[str]) -> dict[str, Any]:
    added = _tool_registry.register_builtin(names)
    return {
        "ok": True,
        "added": added,
    }


def tool_call(name: str, kwargs: dict | None = None) -> dict[str, Any]:
    if kwargs is None:
        kwargs = {}
    return _tool_registry.call(name, kwargs)
