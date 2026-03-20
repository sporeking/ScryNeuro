from __future__ import annotations

import importlib
import pathlib
import sys


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "python" / "web_ui"))
sys.path.insert(0, str(PROJECT_ROOT / "python"))


def test_new_package_skill_discovery_uses_package_path() -> None:
    runtime = importlib.import_module("scryer_agent.runtime")

    rows = runtime.agent_discover_skills()

    assert rows
    assert any("python/scryer_agent/skills" in str(row.get("path", "")) for row in rows)


def test_legacy_webui_launcher_aliases_package_module() -> None:
    legacy = importlib.import_module("app_gradio")
    package = importlib.import_module("scryer_agent.web_ui.app_gradio")

    assert legacy is package


def test_legacy_tool_and_plugin_shims_alias_package_modules() -> None:
    legacy_tools = importlib.import_module("scryer_agent_tools")
    package_tools = importlib.import_module("scryer_agent.tools")
    legacy_tool_runtime = importlib.import_module("scryer_tool_runtime")
    package_tool_runtime = importlib.import_module("scryer_agent.tool_runtime")
    legacy_plugins = importlib.import_module("scryer_agent_plugins")
    package_plugins = importlib.import_module("scryer_agent.plugins")

    assert legacy_tools is package_tools
    assert legacy_tool_runtime is package_tool_runtime
    assert legacy_plugins is package_plugins


def test_legacy_plugin_entrypoint_resolves_to_canonical_callable() -> None:
    legacy_plugins = importlib.import_module("scryer_agent_plugins")
    package_plugins = importlib.import_module("scryer_agent.plugins")

    assert (
        legacy_plugins.memory_compress_plugin is package_plugins.memory_compress_plugin
    )
