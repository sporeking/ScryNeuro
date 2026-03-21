from __future__ import annotations

import importlib
import pathlib
import sys


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "python"))


def test_new_package_skill_discovery_uses_package_path() -> None:
    runtime = importlib.import_module("scryer_agent.runtime")

    rows = runtime.agent_discover_skills()

    assert rows
    assert any("python/scryer_agent/skills" in str(row.get("path", "")) for row in rows)


def test_package_webui_module_imports_cleanly() -> None:
    package = importlib.import_module("scryer_agent.web_ui.app_gradio")

    assert hasattr(package, "build_demo")
    assert hasattr(package, "launch")
