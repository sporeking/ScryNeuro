from __future__ import annotations

import importlib
import pathlib
import sys


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "python" / "web_ui"))
sys.path.insert(0, str(PROJECT_ROOT / "python"))


def _app_gradio_module():
    return importlib.import_module("scryer_agent.web_ui.app_gradio")


def test_default_enabled_tools_are_low_risk_only() -> None:
    app_gradio = _app_gradio_module()

    rows = [
        {"name": "web_fetch", "metadata": {"default_enabled": True}},
        {"name": "read_file", "metadata": {"default_enabled": True}},
        {"name": "list_dir", "metadata": {"default_enabled": True}},
        {"name": "grep_text", "metadata": {"default_enabled": True}},
        {"name": "write_file", "metadata": {"default_enabled": False}},
        {"name": "shell_exec", "metadata": {"default_enabled": False}},
    ]

    assert app_gradio._default_enabled_tool_names(rows) == [
        "web_fetch",
        "read_file",
        "list_dir",
        "grep_text",
    ]


def test_default_tool_info_mentions_privileged_tools() -> None:
    app_gradio = _app_gradio_module()

    assert "shell_exec" in app_gradio._DEFAULT_TOOL_INFO
    assert "write_file" in app_gradio._DEFAULT_TOOL_INFO
