from __future__ import annotations

import importlib
import pathlib
import sys


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))


def _tool_runtime_module():
    return importlib.import_module("scryer_tool_runtime")


def test_builtin_tool_catalog_excludes_removed_tools() -> None:
    tool_runtime = _tool_runtime_module()
    rows = tool_runtime.tool_list_available()
    names = {str(row.get("name", "")) for row in rows if isinstance(row, dict)}

    assert "add" not in names
    assert "multiply" not in names
    assert "reverse" not in names
    assert {
        "web_fetch",
        "shell_exec",
        "read_file",
        "write_file",
        "list_dir",
        "grep_text",
    }.issubset(names)


def test_builtin_tool_catalog_includes_risk_metadata() -> None:
    tool_runtime = _tool_runtime_module()
    rows = tool_runtime.tool_list_available()
    by_name = {
        str(row.get("name", "")): row
        for row in rows
        if isinstance(row, dict) and row.get("name")
    }

    assert by_name["web_fetch"]["metadata"]["risk_level"] == "low"
    assert by_name["web_fetch"]["metadata"]["default_enabled"] is True
    assert by_name["shell_exec"]["metadata"]["risk_level"] == "high"
    assert by_name["shell_exec"]["metadata"]["default_enabled"] is False
    assert by_name["write_file"]["metadata"]["requires_confirmation"] is True
