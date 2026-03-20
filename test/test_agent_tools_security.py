from __future__ import annotations

import importlib
import pathlib
import sys

import pytest


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))


def _agent_tools_module():
    return importlib.import_module("scryer_agent_tools")


def test_read_file_rejects_paths_outside_tool_root(tmp_path: pathlib.Path) -> None:
    agent_tools = _agent_tools_module()
    inside = tmp_path / "inside.txt"
    inside.write_text("ok", encoding="utf-8")
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("nope", encoding="utf-8")

    with pytest.raises(PermissionError):
        agent_tools.tool_read_file(str(outside), tool_root=str(tmp_path))


def test_shell_exec_requires_explicit_unsafe_flag(tmp_path: pathlib.Path) -> None:
    agent_tools = _agent_tools_module()

    with pytest.raises(PermissionError):
        agent_tools.tool_shell_exec("pwd", tool_root=str(tmp_path))


def test_write_file_stays_within_tool_root(tmp_path: pathlib.Path) -> None:
    agent_tools = _agent_tools_module()

    result = agent_tools.tool_write_file(
        "nested/out.txt",
        "hello",
        tool_root=str(tmp_path),
    )

    assert pathlib.Path(result["path"]).is_file()
    assert pathlib.Path(result["path"]).resolve().is_relative_to(tmp_path.resolve())
