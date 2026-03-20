from __future__ import annotations

import json
import os
import pathlib
import re
import subprocess
from typing import Any

from urllib import error as url_error
from urllib import request as url_request

_MAX_TOOL_TIMEOUT_SECONDS = 120


def _resolve_tool_root(tool_root: str | None = None) -> pathlib.Path:
    root = pathlib.Path(tool_root or os.getcwd()).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Tool root not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Tool root is not a directory: {root}")
    return root


def _resolve_sandboxed_path(
    path: str,
    *,
    tool_root: str | None = None,
    allow_missing: bool = False,
) -> pathlib.Path:
    root = _resolve_tool_root(tool_root)
    candidate = pathlib.Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve(strict=not allow_missing)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise PermissionError(f"Path '{resolved}' escapes tool root '{root}'") from exc
    return resolved


def _clamp_timeout(timeout: int) -> int:
    return max(1, min(int(timeout), _MAX_TOOL_TIMEOUT_SECONDS))


def tool_web_fetch(url: str, timeout: int = 20, max_chars: int = 12000) -> dict:
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    req = url_request.Request(url, headers={"User-Agent": "ScryNeuro-Agent/0.1"})
    try:
        with url_request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            text = raw.decode("utf-8", errors="replace")
            status_code = int(getattr(resp, "status", 200))
            final_url = str(resp.geturl())
            content_type = str(resp.headers.get("Content-Type", ""))
    except url_error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        return {
            "url": url,
            "status_code": int(e.code),
            "content_type": str(e.headers.get("Content-Type", "")),
            "length": len(body),
            "content_preview": body[:max_chars],
            "error": str(e),
        }
    return {
        "url": final_url,
        "status_code": status_code,
        "content_type": content_type,
        "length": len(text),
        "content_preview": text[:max_chars],
    }


def tool_shell_exec(
    command: str,
    cwd: str | None = None,
    timeout: int = 30,
    max_chars: int = 12000,
    allow_unsafe_shell: bool = False,
    tool_root: str | None = None,
) -> dict:
    if not allow_unsafe_shell:
        raise PermissionError(
            "shell_exec is privileged and requires allow_unsafe_shell=true"
        )
    root = _resolve_tool_root(tool_root)
    target_cwd = (
        str(_resolve_sandboxed_path(cwd, tool_root=str(root))) if cwd else str(root)
    )
    proc = subprocess.run(
        command,
        shell=True,
        cwd=target_cwd,
        text=True,
        capture_output=True,
        timeout=_clamp_timeout(timeout),
    )
    return {
        "command": command,
        "cwd": target_cwd,
        "returncode": proc.returncode,
        "stdout": proc.stdout[:max_chars],
        "stderr": proc.stderr[:max_chars],
        "tool_root": str(root),
    }


def tool_read_file(
    path: str,
    max_chars: int = 20000,
    tool_root: str | None = None,
) -> dict:
    p = _resolve_sandboxed_path(path, tool_root=tool_root)
    if not p.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    data = p.read_text(encoding="utf-8")
    return {
        "path": str(p.resolve()),
        "length": len(data),
        "content": data[:max_chars],
        "tool_root": str(_resolve_tool_root(tool_root)),
    }


def tool_write_file(
    path: str,
    content: str,
    create_parents: bool = True,
    tool_root: str | None = None,
) -> dict:
    p = _resolve_sandboxed_path(path, tool_root=tool_root, allow_missing=True)
    if create_parents:
        p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return {
        "path": str(p.resolve()),
        "bytes_written": len(content.encode("utf-8")),
        "tool_root": str(_resolve_tool_root(tool_root)),
    }


def tool_list_dir(
    path: str = ".",
    include_hidden: bool = False,
    tool_root: str | None = None,
) -> dict:
    p = _resolve_sandboxed_path(path, tool_root=tool_root)
    if not p.is_dir():
        raise NotADirectoryError(f"Not a directory: {path}")
    entries = []
    for child in sorted(p.iterdir(), key=lambda x: x.name.lower()):
        if not include_hidden and child.name.startswith("."):
            continue
        entries.append(
            {
                "name": child.name,
                "type": "dir" if child.is_dir() else "file",
            }
        )
    return {
        "path": str(p.resolve()),
        "entries": entries,
        "tool_root": str(_resolve_tool_root(tool_root)),
    }


def tool_grep_text(
    path: str,
    pattern: str,
    max_matches: int = 200,
    tool_root: str | None = None,
) -> dict:
    p = _resolve_sandboxed_path(path, tool_root=tool_root)
    if not p.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    regex = re.compile(pattern)
    matches: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            if regex.search(line):
                matches.append({"line": idx, "text": line.rstrip("\n")})
                if len(matches) >= max_matches:
                    break
    return {
        "path": str(p.resolve()),
        "pattern": pattern,
        "matches": matches,
        "truncated": len(matches) >= max_matches,
        "tool_root": str(_resolve_tool_root(tool_root)),
    }
