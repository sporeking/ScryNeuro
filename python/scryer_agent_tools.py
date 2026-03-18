from __future__ import annotations

import json
import os
import pathlib
import re
import shlex
import subprocess
from typing import Any

from urllib import error as url_error
from urllib import request as url_request


def add_numbers(a: float, b: float) -> dict:
    return {
        "a": a,
        "b": b,
        "sum": a + b,
    }


def multiply_numbers(a: float, b: float) -> dict:
    return {
        "a": a,
        "b": b,
        "product": a * b,
    }


def reverse_text(text: str) -> dict:
    return {
        "text": text,
        "reversed": text[::-1],
    }


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
) -> dict:
    target_cwd = cwd or os.getcwd()
    proc = subprocess.run(
        command,
        shell=True,
        cwd=target_cwd,
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    return {
        "command": command,
        "cwd": target_cwd,
        "returncode": proc.returncode,
        "stdout": proc.stdout[:max_chars],
        "stderr": proc.stderr[:max_chars],
    }


def tool_read_file(path: str, max_chars: int = 20000) -> dict:
    p = pathlib.Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    data = p.read_text(encoding="utf-8")
    return {
        "path": str(p.resolve()),
        "length": len(data),
        "content": data[:max_chars],
    }


def tool_write_file(path: str, content: str, create_parents: bool = True) -> dict:
    p = pathlib.Path(path)
    if create_parents:
        p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return {
        "path": str(p.resolve()),
        "bytes_written": len(content.encode("utf-8")),
    }


def tool_list_dir(path: str = ".", include_hidden: bool = False) -> dict:
    p = pathlib.Path(path)
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
    }


def tool_grep_text(
    path: str,
    pattern: str,
    max_matches: int = 200,
) -> dict:
    p = pathlib.Path(path)
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
    }
