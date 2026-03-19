---
name: shell-safety-exec
description: Use when executing shell commands or inspecting local environment. Trigger on requests for command execution, system inspection, or filesystem/CLI diagnostics where safety constraints are required.
license: MIT
compatibility: Requires built-in tool shell_exec and optionally list_dir/read_file.
triggers: command, shell, terminal, exec, run, diagnostics, filesystem
requires_tools: shell_exec
category: safety
priority: 3
max_injection_chars: 900
---

# Shell Safety Execution Skill

## Overview
Execute shell commands carefully, prioritize read-only diagnostics, and report outcomes clearly.

## When to Use
- User asks to run command-line checks.
- User asks to inspect paths, environment, versions, or process outputs.
- Task needs command output embedded in response.

## Workflow
1. Prefer non-destructive commands first.
2. Run command with `shell_exec` and capture `returncode`, `stdout`, and `stderr`.
3. Report command + concise result summary.
4. If command fails, propose the minimal next diagnostic command.

## Safety Rules
- Do not run destructive operations unless explicitly requested.
- Avoid chained commands that can mutate system state unexpectedly.
- Keep command scope narrow (specific cwd/path).

## Output Requirements
- Include command executed.
- Include return code and key output lines.
- State whether operation succeeded.
