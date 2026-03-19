from __future__ import annotations

import json
from pathlib import Path


def test_agent_log_schema() -> None:
    log_path = Path("logs/agent_test_agent_mock_run.jsonl")
    assert log_path.exists(), f"log file not found: {log_path}"

    lines = [
        line
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert lines, "log file is empty"

    required_top = {"schema_version", "run_id", "seq", "ts", "event", "payload"}
    events: list[str] = []
    run_id = None
    selected_lines: list[str] = []
    for line in lines:
        rec = json.loads(line)
        if rec.get("run_id") != "test_agent_mock_run":
            continue
        selected_lines.append(line)

    assert selected_lines, "no events found for run_id=test_agent_mock_run"

    prev_seq = 0
    prev_ts = 0.0
    for line in selected_lines:
        rec = json.loads(line)
        assert required_top.issubset(rec.keys()), f"missing top-level keys in {rec}"
        assert rec["schema_version"] == "1.0"
        assert isinstance(rec["seq"], int)
        assert isinstance(rec["ts"], (int, float))
        if rec["seq"] <= prev_seq:
            assert rec["ts"] >= prev_ts
            prev_seq = 0
        assert rec["seq"] > prev_seq
        prev_seq = rec["seq"]
        prev_ts = float(rec["ts"])
        assert isinstance(rec["payload"], dict)
        events.append(rec["event"])
        if run_id is None:
            run_id = rec["run_id"]
        else:
            assert rec["run_id"] == run_id

    assert "run_start" in events
    assert "run_end" in events
    assert "step_start" in events
    assert "step_end" in events


if __name__ == "__main__":
    test_agent_log_schema()
    print("agent log schema test passed")
