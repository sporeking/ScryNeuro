from __future__ import annotations

from typing import Any


def memory_compress_plugin(
    state: Any, max_messages: int = 20, keep_tail: int = 8
) -> dict:
    max_messages = int(max_messages)
    keep_tail = int(keep_tail)

    def after_step(s: Any, out: dict) -> dict:
        messages = getattr(s, "messages", [])
        if len(messages) <= max_messages:
            return out

        prefix = messages[:-keep_tail]
        suffix = messages[-keep_tail:]

        user_turns = sum(1 for m in prefix if m.get("role") == "user")
        assistant_turns = sum(1 for m in prefix if m.get("role") == "assistant")

        summary = (
            "[memory-compressed] "
            f"{len(prefix)} messages summarized; "
            f"user_turns={user_turns}, assistant_turns={assistant_turns}."
        )

        s.messages = [
            {
                "role": "system",
                "content": summary,
            }
        ] + suffix

        return out

    return {
        "name": "memory_compress",
        "after_step": after_step,
    }
