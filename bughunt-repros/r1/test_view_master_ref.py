"""On a compacted view whose messages are not 1:1 with conversation.jsonl
(e.g. created by summarize), a later rule-based trim embeds master-context
byte ranges by *view* index into conversation.jsonl (the full main log),
so the reference points at an unrelated message."""

import re
from pathlib import Path

from gptme.logmanager import LogManager
from gptme.message import Message
from gptme.tools.autocompact.context_provider import (
    CompressionConfig,
    get_context_provider,
)
from gptme.util.master_context import MessageByteRange, recover_from_master_context

REF_RE = re.compile(r"Master context: (\S+) \(bytes (\d+)-(\d+)\)")


def test_trim_on_summarized_view_reference_points_at_same_message(tmp_path):
    main_history = [Message("system", "system prompt"), Message("user", "task")]
    for i in range(10):
        main_history += [
            Message("assistant", f"step {i}"),
            Message("system", f"short result {i}"),
        ]
    manager = LogManager(list(main_history), logdir=tmp_path / "conv")
    manager.write()

    # A summarize-style view: head + checkpoint summary + recent tail.
    view_msgs = [
        main_history[0],
        main_history[1],
        Message("assistant", "checkpoint summary of steps 0-8"),
        *main_history[-2:],
    ]
    manager.create_view("compacted-001", view_msgs)
    manager.switch_view("compacted-001")

    # Conversation continues on the view (dual-written to main).
    big = "\n".join(f"row {i} of a massive listing" for i in range(4000))
    manager.append(Message("assistant", "listing files"))
    manager.append(Message("system", big))
    manager.append(Message("user", "next"))

    messages = manager.log.messages
    out = get_context_provider("default").compress(
        messages,
        CompressionConfig(limit=3000, logdir=manager.logdir, keep_head=2),
    ).messages

    originals = {m.timestamp: m.content for m in messages}
    n = 0
    for msg in out:
        m = REF_RE.search(msg.content)
        if not m:
            continue
        n += 1
        recovered = recover_from_master_context(
            Path(m.group(1)), MessageByteRange(-1, int(m.group(2)), int(m.group(3)))
        )
        assert recovered == originals[msg.timestamp], (
            f"master reference recovered {recovered[:60]!r}, "
            f"expected truncated content {originals[msg.timestamp][:60]!r}"
        )
    assert n > 0, "expected a master-context reference on the truncated result"
