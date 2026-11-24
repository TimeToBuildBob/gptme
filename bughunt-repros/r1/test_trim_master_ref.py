"""/compact trim rewrites conversation.jsonl with the compacted log, so the
master-context byte ranges it just embedded point into the rewritten file
and no longer recover the original content."""

import re
from pathlib import Path
from unittest.mock import MagicMock

from gptme.logmanager import LogManager
from gptme.message import Message
from gptme.tools.autocompact.handlers import cmd_compact_handler
from gptme.util.master_context import MessageByteRange, recover_from_master_context

REF_RE = re.compile(r"Master context: (\S+) \(bytes (\d+)-(\d+)\)")


def _long_assistant() -> str:
    # >1000 tokens, many sentences, no tool calls -> Phase 3 compression
    return " ".join(
        f"Observation number {i} about the design is noted here." for i in range(400)
    )


def test_trim_master_context_reference_recovers_original(tmp_path):
    original_assistant = _long_assistant()
    big_tool_output = "\n".join(f"line {i} of output data" for i in range(4000))
    history = [
        Message("system", "system prompt"),
        Message("user", "first task"),
        Message("assistant", original_assistant),
        Message("user", "run it"),
        Message("assistant", "ok"),
        Message("system", big_tool_output),
        Message("user", "thanks"),
        Message("assistant", "done"),
    ]
    manager = LogManager(
        [*history, Message("user", "/compact trim")],
        logdir=tmp_path / "conv",
    )
    manager.write()
    ctx = MagicMock()
    ctx.manager = manager
    ctx.args = ["trim"]

    results = list(cmd_compact_handler(ctx))
    assert any("Trim compaction completed" in r.content for r in results), [
        r.content[:200] for r in results
    ]

    originals = {m.timestamp: m.content for m in history}
    checked = 0
    for msg in ctx.manager.log.messages:
        m = REF_RE.search(msg.content)
        if not m:
            continue
        checked += 1
        logfile, start, end = Path(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            recovered = recover_from_master_context(
                logfile,
                MessageByteRange(message_idx=-1, byte_start=start, byte_end=end),
            )
        except ValueError as e:
            recovered = f"<unrecoverable: {e}>"
        assert recovered == originals[msg.timestamp], (
            f"reference in {msg.role} message recovered "
            f"{recovered[:80]!r} instead of the original {originals[msg.timestamp][:80]!r}"
        )
    assert checked > 0, "expected at least one master-context reference"


def test_control_reference_valid_before_rewrite(tmp_path):
    """Control: the same engine reference IS valid until handlers rewrites the file."""
    from gptme.tools.autocompact.context_provider import (
        CompressionConfig,
        get_context_provider,
    )

    history = [
        Message("system", "system prompt"),
        Message("user", "first task"),
        Message("assistant", _long_assistant()),
        Message("user", "run it"),
        Message("assistant", "ok"),
        Message("system", "\n".join(f"line {i} of output data" for i in range(4000))),
        Message("user", "thanks"),
        Message("assistant", "done"),
    ]
    manager = LogManager(list(history), logdir=tmp_path / "conv")
    manager.write()
    out = get_context_provider("default").compress(
        history, CompressionConfig(logdir=manager.logdir, keep_head=2)
    ).messages
    originals = {m.timestamp: m.content for m in history}
    n = 0
    for msg in out:
        m = REF_RE.search(msg.content)
        if m:
            n += 1
            rec = recover_from_master_context(
                Path(m.group(1)),
                MessageByteRange(-1, int(m.group(2)), int(m.group(3))),
            )
            assert rec == originals[msg.timestamp]
    assert n > 0
