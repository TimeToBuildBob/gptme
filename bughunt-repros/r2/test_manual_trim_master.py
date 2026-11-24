"""Manual /compact trim rewrites conversation.jsonl (the 'master context')
that its own truncation references point into, with no lossless backup."""
import json
import re
from pathlib import Path
from types import SimpleNamespace

from gptme.llm.models import set_default_model
from gptme.logmanager import LogManager
from gptme.message import Message
from gptme.tools.autocompact.handlers import _compact_trim


def _long_assistant() -> str:
    # >1000 tokens of distinct prose sentences, no tool calls, no reasoning.
    return " ".join(
        f"Sentence number {i} explains detail alpha{i} beta{i} gamma{i} delta{i}."
        for i in range(400)
    )


def test_manual_trim_master_context_refs_survive(tmp_path):
    set_default_model("openai/gpt-4")
    original_assistant = _long_assistant()
    history = [
        Message("system", "system prompt"),
        Message("user", "explain it"),
        Message("assistant", original_assistant),
        Message("user", "now run it"),
        Message("assistant", "ok"),
        Message("user", "thanks"),
        Message("assistant", "the last answer"),
    ]
    manager = LogManager(history, logdir=tmp_path / "conv")
    manager.write()
    ctx = SimpleNamespace(manager=manager, args=["trim"])

    out = list(_compact_trim(ctx, manager.log.messages))
    assert "Trim compaction completed" in out[-1].content, out[-1].content

    compacted = manager.log.messages[2].content
    m = re.search(r"Master context: (\S+) \(bytes (\d+)-(\d+)\)", compacted)
    assert m, compacted[-400:]
    path, start, end = Path(m.group(1)), int(m.group(2)), int(m.group(3))
    data = path.read_bytes()[start:end].decode(errors="replace")
    # The reference promises exact recovery of the original message.
    assert json.dumps(original_assistant) in data, f"bytes {start}-{end} hold: {data[:120]!r}"


def test_manual_trim_original_text_survives_on_disk(tmp_path):
    set_default_model("openai/gpt-4")
    original_assistant = _long_assistant()
    history = [
        Message("system", "system prompt"),
        Message("user", "explain it"),
        Message("assistant", original_assistant),
        Message("user", "now run it"),
        Message("assistant", "ok"),
        Message("user", "thanks"),
        Message("assistant", "the last answer"),
    ]
    manager = LogManager(history, logdir=tmp_path / "conv")
    manager.write()
    ctx = SimpleNamespace(manager=manager, args=["trim"])
    out = list(_compact_trim(ctx, manager.log.messages))
    assert "Trim compaction completed" in out[-1].content
    assert manager.log.messages[2].content != original_assistant
    needle = json.dumps(original_assistant)
    files_with_original = [
        str(f.relative_to(manager.logdir))
        for f in manager.logdir.rglob("*")
        if f.is_file() and needle in f.read_text(errors="replace")
    ]
    assert files_with_original, "original assistant message no longer exists anywhere in the logdir"
