"""build_master_context_index counts every physical line, while the log reader
skips malformed lines, so one torn line shifts every later byte reference."""
import json
import re
from pathlib import Path

from gptme.llm.models import set_default_model
from gptme.logmanager import LogManager
from gptme.message import Message
from gptme.tools.autocompact.recovery import compact_for_overflow


def test_torn_line_shifts_master_context_refs(tmp_path):
    set_default_model("openai/gpt-4")
    big = "\n".join(f"line {i}: result value {i * 7919} ok" for i in range(1500))
    msgs = [
        Message("system", "system prompt"),
        Message("user", "first question"),
        Message("assistant", "first answer"),
        Message("user", "second question"),
        Message("assistant", "```shell\nseq 1500\n```"),
        Message("system", big),
        Message("assistant", "done"),
    ]
    logdir = tmp_path / "conv"
    logdir.mkdir()
    lines = [json.dumps(m.to_dict()) for m in msgs]
    # A torn/corrupt line early in the file (which the reader tolerates).
    lines.insert(2, '{"role": "assistant", "content": "trunc')
    (logdir / "conversation.jsonl").write_text("\n".join(lines) + "\n")

    manager = LogManager.load(logdir, lock=False)
    assert [m.content for m in manager.log.messages] == [m.content for m in msgs]

    result = compact_for_overflow(manager)
    stub = next(m.content for m in result if "Master context:" in m.content)
    m = re.search(r"Master context: (\S+) \(bytes (\d+)-(\d+)\)", stub)
    assert m
    line = Path(m.group(1)).read_bytes()[int(m.group(2)) : int(m.group(3))]
    referenced = json.loads(line)["content"]
    assert referenced == big, f"reference points at: {referenced[:60]!r}"
