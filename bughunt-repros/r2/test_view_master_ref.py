"""Trim on a compacted view stamps master-context byte ranges by *view*
position, but the index is built from conversation.jsonl (master) positions."""
import json
import re
from pathlib import Path

from gptme.llm.models import set_default_model
from gptme.logmanager import LogManager
from gptme.message import Message
from gptme.tools.autocompact.recovery import compact_for_overflow


def test_overflow_trim_on_view_points_at_wrong_master_message(tmp_path):
    set_default_model("openai/gpt-4")
    big = "\n".join(f"line {i}: result value {i * 7919} ok" for i in range(1500))
    main = [
        Message("system", "system prompt"),
        Message("user", "first question"),
        Message("assistant", "first answer"),
        Message("user", "second question"),
        Message("assistant", "```shell\nseq 1500\n```"),
        Message("system", big),
        Message("assistant", "done"),
    ]
    manager = LogManager(main, logdir=tmp_path / "conv")
    manager.write()
    # A compacted view (e.g. after an earlier summarize): early turns replaced
    # by a checkpoint, so view positions no longer match master positions.
    view = [
        main[0],
        Message("assistant", "## Objective\ncheckpoint"),
        main[4],
        main[5],
        main[6],
    ]
    name = manager.get_next_view_name()
    manager.create_view(name, view)
    manager.switch_view(name)

    result = compact_for_overflow(manager)
    stub = next(m.content for m in result if "Master context:" in m.content)
    m = re.search(r"Master context: (\S+) \(bytes (\d+)-(\d+)\)", stub)
    assert m
    line = Path(m.group(1)).read_bytes()[int(m.group(2)) : int(m.group(3))]
    referenced = json.loads(line)["content"]
    assert referenced == big, f"reference points at: {referenced[:60]!r}"


def test_overflow_trim_on_branch_points_into_main_transcript(tmp_path):
    set_default_model("openai/gpt-4")
    big = "\n".join(f"line {i}: result value {i * 7919} ok" for i in range(1500))
    logdir = tmp_path / "conv"
    main = [
        Message("system", "system prompt"),
        Message("user", "main q1"),
        Message("assistant", "main a1"),
        Message("user", "main q2"),
    ]
    LogManager(main, logdir=logdir).write()
    branch_msgs = [
        Message("system", "system prompt"),
        Message("user", "branch question"),
        Message("assistant", "```shell\nseq 1500\n```"),
        Message("system", big),
        Message("assistant", "done"),
    ]
    manager = LogManager(branch_msgs, logdir=logdir, branch="b")
    manager.write()
    assert manager.logfile.name == "b.jsonl"

    result = compact_for_overflow(manager)
    stub = next(m.content for m in result if "Master context:" in m.content)
    m = re.search(r"Master context: (\S+) \(bytes (\d+)-(\d+)\)", stub)
    assert m
    assert Path(m.group(1)) == manager.logfile, f"reference names {m.group(1)}"
