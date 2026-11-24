"""A tool result that merely *starts with* the result-catalog prefix is dropped.

_build_dropped_result_stubs() emits the catalog as a plain system message with
no metadata, so _is_result_stubs_message() identifies catalogs by content
prefix alone. Any tool output (file read, web page, command output) beginning
with "Tool results dropped from the active context remain in the lossless
master log." is classified as a stale catalog and removed from the recent
tail of the compacted view, leaving its assistant tool call orphaned.
"""

from gptme.logmanager import LogManager
from gptme.message import Message
from gptme.tools.autocompact.resume import _RESULT_STUBS_PREFIX, _resume_via_llm


def test_tool_output_mimicking_catalog_prefix_survives_compaction(tmp_path):
    from gptme.llm.models import set_default_model

    set_default_model("openai/gpt-4o")
    tool_output = f"{_RESULT_STUBS_PREFIX} <- first line of README.md\nimportant data"
    messages = [
        Message("system", "system prompt"),
        Message("user", "read README.md and summarize"),
        Message("assistant", "@read(call-1): {}", call_id=None),
        Message("system", tool_output, call_id="call-1"),
        Message("assistant", "Here is the summary."),
        Message("user", "thanks, continue"),
    ]
    manager = LogManager(messages, logdir=tmp_path / "conv", lock=False)
    manager.write()
    checkpoint = Message("assistant", "## Objective\nSummarize README.")
    applied, _ = None, None
    gen = _resume_via_llm(
        manager,
        messages,
        use_view_branch=True,
        keep_recent_tokens=100_000,  # whole conversation fits in the tail
        checkpoint_response=checkpoint,
    )
    try:
        while True:
            next(gen)
    except StopIteration as stop:
        applied, _ = stop.value
    assert applied
    view_contents = [m.content for m in manager.log.messages]
    # Sanity: the other recent messages were kept verbatim.
    assert "Here is the summary." in view_contents
    assert "thanks, continue" in view_contents
    assert tool_output in view_contents, view_contents
