"""Literal <think>/<thinking> text that is NOT gptme's own thinking wrapper must
survive conversion to the Anthropic payload.

gptme stores Anthropic thinking as a leading ``<think>...<!-- think-sig -->
</think>`` block. _extract_thinking_content() regex-strips *every*
``<think>.*?</think>`` / ``<thinking>.*?</thinking>`` span anywhere in the
assistant text, before tool calls are parsed. So literal tags inside a tool
call's JSON arguments, or in prose, are silently deleted from history.
"""

from gptme.llm.llm_anthropic import _prepare_messages_for_api
from gptme.message import Message
from gptme.tools import get_tool, init_tools


def _base():
    return [
        Message(role="system", content="sys"),
        Message(role="user", content="print a think tag"),
    ]


def test_literal_think_tag_in_tool_args_preserved():
    init_tools(allowlist=["shell"])
    shell = get_tool("shell")
    assert shell
    cmd = "printf '<think>keep</think>'"
    msgs = _base() + [
        Message(
            role="assistant",
            content='Running it.\n@shell(call_1): {"command": "printf \'<think>keep</think>\'"}',
        ),
        Message(role="system", content="<think>keep</think>", call_id="call_1"),
    ]
    dicts, _, _ = _prepare_messages_for_api(msgs, [shell])
    assistant = next(m for m in dicts if m["role"] == "assistant")
    tool_uses = [p for p in assistant["content"] if p["type"] == "tool_use"]
    assert len(tool_uses) == 1
    # On master the tags (and their contents) are stripped: "printf ''"
    assert tool_uses[0]["input"] == {"command": cmd}


def test_literal_thinking_tag_in_prose_preserved():
    init_tools(allowlist=["shell"])
    shell = get_tool("shell")
    assert shell
    prose = (
        "Ask Claude to reason first, e.g. wrap it as "
        "<thinking>step by step</thinking> before the answer."
    )
    msgs = _base() + [
        Message(
            role="assistant",
            content=prose + '\n@shell(call_2): {"command": "ls"}',
        ),
        Message(role="system", content="ok", call_id="call_2"),
    ]
    dicts, _, _ = _prepare_messages_for_api(msgs, [shell])
    assistant = next(m for m in dicts if m["role"] == "assistant")
    text = "".join(p.get("text", "") for p in assistant["content"])
    # On master the example span is deleted from the replayed history.
    assert "<thinking>step by step</thinking>" in text


def test_think_span_across_tool_call_does_not_orphan_tool_result():
    """A '<think>' in prose plus a '</think>' in the tool args makes the regex
    span swallow the whole '@save(call_1): ...' line. The tool_use vanishes but
    its tool_result is still sent -> Anthropic 400 'unexpected tool_use_id'
    on this and every later request of the conversation."""
    init_tools(allowlist=["save"])
    save = get_tool("save")
    assert save
    msgs = _base() + [
        Message(
            role="assistant",
            content=(
                "The page has a dangling <think> tag, I'll close it.\n"
                '@save(call_1): {"path": "x.html", "content": "</think>"}'
            ),
        ),
        Message(role="system", content="Saved to x.html", call_id="call_1"),
        Message(role="user", content="thanks"),
    ]
    dicts, _, _ = _prepare_messages_for_api(msgs, [save])
    use_ids = {
        p["id"]
        for m in dicts
        if isinstance(m["content"], list)
        for p in m["content"]
        if p.get("type") == "tool_use"
    }
    result_ids = {
        p["tool_use_id"]
        for m in dicts
        if isinstance(m["content"], list)
        for p in m["content"]
        if p.get("type") == "tool_result"
    }
    assert result_ids <= use_ids, (result_ids, use_ids)
