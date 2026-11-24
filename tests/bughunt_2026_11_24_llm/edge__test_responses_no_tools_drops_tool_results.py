"""Responses API conversion with tools=[] / None silently drops tool outputs.

Autocompact's resume/chunk summaries call ``llm.reply(prepared_msgs, tools=[])``
on the real log, which still carries call_id tool results. In
_prepare_messages_for_responses_api, ``if tools:`` is falsy for ``[]`` so
_handle_tools never turns the assistant's ``@tool(call_id): {...}`` text into a
function_call item, but _messages_dicts_to_responses_input still turns every
system+call_id message into a function_call_output. _pair_missing_tool_results
then drops each one as an orphan, so the summarizer never sees any tool output.
"""

import pytest

from gptme.llm.llm_openai import _prepare_messages_for_responses_api
from gptme.message import Message

TOOL_OUTPUT = "UNIQUE-TOOL-OUTPUT-42: 3 tests failed in test_foo.py"


def _log():
    return [
        Message(role="system", content="You are gptme."),
        Message(role="user", content="run the tests"),
        Message(
            role="assistant",
            content='Running.\n@shell(call_abc): {"command": "pytest"}',
        ),
        Message(role="system", content=TOOL_OUTPUT, call_id="call_abc"),
        Message(role="user", content="Summarize the conversation so far."),
    ]


@pytest.mark.parametrize("tools", [[], None])
def test_tool_output_reaches_responses_input_without_tools(tools):
    _instructions, items, _tools = _prepare_messages_for_responses_api(
        _log(), "openai/gpt-5", tools
    )
    blob = repr(items)
    # On master the function_call_output is dropped as an orphan, so the
    # tool output appears nowhere in the request input.
    assert TOOL_OUTPUT in blob, items
