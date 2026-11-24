"""Responses API streaming drops refusal content.

Non-streaming ``llm_openai.chat()`` (Responses path) explicitly renders
``refusal`` content parts.  The streaming event processor handles neither
``response.refusal.delta`` nor refusal parts in the terminal output, so a
refusal streams as an empty reply.
"""

from gptme.llm.openai_responses import _stream_responses_events

REFUSAL = "I'm sorry, I can't help with that."

MESSAGE_ITEM = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "status": "completed",
    "content": [{"type": "refusal", "refusal": REFUSAL}],
}

EVENTS = [
    {"type": "response.created", "response": {"model": "gpt-5", "output": []}},
    {
        "type": "response.output_item.added",
        "output_index": 0,
        "item": {**MESSAGE_ITEM, "status": "in_progress", "content": []},
    },
    {
        "type": "response.content_part.added",
        "output_index": 0,
        "content_index": 0,
        "item_id": "msg_1",
        "part": {"type": "refusal", "refusal": ""},
    },
    {
        "type": "response.refusal.delta",
        "output_index": 0,
        "content_index": 0,
        "item_id": "msg_1",
        "delta": REFUSAL,
    },
    {
        "type": "response.refusal.done",
        "output_index": 0,
        "content_index": 0,
        "item_id": "msg_1",
        "refusal": REFUSAL,
    },
    {"type": "response.output_item.done", "output_index": 0, "item": MESSAGE_ITEM},
    {
        "type": "response.completed",
        "response": {"model": "gpt-5", "status": "completed", "output": [MESSAGE_ITEM]},
    },
]


def test_streamed_refusal_is_not_dropped():
    text = "".join(_stream_responses_events(EVENTS))
    assert REFUSAL in text, repr(text)
