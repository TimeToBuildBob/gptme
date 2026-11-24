from unittest.mock import patch

from gptme.message import Message, len_tokens
from gptme.tools.autocompact.recovery import _drop_to_retry_target
from gptme.util.context_measurement import measure_context_tokens, prefix_digests

MODEL = "openai/gpt-4"


def _turn(i):
    return [
        Message("user", " ".join(f"q{i}_{j}" for j in range(300))),
        Message("assistant", " ".join(f"a{i}_{j}" for j in range(300))),
    ]


def test_drop_to_retry_target_reaches_hysteresis_target_with_provider_anchor():
    msgs = [Message("system", "You are helpful.")]
    for i in range(10):
        msgs += _turn(i)
    msgs.append(Message("user", "latest request"))
    anchor = len(msgs) - 2  # last assistant message
    stored = len_tokens(msgs, MODEL)
    digests = prefix_digests(msgs)
    msgs[anchor] = msgs[anchor].replace(metadata={
        "usage": {"input_tokens": stored + 12_000},  # tool schemas / framing overhead
        "input_log_messages": anchor,
        "input_log_digest": digests[anchor],
        "input_log_model": MODEL,
    })
    before = measure_context_tokens(msgs, MODEL)
    with patch("gptme.tools.autocompact.recovery._get_keep_head", return_value=1):
        out = _drop_to_retry_target(msgs, MODEL)
    after_stored = len_tokens(out, MODEL)
    dropped_turns = len(msgs) - len(out)
    print("measured before", before, "stored before", stored, "stored after", after_stored, "messages dropped", dropped_turns)
    # target is 0.7 x the previous request; even ignoring the provider overhead,
    # stored text must shrink by ~30% (3+ of 10 equal turns), not by one turn.
    assert after_stored <= 0.7 * stored, (
        f"dropped {dropped_turns} message(s): stored {stored} -> {after_stored}; "
        f"loop compared provider-anchored target {0.7 * before:.0f} against a stored-text count"
    )
