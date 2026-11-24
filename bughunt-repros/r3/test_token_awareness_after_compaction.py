import re
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from gptme.hooks.token_awareness import (
    _last_warning_tokens_var, _message_counts_var, _token_totals_var,
    add_token_usage_warning,
)
from gptme.hooks.types import ToolExecutePostData
from gptme.message import Message, len_tokens


def _msg(i, n):
    return Message("user" if i % 2 == 0 else "assistant", " ".join(f"w{i}_{j}" for j in range(n)))


def test_retained_estimate_tracks_compacted_log(tmp_path):
    for v in (_token_totals_var, _message_counts_var, _last_warning_tokens_var):
        v.set(None)
    model = MagicMock(context=200_000, model="gpt-4")
    ws = tmp_path / "ws"
    with patch("gptme.llm.models.get_default_model", return_value=model):
        big = [_msg(i, 2000) for i in range(10)]
        list(add_token_usage_warning(ToolExecutePostData(log=SimpleNamespace(messages=big), workspace=ws)))
        # compaction replaces the log with a much shorter one, then the session continues
        compacted = [_msg(100, 20), _msg(101, 20)] + [_msg(200 + i, 600) for i in range(15)]
        out = list(add_token_usage_warning(ToolExecutePostData(log=SimpleNamespace(messages=compacted), workspace=ws)))
    actual = len_tokens(compacted, "gpt-4")
    reported = int(re.search(r"estimate: (\d+)/", out[0].content).group(1))
    print("before", len_tokens(big, "gpt-4"), "actual retained", actual, "reported", reported)
    assert abs(reported - actual) < 1000, f"reported {reported} retained tokens, log actually has {actual}"
