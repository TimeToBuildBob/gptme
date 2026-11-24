"""estimate_compaction_savings counts Phase 3 (30%) savings for long assistant
messages that contain tool calls, but auto_compact_log never compresses those
(message_contains_tool_use guard). should_auto_compact therefore selects
"rule_based" for a log the engine cannot shrink at all."""

from gptme.message import Message, len_tokens
from gptme.tools import init_tools
from gptme.tools.autocompact.decision import (
    MIN_SAVINGS_RATIO,
    estimate_compaction_savings,
    should_auto_compact,
)
from gptme.tools.autocompact.engine import auto_compact_log
from gptme.llm.models import get_default_model, get_model


def _long_tool_call(i: int) -> str:
    prose = " ".join(
        f"Reasoning step {i}.{j} explains why we run this command now." for j in range(150)
    )
    return f"{prose}\n\n```shell\necho step {i}\n```\n"


def test_rule_based_decision_matches_engine_savings():
    init_tools(["shell"])
    log = [Message("system", "system prompt"), Message("user", "do the task")]
    for i in range(6):
        log.append(Message("assistant", _long_tool_call(i)))
        log.append(Message("system", f"Ran command: echo step {i}\nstep {i}"))
    log.append(Message("user", "continue"))
    log.append(Message("assistant", "ok"))

    model = (get_default_model() or get_model("gpt-4")).model
    total = len_tokens(log, model)
    limit = total - 100  # just over budget

    _, est, _ = estimate_compaction_savings(log, limit, keep_head=2)
    decision = should_auto_compact(log, limit=limit, keep_head=2)
    compacted = list(auto_compact_log(log, limit=limit, keep_head=2))
    actual = total - len_tokens(compacted, model)

    assert decision == "rule_based"  # precondition: estimate says trim is worth it
    assert actual / total >= MIN_SAVINGS_RATIO, (
        f"should_auto_compact chose rule_based on estimated savings {est} "
        f"({est / total:.0%}) but the engine saved {actual} tokens ({actual / total:.0%})"
    )
