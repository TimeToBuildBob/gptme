from gptme.llm.models import get_default_model, get_model
from gptme.message import Message, len_tokens
from gptme.tools.autocompact import should_auto_compact, auto_compact_log
from gptme.tools.autocompact.decision import estimate_compaction_savings
from gptme.util.context_measurement import measure_context_tokens, prefix_digests


def test_provider_measured_over_budget_with_big_stale_tool_output_picks_rule_based():
    model = get_default_model() or get_model("gpt-4")
    big = "\n".join(f"entry_{i} size={i * 13} checksum={i * 7919 % 10007}" for i in range(1500))
    log = [
        Message("system", "You are a helpful assistant."),
        Message("user", "list things"),
        Message("assistant", "Running a listing."),
        Message("system", big, call_id="call_1"),
        Message("user", "now something else"),
        Message("assistant", "Sure, done."),
        Message("user", "and another thing"),
    ]
    anchor_idx = 5
    digests = prefix_digests(log)
    stored = len_tokens(log, model.model)
    # provider-reported input includes tool schemas etc. (~20k overhead)
    log[anchor_idx] = log[anchor_idx].replace(metadata={
        "usage": {"input_tokens": stored + 20_000},
        "input_log_messages": anchor_idx,
        "input_log_digest": digests[anchor_idx],
        "input_log_model": model.full,
    })
    measured = measure_context_tokens(log, model.full)
    assert measured > stored + 19_000
    budget = stored + 5_000  # measured is over budget; stored text is under it
    assert measured >= budget > stored

    _, savings, _ = estimate_compaction_savings(log, limit=budget)
    print("stored", stored, "measured", measured, "budget", budget, "est savings", savings)
    # A ~20k-token stale tool output is trimmable; the trigger fired, so the
    # cheap rule-based path should be viable (and the engine should trim).
    action = should_auto_compact(log, limit=budget)
    out = [m for m in auto_compact_log(log, limit=budget) if isinstance(m, Message)]
    trimmed = len_tokens(out, model.model)
    print("action", action, "engine output tokens", trimmed)
    assert action == "rule_based", f"got {action!r}: estimate saw 0 savings because it re-gated on stored-text tokens"
