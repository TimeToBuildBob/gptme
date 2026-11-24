from gptme.message import Message
from gptme.tools.autocompact.scoring import score_tool_output_relevance


def test_referenced_path_with_trailing_period_is_kept():
    body = "\n".join(f"row {i}: value={i * 3}" for i in range(400))
    out = Message("system", f"{body}\nWrote full report to /tmp/build/report.json.", call_id="c1")
    log = [
        Message("user", "build it"),
        Message("assistant", "running"),
        out,
        Message("assistant", "Now let me open /tmp/build/report.json to inspect it"),
        Message("user", "ok"),
        Message("assistant", "reading"),
        Message("user", "go"),
    ]
    score = score_tool_output_relevance(out, 2, log)
    assert score >= 1.0, f"still-referenced output scored {score} (will be stubbed)"


def test_referenced_path_with_trailing_period_not_stubbed_by_phase0():
    from gptme.tools.autocompact.engine import prune_stale_tool_outputs

    body = "\n".join(f"row {i}: value={i * 3}" for i in range(400))
    out = Message("system", f"{body}\nWrote full report to /tmp/build/report.json.", call_id="c1")
    log = [
        Message("user", "build it"),
        Message("assistant", "running"),
        out,
        Message("assistant", "Now let me open /tmp/build/report.json to inspect it"),
        Message("user", "ok"),
        Message("assistant", "reading"),
        Message("user", "go"),
    ]
    pruned, saved = prune_stale_tool_outputs(log, "gpt-4")
    assert pruned[2].content == out.content, f"still-referenced output stubbed (saved {saved} tokens)"
