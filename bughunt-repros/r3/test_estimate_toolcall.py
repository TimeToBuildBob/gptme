from gptme.message import Message, len_tokens
from gptme.llm.models import get_default_model, get_model
from gptme.tools.autocompact import auto_compact_log, should_auto_compact
from gptme.tools.autocompact.decision import estimate_compaction_savings
from gptme.util.reduce import message_contains_tool_use
from gptme.tools import init_tools


def _big_save_call():
    body = "\n".join(f"def func_{i}(x):\n    return x * {i} + {i * 7} - 'value_{i}'" for i in range(600))
    return Message("assistant", f"Writing the file.\n\n```save src/big_module.py\n{body}\n```")


def test_phase3_estimate_excludes_tool_call_messages():
    init_tools(["save", "shell"], include_mcp=False)
    model = get_default_model() or get_model("gpt-4")
    big = _big_save_call()
    assert message_contains_tool_use(big)  # engine Phase 3 will skip it
    log = [
        Message("user", "write a big module"),
        big,
        Message("system", "Saved to src/big_module.py"),
        Message("user", "thanks"),
        Message("assistant", "done"),
        Message("user", "next"),
    ]
    total = len_tokens(log, model.model)
    limit = total - 10  # just over budget
    _, est, _ = estimate_compaction_savings(log, limit=limit)
    out = [m for m in auto_compact_log(log, limit=limit) if isinstance(m, Message)]
    # engine leaves the tool-call message verbatim
    assert out[1].content == big.content
    # estimate must match what engine can actually realize
    assert est == 0, f"estimate claims {est} savings from an uncompressible tool call"
    assert should_auto_compact(log, limit=limit) == "summarize"
