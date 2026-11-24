from gptme.llm.models import get_model, set_default_model
from gptme.message import Message, len_tokens
from gptme.util.reduce import limit_log


def test_limit_log_leaves_room_for_output():
    model = get_model("openai/gpt-4")  # 8192-token window
    set_default_model(model)
    msgs = [Message("system", "You are a helpful assistant.")]
    for i in range(40):
        role = "user" if i % 2 == 0 else "assistant"
        msgs.append(Message(role, " ".join(f"word{i}_{j}" for j in range(150))))
    assert len_tokens(msgs, model.model) > 2 * model.context
    kept = limit_log(msgs)
    kept_tokens = len_tokens(kept, model.model)
    print("window", model.context, "kept input tokens", kept_tokens)
    # Request = kept input + output allowance, so the "last-resort guard against
    # the provider window" must reserve room for the completion. Require at least
    # gptme's own budget headroom constant (1000 tokens).
    assert kept_tokens <= model.context - 1000, (
        f"limit_log kept {kept_tokens} input tokens of an {model.context}-token window; "
        f"only {model.context - kept_tokens} tokens left for the completion"
    )
