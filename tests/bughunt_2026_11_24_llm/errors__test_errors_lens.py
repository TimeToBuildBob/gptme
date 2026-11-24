"""Error-handling lens: RED tests for confirmed defects on gptme master 9f7e679a3."""

from unittest.mock import Mock

import httpx
import pytest

from gptme import llm
from gptme.hooks import clear_hooks
from gptme.llm import recovery
from gptme.message import Message


@pytest.fixture
def offline(monkeypatch):
    clear_hooks()
    monkeypatch.setattr(llm, "init_llm", lambda *a: None)
    monkeypatch.setattr(llm, "is_output_quiet", lambda: True)
    monkeypatch.setattr(
        llm, "get_model", lambda *a: Mock(supports_parallel_tool_calls=False)
    )
    monkeypatch.delenv("GPTME_TEST_MAX_RETRIES", raising=False)
    monkeypatch.setattr(recovery, "get_max_retries", lambda: 11)
    waits: list[float] = []
    monkeypatch.setattr(
        recovery, "backoff_wait", lambda delay, *a: waits.append(delay) or False
    )
    yield waits
    clear_hooks()


# ---------------------------------------------------------------------------
# 1. OpenAI 429 insufficient_quota (permanent: billing) is retried ~5 min
# ---------------------------------------------------------------------------

QUOTA_BODY = {
    "message": "You exceeded your current quota, please check your plan and "
    "billing details.",
    "type": "insufficient_quota",
    "param": None,
    "code": "insufficient_quota",
}


def _quota_error():
    from openai import RateLimitError

    req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    resp = httpx.Response(429, request=req, json={"error": QUOTA_BODY})
    # The SDK passes the inner ``error`` object as ``body``.
    return RateLimitError(QUOTA_BODY["message"], response=resp, body=QUOTA_BODY)


def test_reply_does_not_replay_openai_insufficient_quota(monkeypatch, offline):
    calls = []

    def fail(*a, **k):
        calls.append(1)
        raise _quota_error()

    monkeypatch.setattr(llm, "_chat_complete", fail)
    from openai import RateLimitError

    with pytest.raises(RateLimitError):
        llm.reply([Message("user", "hi")], "openai/gpt-4o", stream=False)
    # Exhausted billing quota never recovers by waiting; one send only.
    assert len(calls) == 1, f"retried {len(calls)} times, waits={offline}"


def test_openai_adapter_does_not_retry_insufficient_quota(monkeypatch):
    from gptme.llm import llm_openai

    monkeypatch.delenv("GPTME_TEST_MAX_RETRIES", raising=False)
    waits: list[float] = []
    monkeypatch.setattr(
        llm_openai, "backoff_wait", lambda delay, *a: waits.append(delay) or False
    )
    err = _quota_error()
    # attempt 0 of 11: a permanent error must be re-raised, not backed off.
    with pytest.raises(type(err)):
        llm_openai._handle_openai_transient_error(err, 0, 11, 1.0)
    assert waits == []


# ---------------------------------------------------------------------------
# 2. Anthropic native context overflow is not classified as context-length
# ---------------------------------------------------------------------------


def test_anthropic_prompt_too_long_is_context_length_error():
    from anthropic import BadRequestError

    body = {
        "type": "error",
        "error": {
            "type": "invalid_request_error",
            "message": "prompt is too long: 208310 tokens > 200000 maximum",
        },
    }
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    resp = httpx.Response(400, request=req, json=body)
    err = BadRequestError(
        f"Error code: 400 - {body}", response=resp, body=body
    )
    llm.mark_llm_reply_origin(err)
    # Without this, autocompact overflow recovery never runs for Anthropic.
    assert llm.is_context_length_error(err)


# ---------------------------------------------------------------------------
# 3. Clamped Anthropic thinking budget drops below the API minimum (1024)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("max_tokens", [300, 800, 1200])
def test_clamped_thinking_budget_respects_anthropic_minimum(max_tokens):
    from gptme.llm.llm_anthropic import _adjust_thinking_budget, _build_thinking_param

    model = "claude-sonnet-4-5"  # legacy budget_tokens thinking model
    budget, use_thinking = _adjust_thinking_budget(
        max_tokens, 16000, True, model=model
    )
    param = _build_thinking_param(model, use_thinking, budget)
    # Anthropic rejects budget_tokens < 1024 with HTTP 400; either disable
    # thinking or keep a valid budget.
    if param is not None and param.get("type") == "enabled":
        assert param["budget_tokens"] >= 1024, param
        assert param["budget_tokens"] < max_tokens, param
