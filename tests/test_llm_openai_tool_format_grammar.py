"""Tests for GPTME_TOOL_FORMAT_GRAMMAR passthrough to structured outputs."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from gptme.llm import llm_openai
from gptme.llm.models import Provider, get_model
from gptme.message import Message

GRAMMAR = 'root ::= "hello"\n'


@pytest.fixture
def grammar_file(tmp_path):
    path = tmp_path / "gptme_markdown.ebnf"
    path.write_text(GRAMMAR)
    return path


@pytest.fixture(autouse=True)
def _clear_grammar_cache():
    llm_openai._load_tool_format_grammar.cache_clear()
    yield
    llm_openai._load_tool_format_grammar.cache_clear()


def _set_tool_format(monkeypatch, fmt):
    from gptme.tools import base

    monkeypatch.setattr(base, "tool_format", fmt, raising=False)


def test_grammar_sent_as_structured_outputs(monkeypatch, grammar_file):
    monkeypatch.setenv("GPTME_TOOL_FORMAT_GRAMMAR", str(grammar_file))
    _set_tool_format(monkeypatch, "markdown")

    body = llm_openai.extra_body("local", get_model("openai/gpt-4o"))

    assert body["structured_outputs"] == {"grammar": GRAMMAR}


def test_grammar_sent_for_xml_format(monkeypatch, grammar_file):
    monkeypatch.setenv("GPTME_TOOL_FORMAT_GRAMMAR", str(grammar_file))
    _set_tool_format(monkeypatch, "xml")

    body = llm_openai.extra_body("local", get_model("openai/gpt-4o"))

    assert body["structured_outputs"] == {"grammar": GRAMMAR}


def test_grammar_skipped_for_native_tool_format(monkeypatch, grammar_file):
    """Native tool calling is already constrained against the tool schemas."""
    monkeypatch.setenv("GPTME_TOOL_FORMAT_GRAMMAR", str(grammar_file))
    _set_tool_format(monkeypatch, "tool")

    body = llm_openai.extra_body("local", get_model("openai/gpt-4o"))

    assert "structured_outputs" not in body


def test_grammar_skipped_for_hosted_providers(monkeypatch, grammar_file):
    """Hosted APIs would 400 on the unknown request field."""
    monkeypatch.setenv("GPTME_TOOL_FORMAT_GRAMMAR", str(grammar_file))
    _set_tool_format(monkeypatch, "markdown")

    providers: list[Provider] = ["openai", "anthropic", "openrouter"]
    for provider in providers:
        body = llm_openai.extra_body(provider, get_model("openai/gpt-4o"))
        assert "structured_outputs" not in body, provider


def test_no_grammar_field_when_unset(monkeypatch):
    monkeypatch.delenv("GPTME_TOOL_FORMAT_GRAMMAR", raising=False)
    _set_tool_format(monkeypatch, "markdown")

    body = llm_openai.extra_body("local", get_model("openai/gpt-4o"))

    assert "structured_outputs" not in body
    assert "guided_grammar" not in body


def test_legacy_field_for_old_vllm(monkeypatch, grammar_file):
    monkeypatch.setenv("GPTME_TOOL_FORMAT_GRAMMAR", str(grammar_file))
    monkeypatch.setenv("GPTME_TOOL_FORMAT_GRAMMAR_LEGACY", "1")
    _set_tool_format(monkeypatch, "markdown")

    body = llm_openai.extra_body("local", get_model("openai/gpt-4o"))

    assert body["guided_grammar"] == GRAMMAR
    assert "structured_outputs" not in body


def test_missing_grammar_file_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("GPTME_TOOL_FORMAT_GRAMMAR", str(tmp_path / "nope.ebnf"))
    _set_tool_format(monkeypatch, "markdown")

    with pytest.raises(FileNotFoundError, match="GPTME_TOOL_FORMAT_GRAMMAR"):
        llm_openai.extra_body("local", get_model("openai/gpt-4o"))


def test_empty_grammar_file_raises(monkeypatch, tmp_path):
    path = tmp_path / "empty.ebnf"
    path.write_text("   \n")
    monkeypatch.setenv("GPTME_TOOL_FORMAT_GRAMMAR", str(path))
    _set_tool_format(monkeypatch, "markdown")

    with pytest.raises(ValueError, match="empty"):
        llm_openai.extra_body("local", get_model("openai/gpt-4o"))


def test_chat_request_carries_the_grammar(monkeypatch, grammar_file):
    """End-to-end through chat(): the field lands in the outgoing request."""
    monkeypatch.setenv("GPTME_TOOL_FORMAT_GRAMMAR", str(grammar_file))
    _set_tool_format(monkeypatch, "markdown")

    completion = SimpleNamespace(
        usage=None,
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content="ok", tool_calls=None),
            )
        ],
    )
    raw_resp = SimpleNamespace(parse=lambda: completion, headers={})
    completions_create = Mock(return_value=raw_resp)
    mock_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                with_raw_response=SimpleNamespace(create=completions_create)
            )
        )
    )
    monkeypatch.setattr(llm_openai, "get_client", lambda provider: mock_client)
    monkeypatch.setattr(llm_openai, "_is_proxy", lambda client: False)
    monkeypatch.setattr(
        llm_openai, "get_provider_from_model", lambda model: "local", raising=False
    )

    llm_openai.chat([Message(role="user", content="hi")], "local/my-model", None)

    assert completions_create.call_args.kwargs["extra_body"]["structured_outputs"] == {
        "grammar": GRAMMAR
    }
