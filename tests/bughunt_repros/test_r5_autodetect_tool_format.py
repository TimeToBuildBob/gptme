# mypy: ignore-errors
"""setup_config_from_cli() picks (and persists) tool_format from the *resolved*
model before init_model() has auto-detected the provider. With only an API key
configured (the common first-run setup), resolved_model is None, so the format
falls back to "markdown" even though the model init_model() then auto-selects
declares default_tool_format="tool". `-m openai` gets "tool"; auto-detection of
the very same model gets "markdown"."""

from gptme.config import setup_config_from_cli
from gptme.llm import guess_provider_from_config
from gptme.llm.models import get_model, get_recommended_model


def test_autodetected_model_gets_its_default_tool_format(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    ws = tmp_path / "ws"
    ws.mkdir()

    explicit = setup_config_from_cli(
        workspace=ws, logdir=tmp_path / "logs" / "explicit", model="openai"
    )
    assert explicit.chat is not None
    assert explicit.chat.tool_format == "tool"  # control

    auto = setup_config_from_cli(workspace=ws, logdir=tmp_path / "logs" / "auto")
    assert auto.chat is not None
    assert auto.chat.model is None  # model left for init_model() to auto-detect

    provider = guess_provider_from_config()
    assert provider == "openai"
    selected = get_model(f"{provider}/{get_recommended_model(provider)}")
    assert selected.default_tool_format == "tool"
    assert auto.chat.tool_format == selected.default_tool_format
