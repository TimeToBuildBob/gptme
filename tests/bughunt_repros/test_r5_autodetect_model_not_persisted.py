# mypy: ignore-errors
"""When no model is configured, the model init_model() auto-detects is never
written to the conversation's config.toml (chat.model stays unset). Each resume
re-runs provider auto-detection, so adding a second API key later silently
switches an existing conversation to a different provider/model."""

from unittest.mock import patch

from gptme.config import ChatConfig, setup_config_from_cli
from gptme.init import init_model
from gptme.llm.models import get_default_model, set_default_model


def test_autodetected_model_is_stable_across_resume(tmp_path, monkeypatch):
    prev = get_default_model()
    ws = tmp_path / "ws"
    ws.mkdir()
    logdir = tmp_path / "logs" / "conv"
    try:
        # Day 1: only an Anthropic key. `gptme --name conv "hi"`
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-not-real")
        cfg = setup_config_from_cli(workspace=ws, logdir=logdir)
        with patch("gptme.init.init_llm"):
            init_model(cfg.chat.model)
        day1 = get_default_model()
        assert day1 is not None and day1.provider == "anthropic"

        # Day 2: user also adds an OpenAI key, then `gptme --resume --name conv`
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
        cfg = setup_config_from_cli(workspace=ws, logdir=logdir)
        with patch("gptme.init.init_llm"):
            init_model(cfg.chat.model)
        day2 = get_default_model()
        assert day2 is not None
        assert day2.full == day1.full, (
            f"resumed conversation switched model {day1.full} -> {day2.full}; "
            f"persisted chat.model={ChatConfig.from_logdir(logdir).model!r}"
        )
    finally:
        set_default_model(prev) if prev else None
