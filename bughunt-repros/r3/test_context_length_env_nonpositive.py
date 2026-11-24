import os
from unittest.mock import MagicMock, patch

import pytest

from gptme.llm.models import ModelMeta
from gptme.util.context_budget import get_context_budget


@pytest.mark.parametrize("raw", ["0", "-4096"])
@patch("gptme.init.set_default_model")
@patch("gptme.init.get_model")
@patch("gptme.init.init_llm")
@patch("gptme.init.get_recommended_model")
@patch("gptme.init.is_custom_provider", return_value=False)
@patch("gptme.init.get_config")
@patch("gptme.init.console")
def test_nonpositive_context_length_override_is_rejected(
    mock_console, mock_config_fn, mock_custom, mock_recommend, mock_init_llm,
    mock_get_model, mock_set_default, raw,
):
    from gptme.init import init_model

    mock_config_fn.return_value = MagicMock(
        chat=MagicMock(model=None),
        user=MagicMock(models=MagicMock(default=None)),
        get_env=MagicMock(return_value=None),
    )
    mock_get_model.return_value = ModelMeta(
        provider="anthropic", model="anthropic/claude-sonnet-4-6", context=200000
    )
    with patch.dict(os.environ, {"GPTME_CONTEXT_LENGTH": raw}):
        init_model(model="anthropic/claude-sonnet-4-6")
    saved = mock_set_default.call_args[0][0]
    # downstream: a 0/negative window collapses the compaction budget to 1 token
    print(raw, "-> context", saved.context, "budget", get_context_budget(saved.context))
    assert saved.context > 0, f"GPTME_CONTEXT_LENGTH={raw} accepted as window {saved.context}"
