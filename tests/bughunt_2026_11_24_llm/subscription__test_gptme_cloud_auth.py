"""gptme cloud provider: token keying and credential/URL pairing defects."""

import json
import time
from unittest.mock import MagicMock, patch

import pytest


def _cfg(env: dict[str, str]):
    class C:
        def get_env(self, key):
            return env.get(key)

        def get_env_required(self, key):
            return env[key]

    return C()


def _resp(status: int, payload: dict):
    r = MagicMock()
    r.status_code = status
    r.json.return_value = payload
    r.raise_for_status.return_value = None
    return r


@pytest.mark.parametrize(
    "base_url",
    [
        # documented chat-completions override for the managed service
        "https://auth.gptme.ai/functions/v1/messages",
        # self-hosted / custom OpenAI-compatible gptme server
        "https://my-server.example.com/v1",
    ],
)
def test_inline_login_token_is_found_by_provider(tmp_path, base_url):
    """`gptme -m gptme/...` -> "Start device-flow login now? [Y/n]" -> success,
    then init.py retries init_llm(), which calls get_api_key(). The freshly
    saved token must be the one get_api_key() returns."""
    from gptme import init as gptme_init
    from gptme.llm import llm_gptme

    config = _cfg({"GPTME_CLOUD_BASE_URL": base_url})
    posts = [
        _resp(
            200,
            {
                "device_code": "dc",
                "user_code": "ABCD-EFGH",
                "verification_uri": "https://gptme.ai/device",
                "interval": 0,
                "expires_in": 60,
            },
        ),
        _resp(200, {"access_token": "fresh-device-token", "expires_in": 3600}),
    ]
    with (
        patch.object(llm_gptme, "_TOKEN_DIR", tmp_path),
        patch("requests.post", side_effect=posts),
        patch("time.sleep"),
        patch.object(gptme_init, "is_output_json", return_value=False),
        patch.object(gptme_init.console, "input", return_value="y"),
    ):
        assert gptme_init._maybe_authenticate_gptme_interactively(
            "gptme", True, config
        )
        # What init_llm("gptme") -> llm_openai.init -> get_api_key does next:
        try:
            key = llm_gptme.get_api_key(config)
        except llm_gptme.GptmeAuthError as e:
            pytest.fail(
                f"login succeeded but provider cannot find the token "
                f"({sorted(p.name for p in tmp_path.iterdir())}): {e}"
            )
    assert key == "fresh-device-token"


def test_models_listing_does_not_send_gptme_token_to_env_host():
    """With a gptme.ai Device Flow token, chat goes to the token's base_url
    (token wins over GPTME_CLOUD_BASE_URL). `--list-models` must target the
    same service: it pairs get_models_url() with the same gptme.ai bearer."""
    from gptme.llm.llm_gptme import (
        DEFAULT_BASE_URL,
        DEFAULT_MODELS_BASE_URL,
        get_api_key,
        get_base_url,
        get_models_url,
    )

    token = {
        "access_token": "gptme-ai-bearer",
        "expires_at": time.time() + 3600,
        "server_url": "https://auth.gptme.ai",
        "base_url": DEFAULT_BASE_URL,
    }
    config = _cfg({"GPTME_CLOUD_BASE_URL": "https://other.example.com/v1"})
    with patch("gptme.llm.llm_gptme._load_token", return_value=token):
        assert get_api_key(config) == "gptme-ai-bearer"
        assert get_base_url(config) == DEFAULT_BASE_URL  # chat: token wins
        # models: must not ship the gptme.ai bearer to other.example.com
        assert get_models_url(config) == DEFAULT_MODELS_BASE_URL
