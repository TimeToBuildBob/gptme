"""grok-subscription: tokens loaded from an alternate grok CLI auth key are
refreshed (consuming the CLI's refresh token) but never written back, so
~/.grok/auth.json keeps the stale/consumed refresh token."""

import json
from unittest.mock import MagicMock, patch

from gptme.llm import llm_grok_subscription as grok


def test_refresh_of_alternate_key_entry_is_written_back(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    grok_dir = tmp_path / ".grok"
    grok_dir.mkdir()
    auth_file = grok_dir / "auth.json"
    # Variant issuer spelling (trailing slash) -- explicitly supported by
    # _load_grok_cli_tokens' "any key that embeds our exact client ID" fallback.
    alt_key = f"https://auth.x.ai/::{grok.OAUTH_CLIENT_ID}"
    auth_file.write_text(
        json.dumps(
            {
                alt_key: {
                    "key": "cli-old-access",
                    "refresh_token": "cli-old-refresh",
                    "expires_at": "2020-01-01T00:00:00.000000000Z",
                }
            }
        )
    )
    monkeypatch.setattr(grok, "_get_grok_cli_auth_path", lambda: auth_file)
    monkeypatch.setattr(grok, "_auth", None)

    resp = MagicMock(status_code=200)
    resp.json.return_value = {
        "access_token": "new-access",
        "refresh_token": "rotated-refresh",
        "expires_in": 21600,
    }
    with patch.object(grok.requests, "post", return_value=resp) as post:
        auth = grok.get_auth()

    assert post.call_args.kwargs["data"]["refresh_token"] == "cli-old-refresh"
    assert auth.access_token == "new-access"

    entry = json.loads(auth_file.read_text())[alt_key]
    # The refresh token the grok CLI holds was just spent; it must be replaced.
    assert entry["refresh_token"] == "rotated-refresh"
    assert entry["key"] == "new-access"
