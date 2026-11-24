"""Subscription OAuth token files: world-readable creation window and
non-atomic overwrite that destroys the previous login on a failed write.

Compare gptme/llm/llm_gptme.py::_save_token (mkstemp 0600 + os.replace),
which was hardened for exactly this in #4791.
"""

import json
import os
import stat
from pathlib import Path

import pytest

from gptme.llm import llm_grok_subscription as grok
from gptme.llm import llm_openai_subscription as oai


def _oai_auth(tag: str):
    return oai.SubscriptionAuth(
        access_token=f"access-{tag}",
        refresh_token=f"refresh-{tag}",
        account_id="acct",
        expires_at=9_999_999_999.0,
    )


def _grok_auth(tag: str):
    return grok.SubscriptionAuth(
        access_token=f"access-{tag}",
        refresh_token=f"refresh-{tag}",
        expires_at=9_999_999_999.0,
    )


CASES = [
    pytest.param(oai, _oai_auth, "openai_subscription.json", id="openai"),
    pytest.param(grok, _grok_auth, "grok_subscription.json", id="grok"),
]


@pytest.mark.parametrize(("mod", "make", "fname"), CASES)
def test_token_file_never_exists_with_group_or_world_bits(
    tmp_path, monkeypatch, mod, make, fname
):
    """The secret must never be on disk with mode wider than 0600."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    old_umask = os.umask(0o022)  # common default umask
    observed: list[int] = []
    real_write_text = Path.write_text

    def spy_write_text(self, data, *a, **kw):
        n = real_write_text(self, data, *a, **kw)
        if self.name == fname:
            # Bytes of the refresh token are on disk right now.
            observed.append(stat.S_IMODE(self.stat().st_mode))
        return n

    monkeypatch.setattr(Path, "write_text", spy_write_text)
    try:
        mod._save_tokens(make("new"))
    finally:
        os.umask(old_umask)

    path = tmp_path / "gptme" / "oauth" / fname
    assert "refresh-new" in path.read_text()
    assert observed, "token file was not written via write_text"
    assert observed[0] & 0o077 == 0, (
        f"token file held the secret with mode {oct(observed[0])} before chmod"
    )


@pytest.mark.parametrize(("mod", "make", "fname"), CASES)
def test_failed_write_keeps_previous_login(tmp_path, monkeypatch, mod, make, fname):
    """A write failure (ENOSPC, quota, crash) must not truncate the existing login."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    mod._save_tokens(make("old"))
    path = tmp_path / "gptme" / "oauth" / fname
    assert json.loads(path.read_text())["refresh_token"] == "refresh-old"

    real_open = open

    class _FullDisk:
        """File object whose write fails after open() already truncated it."""

        def __init__(self, f):
            self._f = f

        def write(self, _data):
            raise OSError(28, "No space left on device")

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self._f.close()
            return False

        def __getattr__(self, name):
            return getattr(self._f, name)

    def failing_open(file, mode="r", *a, **kw):
        f = real_open(file, mode, *a, **kw)
        if str(file).endswith(fname) and "w" in mode:
            return _FullDisk(f)
        return f

    monkeypatch.setattr(Path, "open", lambda self, mode="r", *a, **kw: failing_open(self, mode, *a, **kw))
    import builtins

    monkeypatch.setattr(builtins, "open", failing_open)
    import io

    monkeypatch.setattr(io, "open", failing_open)

    with pytest.raises(OSError):
        mod._save_tokens(make("new"))

    monkeypatch.undo()
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    # Previous usable login must survive a failed save.
    text = path.read_text()
    assert text, "existing token file was truncated to empty by a failed save"
    assert json.loads(text)["refresh_token"] == "refresh-old"
