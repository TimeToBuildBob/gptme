# mypy: ignore-errors
"""init() calls bare load_dotenv(). python-dotenv's find_dotenv() then searches
upward from the *calling module's file* (site-packages/gptme/), not from the
cwd/workspace. A project .env in the directory where the user runs gptme is
never loaded, while an unrelated .env in an ancestor of the install dir
(e.g. ~/.env for pipx/uv-tool installs, or a venv's parent project) is."""

import os
from unittest.mock import patch

import pytest

import gptme.init as gi


class _Stop(Exception):
    pass


def test_init_loads_dotenv_from_cwd(tmp_path, monkeypatch):
    key = "GPTME_R5_DOTENV_SENTINEL"
    monkeypatch.delenv(key, raising=False)
    (tmp_path / ".env").write_text(f"{key}=from-cwd\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(gi, "_init_done", False)
    try:
        # stop init() right after load_dotenv() runs
        with patch.object(gi, "_init_plugins", side_effect=_Stop):  # noqa: SIM117
            with pytest.raises(_Stop):
                gi.init(None, False, None, "markdown")
        assert os.environ.get(key) == "from-cwd"
    finally:
        os.environ.pop(key, None)
