# mypy: ignore-errors
import os

import pytest


@pytest.fixture(autouse=True)
def isolate(tmp_path_factory, monkeypatch):
    home = tmp_path_factory.mktemp("home")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home / ".config"))
    monkeypatch.setenv("XDG_DATA_HOME", str(home / ".local/share"))
    monkeypatch.setenv("XDG_STATE_HOME", str(home / ".local/state"))
    for k in list(os.environ):
        if k.endswith("_API_KEY") or k in (
            "MODEL",
            "GPTME_MODEL",
            "TOOL_FORMAT",
            "TOOL_ALLOWLIST",
            "GPTME_CONTEXT_LENGTH",
        ):
            monkeypatch.delenv(k, raising=False)
    import gptme.config as config_module
    import gptme.config.user as user_module
    from gptme.config.core import _config_var

    cfg = home / ".config" / "gptme" / "config.toml"
    monkeypatch.setattr(user_module, "config_path", str(cfg))
    monkeypatch.setattr(config_module, "config_path", str(cfg))
    token = _config_var.set(None)
    cwd = os.getcwd()
    try:
        yield
    finally:
        _config_var.reset(token)
        os.chdir(cwd)
