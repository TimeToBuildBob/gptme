"""Config writes are atomic, private for secrets, and safe under concurrency.

Regression tests for gptme/gptme#5128.
"""

import multiprocessing
import os
import stat
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import tomlkit

from gptme.config import user as user_mod
from gptme.config.models import ProviderConfig
from gptme.config.user import (
    load_user_config,
    save_provider_config,
    set_config_value,
)

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission bits")


@pytest.fixture
def config_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "gptme" / "config.toml"
    path.parent.mkdir()
    path.write_text('[env]\nEXISTING = "keep"\n', encoding="utf-8")
    monkeypatch.setattr(user_mod, "config_path", str(path))
    return path


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


@posix_only
def test_new_local_config_is_owner_only(config_file: Path) -> None:
    old_umask = os.umask(0o022)
    try:
        set_config_value("env.OPENROUTER_API_KEY", "sk-or-x", reload=False, local=True)
    finally:
        os.umask(old_umask)

    local = config_file.with_name("config.local.toml")
    assert _mode(local) == 0o600
    assert "sk-or-x" in local.read_text(encoding="utf-8")


@posix_only
def test_existing_local_config_is_tightened(config_file: Path) -> None:
    local = config_file.with_name("config.local.toml")
    local.write_text('[env]\nOLD = "1"\n', encoding="utf-8")
    local.chmod(0o644)

    save_provider_config(
        ProviderConfig(name="p", base_url="https://x.invalid", api_key="secret"),
        reload=False,
        local=True,
    )

    assert _mode(local) == 0o600
    doc = tomlkit.parse(local.read_text(encoding="utf-8"))
    assert doc["env"]["OLD"] == "1"
    assert doc["providers"][0]["api_key"] == "secret"


@posix_only
def test_main_config_keeps_its_mode(config_file: Path) -> None:
    config_file.chmod(0o640)
    set_config_value("models.default", "openai/gpt-x", reload=False)
    assert _mode(config_file) == 0o640


def test_failed_write_leaves_original_intact(
    config_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local = config_file.with_name("config.local.toml")
    original = '[env]\nANTHROPIC_API_KEY = "sk-ant-keep"\n'
    local.write_text(original, encoding="utf-8")

    def fail(_fd: int) -> None:
        raise OSError(27, "File too large")

    monkeypatch.setattr(user_mod.os, "fsync", fail)
    with pytest.raises(OSError, match="File too large"):
        set_config_value("env.OTHER", "x" * 10_000, reload=False, local=True)

    assert local.read_text(encoding="utf-8") == original
    # No staging file is left behind next to the config.
    leftovers = sorted(
        p.name for p in config_file.parent.iterdir() if p.name.endswith(".tmp")
    )
    assert leftovers == []


def test_write_through_symlink_updates_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dotfiles = tmp_path / "dotfiles"
    dotfiles.mkdir()
    target = dotfiles / "config.toml"
    target.write_text("[env]\n", encoding="utf-8")
    link_dir = tmp_path / "home"
    link_dir.mkdir()
    link = link_dir / "config.toml"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlinks unavailable")
    monkeypatch.setattr(user_mod, "config_path", str(link))

    set_config_value("models.default", "anthropic/claude-x", reload=False)

    assert link.is_symlink()
    assert tomlkit.parse(target.read_text())["models"]["default"] == (
        "anthropic/claude-x"
    )


def _set_key(path: str, i: int) -> None:
    user_mod.config_path = path
    set_config_value(f"env.KEY_{i}", str(i), reload=False, local=True)


@pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(), reason="needs fork"
)
def test_concurrent_writers_do_not_lose_updates(config_file: Path) -> None:
    ctx = multiprocessing.get_context("fork")
    procs = [
        ctx.Process(target=_set_key, args=(str(config_file), i)) for i in range(32)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=60)
        assert p.exitcode == 0

    local = config_file.with_name("config.local.toml")
    env = tomlkit.parse(local.read_text(encoding="utf-8"))["env"]
    assert sorted(env) == sorted(f"KEY_{i}" for i in range(32))


def test_onboard_saves_model_where_gptme_reads_it(config_file: Path) -> None:
    from gptme.cli import onboard

    model = 'openai/odd"name'
    with (
        patch.object(onboard, "config_path", str(config_file)),
        patch.object(
            onboard,
            "get_config",
            return_value=SimpleNamespace(
                user=SimpleNamespace(models=SimpleNamespace(default=None))
            ),
        ),
        patch("gptme.config.core.reload_config"),
    ):
        onboard._create_config("openai", model)

    config = load_user_config(str(config_file))
    assert config.models.default == model
    # The rest of the existing config survives.
    assert config.env["EXISTING"] == "keep"
    assert "chat" not in tomlkit.parse(config_file.read_text(encoding="utf-8"))


def test_onboard_keeps_existing_default_when_declined(config_file: Path) -> None:
    from gptme.cli import onboard

    with (
        patch.object(onboard, "config_path", str(config_file)),
        patch.object(
            onboard,
            "get_config",
            return_value=SimpleNamespace(
                user=SimpleNamespace(models=SimpleNamespace(default="anthropic/x"))
            ),
        ),
        patch.object(onboard.Confirm, "ask", return_value=False),
        patch.object(onboard, "set_config_value") as save,
    ):
        onboard._create_config("openai", "openai/y")

    save.assert_not_called()
