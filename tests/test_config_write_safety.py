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


@posix_only
def test_symlinked_local_config_with_other_target_name_is_private(
    config_file: Path,
) -> None:
    target = config_file.parent / "secrets.toml"
    target.write_text("[env]\n", encoding="utf-8")
    target.chmod(0o644)
    config_file.with_name("config.local.toml").symlink_to(target)

    set_config_value("env.KEY", "secret", reload=False, local=True)

    assert _mode(target) == 0o600


def test_first_run_create_does_not_replace_concurrent_save(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('[models]\ndefault = "theirs"\n', encoding="utf-8")
    assert not user_mod._write_config_text(path, "[env]\n", exclusive=True)
    assert "theirs" in path.read_text(encoding="utf-8")
    # A lock sidecar is expected; no staged or placeholder file is left.
    left = [p.name for p in tmp_path.iterdir() if not p.name.endswith(".lock")]
    assert left == ["config.toml"]


def test_onboard_updates_local_default_when_it_wins(config_file: Path) -> None:
    from gptme.cli import onboard

    local = config_file.with_name("config.local.toml")
    local.write_text('[models]\ndefault = "anthropic/old"\n', encoding="utf-8")
    with (
        patch.object(onboard, "config_path", str(config_file)),
        patch.object(
            onboard,
            "get_config",
            return_value=SimpleNamespace(
                user=SimpleNamespace(models=SimpleNamespace(default="anthropic/old"))
            ),
        ),
        patch.object(onboard.Confirm, "ask", return_value=True),
        patch("gptme.config.core.reload_config"),
    ):
        onboard._create_config("openai", "openai/new")

    assert load_user_config(str(config_file)).models.default == "openai/new"


def test_onboard_reads_legacy_encoded_local_config(config_file: Path) -> None:
    """A config.local.toml gptme can load (non-UTF-8) must not crash onboarding."""
    from gptme.cli import onboard

    local = config_file.with_name("config.local.toml")
    local.write_bytes('# café\n[models]\ndefault = "anthropic/old"\n'.encode("cp1252"))
    with (
        patch.object(onboard, "config_path", str(config_file)),
        patch.object(
            onboard,
            "get_config",
            return_value=SimpleNamespace(
                user=SimpleNamespace(models=SimpleNamespace(default="anthropic/old"))
            ),
        ),
        patch.object(onboard.Confirm, "ask", return_value=True),
        patch("gptme.config.core.reload_config"),
    ):
        assert onboard._create_config("openai", "openai/new") == local

    assert load_user_config(str(config_file)).models.default == "openai/new"


@pytest.mark.parametrize("existing", [False, True])
def test_first_run_create_without_hard_links(tmp_path: Path, existing: bool) -> None:
    """Exclusive create still works (and still refuses to replace) without os.link."""
    path = tmp_path / "config.toml"
    if existing:
        path.write_text('[models]\ndefault = "theirs"\n', encoding="utf-8")

    def no_links(src, dst):
        raise PermissionError(1, "Operation not permitted")

    with patch.object(user_mod.os, "link", no_links):
        created = user_mod._write_config_text(path, "[env]\n", exclusive=True)

    assert created is not existing
    expected = "theirs" if existing else "[env]"
    assert expected in path.read_text(encoding="utf-8")
    # A lock sidecar is expected; no staged or placeholder file is left.
    left = [p.name for p in tmp_path.iterdir() if not p.name.endswith(".lock")]
    assert left == ["config.toml"]


@posix_only
def test_reencoding_backup_is_never_wider_than_the_original(
    config_file: Path,
) -> None:
    """The .orig backup of a legacy-encoded local config stays owner-only."""
    local = config_file.with_name("config.local.toml")
    original = '[env]\nKEY = "café"\n'.encode("cp1252")
    local.write_bytes(original)
    local.chmod(0o600)
    load_user_config(str(config_file))  # marks the guessed decoding

    old_umask = os.umask(0o022)
    try:
        set_config_value("models.default", "openai/new", reload=False, local=True)
    finally:
        os.umask(old_umask)

    backup = local.with_name("config.local.toml.orig")
    assert _mode(backup) == 0o600
    assert backup.read_bytes() == original


def test_first_run_create_failed_publish_leaves_no_placeholder(
    tmp_path: Path,
) -> None:
    """If publishing fails without hard links, no empty config is left behind."""
    path = tmp_path / "config.toml"

    def no_links(src, dst):
        raise PermissionError(1, "Operation not permitted")

    def failing_replace(src, dst):
        raise OSError(5, "Input/output error")

    with (
        patch.object(user_mod.os, "link", no_links),
        patch.object(user_mod.os, "replace", failing_replace),
        pytest.raises(OSError, match="Input/output error"),
    ):
        user_mod._write_config_text(path, "[env]\n", exclusive=True)

    assert [p for p in tmp_path.iterdir() if not p.name.endswith(".lock")] == []


def test_first_run_cleanup_holds_the_write_lock(tmp_path: Path) -> None:
    """A concurrent save cannot land between the cleanup check and the unlink."""
    import threading

    path = tmp_path / "config.toml"
    other_writer_got_lock = threading.Event()
    writers: list[threading.Thread] = []

    def other_writer() -> None:
        with user_mod._config_write_lock(path):
            other_writer_got_lock.set()

    def no_links(src, dst):
        raise PermissionError(1, "Operation not permitted")

    def replace_fails_while_another_writer_waits(src, dst):
        writer = threading.Thread(target=other_writer)
        writers.append(writer)
        writer.start()
        assert not other_writer_got_lock.wait(0.3)
        raise OSError(5, "Input/output error")

    try:
        with (
            patch.object(user_mod.os, "link", no_links),
            patch.object(
                user_mod.os, "replace", replace_fails_while_another_writer_waits
            ),
            pytest.raises(OSError, match="Input/output error"),
        ):
            user_mod._write_config_text(path, "[env]\n", exclusive=True)

        assert other_writer_got_lock.wait(5)
        assert not path.exists()
    finally:
        for writer in writers:
            writer.join(5)
    assert not any(writer.is_alive() for writer in writers)


def test_config_write_lock_is_reentrant_in_one_thread(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    with user_mod._config_write_lock(path), user_mod._config_write_lock(path):
        pass
