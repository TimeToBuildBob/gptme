"""Config writes are atomic, private for secrets, and safe under concurrency.

Regression tests for gptme/gptme#5128.
"""

import errno
import multiprocessing
import os
import stat
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any
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


@posix_only
@pytest.mark.parametrize("final_mode", [0o600, 0o640, 0o644])
def test_staging_file_is_owner_only_until_written(
    config_file: Path, monkeypatch: pytest.MonkeyPatch, final_mode: int
) -> None:
    """The staged copy is never more open than 0600 while it gets the contents.

    Another user who opens a wider staging file keeps that descriptor after a
    later chmod, and would read the config (which may hold API keys) through it.
    """
    config_file.chmod(final_mode)
    real_open = os.open
    real_fchmod = os.fchmod
    staged_fds: set[int] = set()
    modes_at_create: list[int] = []
    # (mode, staged size) for every chmod of the staging file
    chmods: list[tuple[int, int]] = []

    def recording_open(path, flags, mode=0o777, *args, **kwargs):
        fd = real_open(path, flags, mode, *args, **kwargs)
        if str(path).endswith(".tmp"):
            staged_fds.add(fd)
            modes_at_create.append(stat.S_IMODE(os.fstat(fd).st_mode))
        return fd

    def recording_fchmod(fd: int, mode: int) -> None:
        if fd in staged_fds:
            chmods.append((mode, os.fstat(fd).st_size))
        real_fchmod(fd, mode)

    monkeypatch.setattr(user_mod.os, "open", recording_open)
    monkeypatch.setattr(user_mod.os, "fchmod", recording_fchmod)
    old_umask = os.umask(0o022)
    try:
        set_config_value("env.OPENAI_API_KEY", "sk-secret", reload=False)
    finally:
        os.umask(old_umask)

    assert modes_at_create, "staging file was not observed"
    assert all(m & 0o077 == 0 for m in modes_at_create), [
        oct(m) for m in modes_at_create
    ]
    # Group/other bits are only granted once the complete contents are in place.
    final_size = config_file.stat().st_size
    assert all(size == final_size for mode, size in chmods if mode & 0o077), chmods
    assert _mode(config_file) == final_mode
    assert "sk-secret" in config_file.read_text(encoding="utf-8")


class _FakeMsvcrt:
    """Models msvcrt.locking: LK_LOCK gives up after 10 one-second retries."""

    LK_LOCK, LK_NBLCK, LK_UNLCK = 1, 2, 0

    def __init__(self, busy_attempts: int, fail_errno: int = 13) -> None:
        self.busy_attempts = busy_attempts
        self.fail_errno = fail_errno
        self.locked = False

    def _try(self) -> bool:
        if self.busy_attempts > 0:
            self.busy_attempts -= 1
            return False
        self.locked = True
        return True

    def locking(self, fd: int, mode: int, nbytes: int) -> None:
        if mode == self.LK_UNLCK:
            self.locked = False
        elif mode == self.LK_NBLCK:
            if not self._try():
                raise OSError(self.fail_errno, "Permission denied")
        elif mode == self.LK_LOCK:
            if not any(self._try() for _ in range(10)):
                raise OSError(36, "Resource deadlock avoided")


def test_windows_config_write_lock_waits_for_a_long_held_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Writers wait for the lock on Windows too, not give up after ~10s."""

    fake = _FakeMsvcrt(busy_attempts=15, fail_errno=errno.EACCES)
    monkeypatch.setattr(user_mod, "fcntl", None)
    monkeypatch.setattr(user_mod, "msvcrt", fake)
    monkeypatch.setattr(time, "sleep", lambda _s: None)

    with user_mod._config_write_lock(tmp_path / "config.toml"):
        assert fake.locked
    assert not fake.locked
    assert fake.busy_attempts == 0


def test_windows_config_write_lock_reraises_other_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeMsvcrt(busy_attempts=1, fail_errno=errno.EBADF)
    monkeypatch.setattr(user_mod, "fcntl", None)
    monkeypatch.setattr(user_mod, "msvcrt", fake)
    monkeypatch.setattr(time, "sleep", lambda _s: None)

    with (
        pytest.raises(OSError, match="Permission denied") as excinfo,
        user_mod._config_write_lock(tmp_path / "config.toml"),
    ):
        pass
    assert excinfo.value.errno == errno.EBADF


def test_replace_retries_transient_permission_error(
    config_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Windows sharing violation (reader/AV) must not fail the save."""
    real_replace = os.replace
    calls = 0

    def flaky_replace(src, dst):
        nonlocal calls
        calls += 1
        if calls <= 2:
            raise PermissionError(13, "Permission denied")
        return real_replace(src, dst)

    monkeypatch.setattr(user_mod.os, "replace", flaky_replace)
    monkeypatch.setattr(user_mod.time, "sleep", lambda _s: None)

    set_config_value("models.default", "openai/retried", reload=False)

    assert calls == 3
    assert load_user_config(str(config_file)).models.default == "openai/retried"


def test_replace_raises_clear_error_after_exhausting_retries(
    config_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A permanently locked config surfaces a clear error, not a bare traceback."""
    original = config_file.read_text(encoding="utf-8")

    def always_locked(src, dst):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(user_mod.os, "replace", always_locked)
    monkeypatch.setattr(user_mod.time, "sleep", lambda _s: None)

    with pytest.raises(OSError, match="remained locked by another process"):
        set_config_value("models.default", "openai/x", reload=False)

    # The old file is intact and no staging file is left behind.
    assert config_file.read_text(encoding="utf-8") == original
    leftovers = sorted(
        p.name for p in config_file.parent.iterdir() if p.name.endswith(".tmp")
    )
    assert leftovers == []


def test_onboard_offers_to_overwrite_corrupt_config(config_file: Path) -> None:
    """A corrupt config must keep the overwrite recovery path, not crash."""
    from gptme.cli import onboard

    corrupt = '[models\ndefault = "broken"\n'
    config_file.write_text(corrupt, encoding="utf-8")

    with (
        patch.object(onboard, "config_path", str(config_file)),
        patch.object(
            onboard,
            "get_config",
            side_effect=tomlkit.exceptions.TOMLKitError("corrupt"),
        ),
        patch.object(onboard.Confirm, "ask", return_value=True) as ask,
        patch("gptme.config.core.reload_config"),
    ):
        result = onboard._create_config("openai", "openai/new")

    assert result == config_file
    ask.assert_called_once()
    assert "Overwrite the corrupt config?" in ask.call_args.args[0]
    backup = config_file.with_name("config.toml.corrupt")
    assert backup.read_text(encoding="utf-8") == corrupt
    assert load_user_config(str(config_file)).models.default == "openai/new"


def test_onboard_declining_corrupt_overwrite_writes_nothing(
    config_file: Path,
) -> None:
    from gptme.cli import onboard

    corrupt = '[models\ndefault = "broken"\n'
    config_file.write_text(corrupt, encoding="utf-8")

    with (
        patch.object(onboard, "config_path", str(config_file)),
        patch.object(
            onboard,
            "get_config",
            side_effect=tomlkit.exceptions.TOMLKitError("corrupt"),
        ),
        patch.object(onboard.Confirm, "ask", return_value=False),
        patch.object(onboard, "set_config_value") as save,
    ):
        assert onboard._create_config("openai", "openai/new") == config_file

    save.assert_not_called()
    assert config_file.read_text(encoding="utf-8") == corrupt


def _hold_config_lock(path: str, ready: Any, release: Any) -> None:
    with user_mod._config_write_lock(path):
        ready.set()
        release.wait(30)


def _acquire_config_lock(path: str, acquired: Any) -> None:
    with user_mod._config_write_lock(path):
        acquired.set()


@pytest.mark.skipif(
    sys.platform != "win32",
    reason="Windows file locking (msvcrt); spawn only exercised there",
)
def test_windows_spawned_writer_waits_for_the_lock(tmp_path: Path) -> None:
    """One spawn-based writer waits while another process holds the lock.

    The fake-msvcrt test and the fork-based concurrency test both skip Windows,
    so this is the only test that exercises the real Windows lock across
    processes. It is skipped on non-Windows hosts.
    """
    ctx = multiprocessing.get_context("spawn")
    path = str(tmp_path / "config.toml")
    ready = ctx.Event()
    release = ctx.Event()
    holder = ctx.Process(target=_hold_config_lock, args=(path, ready, release))
    holder.start()
    waiter = None
    try:
        assert ready.wait(30), "lock holder did not start"
        acquired = ctx.Event()
        waiter = ctx.Process(target=_acquire_config_lock, args=(path, acquired))
        waiter.start()
        assert not acquired.wait(1.0), "second writer did not wait for the lock"
        release.set()
        assert acquired.wait(30), "second writer never got the lock"
        waiter.join(30)
        assert waiter.exitcode == 0
    finally:
        release.set()
        holder.join(30)
        if waiter is not None:
            waiter.join(30)
        assert holder.exitcode == 0
