# mypy: ignore-errors
"""`gptme --resume --name conv -w .` (or -w $PWD) is silently ignored: WorkspacePath
resolves the flag to an absolute path equal to Path.cwd(), which equals the
ChatConfig.workspace default_factory, so load_or_create treats it as "not provided".
Any other directory is honoured, so the behaviour depends on where you stand."""

import os

from gptme.config import setup_config_from_cli


def test_explicit_workspace_equal_to_cwd_overrides_on_resume(tmp_path):
    ws_a = tmp_path / "a"
    ws_b = tmp_path / "b"
    ws_a.mkdir()
    ws_b.mkdir()
    logdir = tmp_path / "logs" / "conv"

    os.chdir(ws_a)
    setup_config_from_cli(workspace=ws_a, logdir=logdir, model="openai/gpt-4o")

    # user cds to b and explicitly asks to move the conversation there
    os.chdir(ws_b)
    cfg = setup_config_from_cli(workspace=ws_b.resolve(), logdir=logdir)
    assert cfg.chat is not None
    assert cfg.chat.workspace.resolve() == ws_b.resolve()


def test_control_explicit_workspace_other_than_cwd_is_honoured(tmp_path):
    ws_a = tmp_path / "a"
    ws_b = tmp_path / "b"
    ws_a.mkdir()
    ws_b.mkdir()
    logdir = tmp_path / "logs" / "conv"
    os.chdir(ws_a)
    setup_config_from_cli(workspace=ws_a, logdir=logdir, model="openai/gpt-4o")
    os.chdir(tmp_path)  # stand elsewhere
    cfg = setup_config_from_cli(workspace=ws_b.resolve(), logdir=logdir)
    assert cfg.chat is not None
    assert cfg.chat.workspace.resolve() == ws_b.resolve()
