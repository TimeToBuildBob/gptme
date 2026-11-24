"""Credential files inside the workspace are re-injected by resume.

_load_context_files() loads whatever the (model-written, injection-steerable)
checkpoint lists under "Context Files". _is_secret_bearing_path() only blocks
dotenv names and .ssh/.gnupg/.aws/.config/secrets, so common credential files
(.envrc, .netrc, .git-credentials) are read verbatim into a system message of
the compacted view and sent to the provider.
"""

import pytest

from gptme.logmanager import LogManager
from gptme.message import Message
from gptme.tools.autocompact.resume import _load_context_files, _resume_via_llm

SECRET = "ghp_SUPERSECRETTOKEN1234567890"


@pytest.mark.parametrize(
    "name,body",
    [
        (".envrc", f"export GITHUB_TOKEN={SECRET}\n"),
        (".netrc", f"machine api.github.com login x password {SECRET}\n"),
        (".git-credentials", f"https://user:{SECRET}@github.com\n"),
    ],
)
def test_credential_file_not_loaded(tmp_path, name, body):
    (tmp_path / name).write_text(body)
    loaded = _load_context_files([name], workspace=tmp_path)
    assert loaded == [], f"credential file {name} was loaded: {loaded}"


def test_checkpoint_does_not_inline_git_credentials(tmp_path):
    from gptme.llm.models import set_default_model

    set_default_model("openai/gpt-4o")
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / ".git-credentials").write_text(f"https://user:{SECRET}@github.com\n")
    messages = [
        Message("system", "system prompt"),
        Message("user", "please continue"),
        Message("assistant", "ok"),
        Message("user", "more"),
    ]
    logdir = tmp_path / "conv"
    logdir.mkdir()
    (logdir / "workspace").symlink_to(ws)
    manager = LogManager(messages, logdir=logdir, lock=False)
    manager.write()
    assert manager.workspace == ws.resolve()
    checkpoint = Message(
        "assistant",
        "## Objective\nx\n\n## Context Files\n- `.git-credentials` — needed\n",
    )
    list(
        _resume_via_llm(
            manager, messages, use_view_branch=True, checkpoint_response=checkpoint
        )
    )
    leaked = [m.content for m in manager.log.messages if SECRET in m.content]
    assert not leaked, leaked
