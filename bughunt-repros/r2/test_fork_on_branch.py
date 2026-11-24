"""LogManager.fork copies self.logfile.parent; on a non-main branch that is
<logdir>/branches, so the fork loses tool-outputs/, config.toml, the
compaction ledger and the active-view marker that compaction relies on."""
import pytest

from gptme.logmanager import LogManager
from gptme.message import Message


@pytest.mark.parametrize("branch", ["main", "feature"])
def test_fork_keeps_compaction_artifacts(tmp_path, monkeypatch, branch):
    logs = tmp_path / "logs"
    logs.mkdir()
    monkeypatch.setattr("gptme.logmanager.manager.get_logs_dir", lambda: logs)
    logdir = logs / "orig"
    manager = LogManager(
        [Message("system", "s"), Message("user", "u"), Message("assistant", "a")],
        logdir=logdir,
    )
    manager.write()
    # Artifacts written by compaction next to the transcript.
    (logdir / "tool-outputs").mkdir()
    (logdir / "tool-outputs" / "autocompact-abc.txt").write_text("original output")
    (logdir / "compaction.jsonl").write_text("{}\n")
    (logdir / "config.toml").write_text('[chat]\nworkspace = "/tmp"\n')
    if branch != "main":
        manager.branch(branch)

    manager.fork("forked")

    new = logs / "forked"
    missing = [
        p
        for p in ("tool-outputs/autocompact-abc.txt", "compaction.jsonl", "config.toml")
        if not (new / p).exists()
    ]
    assert not missing, f"fork from branch {branch!r} lost: {missing}"
