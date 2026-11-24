# mypy: ignore-errors
"""F3/F4: `gptme init` data-safety bugs."""

import subprocess
from pathlib import Path

from click.testing import CliRunner

import gptme.cli.cmd_init as cmd_init


def _make_template_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "tmpl-src"
    repo.mkdir()
    (repo / "gptme.toml").write_text('[agent]\nname = "template"\n')
    (repo / "README.md").write_text("# template\n")
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-C", str(repo)]
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run([*git, "add", "."], check=True)
    subprocess.run([*git, "commit", "-qm", "init"], check=True)
    return repo


def test_force_with_template_into_nonempty_dir(tmp_path, monkeypatch):
    """The error for a non-empty dir says 'Use --force', but --force + --template
    then always fails because git/gh refuse to clone into a non-empty directory."""
    src = _make_template_repo(tmp_path)
    real_run = subprocess.run

    def fake_run(cmd, *a, **kw):
        # redirect the github URL to the local template repo, keep real git
        cmd = [str(src) if str(c).startswith("https://github.com/") else c for c in cmd]
        return real_run(cmd, *a, **kw)

    monkeypatch.setattr(cmd_init.shutil, "which", lambda name: None)  # no gh
    monkeypatch.setattr(cmd_init.subprocess, "run", fake_run)

    target = tmp_path / "proj"
    target.mkdir()
    (target / "notes.txt").write_text("existing user file\n")

    r = CliRunner().invoke(cmd_init.main, [str(target), "--template", "org/tmpl"])
    assert r.exit_code == 1 and "Use --force" in r.output

    r = CliRunner().invoke(
        cmd_init.main, [str(target), "--template", "org/tmpl", "--force"]
    )
    assert r.exit_code == 0, r.output


def test_dangling_claude_symlink_writes_outside_project(tmp_path):
    """A dangling CLAUDE.md symlink makes the copy fallback write through the
    link to an arbitrary path outside the project."""
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    victim = outside / "notes.md"  # does not exist yet -> dangling link
    target = tmp_path / "proj"
    target.mkdir()
    (target / "CLAUDE.md").symlink_to(victim)

    r = CliRunner().invoke(cmd_init.main, [str(target), "--force"])
    assert r.exit_code == 0, r.output
    assert not victim.exists(), (
        f"gptme init wrote {victim} outside the project: {victim.read_text()[:40]!r}"
    )


def test_force_does_not_clobber_claude_md_through_agents_symlink(tmp_path):
    """Common layout: real CLAUDE.md + AGENTS.md -> CLAUDE.md. The code skips
    CLAUDE.md to preserve it, but writing AGENTS.md follows the link."""
    target = tmp_path / "proj"
    target.mkdir()
    (target / "CLAUDE.md").write_text("MY HAND-WRITTEN INSTRUCTIONS\n")
    (target / "AGENTS.md").symlink_to("CLAUDE.md")

    r = CliRunner().invoke(cmd_init.main, [str(target), "--force"])
    assert r.exit_code == 0, r.output
    assert (target / "CLAUDE.md").read_text() == "MY HAND-WRITTEN INSTRUCTIONS\n"
