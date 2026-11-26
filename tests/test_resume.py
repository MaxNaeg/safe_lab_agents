"""Resume-command behaviour.

``agent resume`` is interactive by default and autonomous when given a new task
(``--task`` / ``--task-file``). The entrypoint is baked into each committed
session image, so an autonomous resume is refused for images that predate it.
"""

from __future__ import annotations

import inspect
import shutil
import subprocess
import types
from importlib.resources import files as pkg_files
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from safe_lab_agents import cli
from safe_lab_agents.cli import _resolve_task, resume
from safe_lab_agents.config import SessionConfig, SessionMetadata


def test_resume_exposes_task_options() -> None:
    """`resume` takes a new task, directly or from a file."""
    params = inspect.signature(resume).parameters
    assert "task" in params
    assert "task_file" in params


class TestResolveTask:
    """`_resolve_task` is shared by `start` and `resume`."""

    def test_task_passes_through(self) -> None:
        assert _resolve_task("do it", None) == "do it"

    def test_no_task(self) -> None:
        assert _resolve_task(None, None) is None

    def test_reads_and_strips_file(self, tmp_path: Path) -> None:
        task_file = tmp_path / "task.md"
        task_file.write_text("\n  measure the power  \n", encoding="utf-8")
        assert _resolve_task(None, task_file) == "measure the power"

    def test_task_and_file_are_mutually_exclusive(self, tmp_path: Path) -> None:
        task_file = tmp_path / "task.md"
        task_file.write_text("x", encoding="utf-8")
        with pytest.raises(typer.Exit):
            _resolve_task("do it", task_file)

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(typer.Exit):
            _resolve_task(None, tmp_path / "nope.md")


class _FakeDockerManager:
    """Stands in for DockerManager up to the autonomous-resume capability check."""

    supports = False
    created: list["_FakeDockerManager"] = []

    def __init__(self) -> None:
        self.asked: list[str] = []
        type(self).created.append(self)

    def get_session_image_tag(self, name: str) -> str:
        return f"safe-lab-agents-session-{name}:latest"

    def supports_autonomous_resume(self, image_tag: str) -> bool:
        self.asked.append(image_tag)
        return type(self).supports


@pytest.fixture
def resume_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Patch `resume`'s collaborators so it runs up to (not past) the MCP server.

    Constructing the MCP server raises, so a test that expects an early refusal
    also proves nothing was started.
    """
    config = SessionConfig(
        name="s1",
        agent_type="claude-code",
        tools_file=tmp_path / "tools.py",
        workspace_dir=tmp_path / "ws",
        container_runtime="docker",
        task="the original task",
    )
    metadata = SessionMetadata(config=config, status="committed")
    monkeypatch.setattr(
        cli.SessionMetadata, "load", staticmethod(lambda name: metadata)
    )
    monkeypatch.setattr(cli, "_print_banner", lambda: None)
    monkeypatch.setattr(cli, "_activate_container_runtime", lambda runtime: None)
    monkeypatch.setattr(
        cli, "_warn_on_runtime_engine_mismatch", lambda runtime, mgr: None
    )
    _FakeDockerManager.created = []
    _FakeDockerManager.supports = False
    monkeypatch.setattr(cli, "DockerManager", _FakeDockerManager)

    class _ReachedMCP(Exception):
        pass

    def _mcp_server(**kwargs):
        raise _ReachedMCP

    monkeypatch.setattr(cli, "ExperimentMCPServer", _mcp_server)
    monkeypatch.setattr(cli, "find_free_port", lambda: 5000)
    return types.SimpleNamespace(metadata=metadata, reached_mcp=_ReachedMCP)


def test_autonomous_resume_refused_for_older_session_image(resume_env) -> None:
    """A session image without the capability label can't resume autonomously:
    the command fails before the MCP server (or anything else) is started."""
    result = CliRunner().invoke(cli.app, ["resume", "--name", "s1", "--task", "next"])

    assert result.exit_code == 1
    assert not isinstance(result.exception, resume_env.reached_mcp)
    assert _FakeDockerManager.created[0].asked == ["safe-lab-agents-session-s1:latest"]
    assert "cannot be resumed autonomously" in result.output.replace("\n", " ")


def test_autonomous_resume_proceeds_for_capable_image(resume_env) -> None:
    """With the label present the run goes on (here: as far as the MCP server),
    carrying the new task so the agent builds an autonomous environment."""
    _FakeDockerManager.supports = True

    result = CliRunner().invoke(cli.app, ["resume", "--name", "s1", "--task", "next"])

    assert isinstance(result.exception, resume_env.reached_mcp)
    assert resume_env.metadata.config.task == "next"


def test_interactive_resume_skips_capability_check(resume_env) -> None:
    """Without a task, resume stays interactive for every session image and never
    re-runs a task persisted by an earlier run."""
    result = CliRunner().invoke(cli.app, ["resume", "--name", "s1"])

    assert isinstance(result.exception, resume_env.reached_mcp)
    assert _FakeDockerManager.created[0].asked == []
    assert resume_env.metadata.config.task is None


def test_empty_task_is_rejected(resume_env, tmp_path: Path) -> None:
    """An empty task must not silently fall back to an interactive resume."""
    empty = tmp_path / "empty.md"
    empty.write_text("  \n", encoding="utf-8")

    result = CliRunner().invoke(
        cli.app, ["resume", "--name", "s1", "--task-file", str(empty)]
    )

    assert result.exit_code == 1
    assert _FakeDockerManager.created == []


@pytest.mark.parametrize("agent_type", ["claude-code", "openclaw"])
def test_entrypoint_shell_syntax(agent_type: str) -> None:
    """The entrypoints only run inside a container, so at least parse them."""
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash not available")
    script = pkg_files("safe_lab_agents.docker.dockerfiles").joinpath(
        f"entrypoint.{agent_type}.sh"
    )
    result = subprocess.run(
        [bash, "-n", str(script)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("agent_type", ["claude-code", "openclaw"])
def test_dockerfile_advertises_autonomous_resume(agent_type: str) -> None:
    """The label the host checks must be set by the image that ships the
    autonomous-resume entrypoint."""
    from safe_lab_agents.docker.manager import _AUTONOMOUS_RESUME_LABEL

    dockerfile = (
        pkg_files("safe_lab_agents.docker.dockerfiles")
        .joinpath(f"Dockerfile.{agent_type}")
        .read_text(encoding="utf-8")
    )
    assert f'LABEL {_AUTONOMOUS_RESUME_LABEL}="1"' in dockerfile
