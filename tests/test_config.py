"""Tests for configuration models."""

from __future__ import annotations

import platform
import subprocess
from datetime import datetime
from pathlib import Path

import pytest

from safe_lab_agents.config import SessionConfig, SessionMetadata


class TestSessionConfig:
    """Tests for :class:`SessionConfig`."""

    def test_defaults(self, tmp_path: Path) -> None:
        """Default values are applied correctly."""
        cfg = SessionConfig(
            name="test",
            tools_file=tmp_path / "tools.py",
            workspace_dir=tmp_path / "ws",
        )
        assert cfg.agent_type == "claude-code"
        assert cfg.context_dir is None
        assert cfg.shared_dir is None
        assert cfg.task is None
        assert cfg.predefined_servers == []
        assert cfg.agent_args == {}
        # Egress lockdown (host reachable only on the MCP port) is on by default.
        assert cfg.egress_lockdown is True
        # No explicit resource limits — the runtime-sized defaults apply.
        assert cfg.mem_limit is None
        assert cfg.cpu_limit is None

    def test_mem_limit_accepts_docker_byte_strings(self, tmp_path: Path) -> None:
        """mem_limit takes anything Docker's parse_bytes takes; junk fails early."""
        for good in ("8g", "512M", "2gb", "1073741824", "1.5g"):
            cfg = SessionConfig(
                name="t", tools_file=tmp_path / "t.py",
                workspace_dir=tmp_path / "ws", mem_limit=good,
            )
            assert cfg.mem_limit == good
        for bad in ("8 g", "lots", "-1g", "g8", ""):
            with pytest.raises(ValueError):
                SessionConfig(
                    name="t", tools_file=tmp_path / "t.py",
                    workspace_dir=tmp_path / "ws", mem_limit=bad,
                )

    def test_cpu_limit_must_be_positive(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            SessionConfig(
                name="t", tools_file=tmp_path / "t.py",
                workspace_dir=tmp_path / "ws", cpu_limit=0,
            )

    def test_container_runtime_rejects_unknown_value(self, tmp_path: Path) -> None:
        """container_runtime is a Literal — only 'docker'/'podman' are accepted."""
        for runtime in ("docker", "podman"):
            cfg = SessionConfig(
                name="t", tools_file=tmp_path / "t.py",
                workspace_dir=tmp_path / "ws", container_runtime=runtime,
            )
            assert cfg.container_runtime == runtime
        with pytest.raises(ValueError):
            SessionConfig(
                name="t", tools_file=tmp_path / "t.py",
                workspace_dir=tmp_path / "ws", container_runtime="kubernetes",
            )

    def test_all_fields(self, tmp_path: Path) -> None:
        """All fields can be set explicitly."""
        cfg = SessionConfig(
            name="full",
            agent_type="openclaw",
            tools_file=tmp_path / "tools.py",
            context_dir=tmp_path / "ctx",
            shared_dir=tmp_path / "shared",
            workspace_dir=tmp_path / "ws",
            requirements_file=tmp_path / "req.txt",
            mcp_port=9999,
            task="run experiment",
            predefined_servers=["lab-notebook"],
            agent_args={"api-key": "sk-test", "provider": "anthropic", "model": "claude-sonnet-4-20250514"},
        )
        assert cfg.agent_type == "openclaw"
        assert cfg.mcp_port == 9999
        assert cfg.agent_args["model"] == "claude-sonnet-4-20250514"
        assert cfg.agent_args["api-key"] == "sk-test"


class TestGetBaseDir:
    """Tests for :func:`get_base_dir` permissions."""

    @pytest.mark.skipif(
        platform.system() == "Windows",
        reason="POSIX mode bits; Windows uses ACLs (see test_base_dir_owner_only_windows)",
    )
    def test_base_dir_is_owner_only(self, tmp_path: Path, monkeypatch) -> None:
        """The base dir is gated at 0700 so other local users cannot traverse
        into the (deliberately world-writable) session trees or read secrets
        in metadata.json — and pre-existing wide installs are tightened."""
        import safe_lab_agents.config as config_mod

        monkeypatch.setattr(config_mod.Path, "home", lambda: tmp_path)

        base = config_mod.get_base_dir()
        assert base == tmp_path / ".safe_lab_agents"
        assert (base.stat().st_mode & 0o777) == 0o700

        # An existing install with loose permissions is re-tightened.
        base.chmod(0o755)
        config_mod.get_base_dir()
        assert (base.stat().st_mode & 0o777) == 0o700

    def test_base_dir_owner_only_windows(self, tmp_path: Path, monkeypatch) -> None:
        """On Windows, chmod is a no-op on NTFS, so the base dir is restricted
        via ``icacls``: inheritance stripped and an inheritable full-control
        grant to the current user only."""
        import safe_lab_agents.config as config_mod

        monkeypatch.setattr(config_mod.Path, "home", lambda: tmp_path)
        monkeypatch.setattr(config_mod.platform, "system", lambda: "Windows")
        monkeypatch.setenv("USERNAME", "alice")
        monkeypatch.setenv("USERDOMAIN", "LABPC")

        calls: list[list[str]] = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            return subprocess.CompletedProcess(cmd, 0, "", "")

        monkeypatch.setattr(config_mod.subprocess, "run", fake_run)

        base = config_mod.get_base_dir()

        assert base.is_dir()
        assert len(calls) == 1
        cmd = calls[0]
        assert cmd[0] == "icacls"
        assert str(base) in cmd
        assert "/inheritance:r" in cmd
        # Inheritable (OI)(CI) full-control grant to DOMAIN\user only.
        assert "/grant:r" in cmd
        assert "LABPC\\alice:(OI)(CI)F" in cmd

    def test_base_dir_owner_only_windows_best_effort(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """An icacls failure is logged, not fatal — the dir still exists."""
        import safe_lab_agents.config as config_mod

        monkeypatch.setattr(config_mod.Path, "home", lambda: tmp_path)
        monkeypatch.setattr(config_mod.platform, "system", lambda: "Windows")
        monkeypatch.setenv("USERNAME", "alice")
        monkeypatch.delenv("USERDOMAIN", raising=False)

        def boom(cmd, **kwargs):
            raise subprocess.CalledProcessError(1, cmd, stderr="No mapping")

        monkeypatch.setattr(config_mod.subprocess, "run", boom)

        base = config_mod.get_base_dir()  # must not raise
        assert base.is_dir()


class TestSessionMetadata:
    """Tests for :class:`SessionMetadata` persistence."""

    def test_save_and_load(self, tmp_path: Path, monkeypatch) -> None:
        """Metadata round-trips through save/load."""
        monkeypatch.setattr(
            "safe_lab_agents.config.get_sessions_dir",
            lambda: tmp_path / "sessions",
        )
        cfg = SessionConfig(
            name="roundtrip",
            tools_file=tmp_path / "tools.py",
            workspace_dir=tmp_path / "ws",
        )
        meta = SessionMetadata(
            config=cfg,
            container_id="abc123",
            status="running",
            started_at=datetime(2026, 4, 13, 12, 0, 0),
        )
        meta.save()

        loaded = SessionMetadata.load("roundtrip")
        assert loaded.config.name == "roundtrip"
        assert loaded.container_id == "abc123"
        assert loaded.status == "running"

    def test_load_missing_raises(self, tmp_path: Path, monkeypatch) -> None:
        """FileNotFoundError for a non-existent session."""
        monkeypatch.setattr(
            "safe_lab_agents.config.get_sessions_dir",
            lambda: tmp_path / "sessions",
        )
        with pytest.raises(FileNotFoundError):
            SessionMetadata.load("nonexistent")

    def test_list_sessions(self, tmp_path: Path, monkeypatch) -> None:
        """list_sessions returns all saved sessions."""
        monkeypatch.setattr(
            "safe_lab_agents.config.get_sessions_dir",
            lambda: tmp_path / "sessions",
        )
        for name in ("s1", "s2"):
            cfg = SessionConfig(
                name=name,
                tools_file=tmp_path / "tools.py",
                workspace_dir=tmp_path / "ws",
            )
            SessionMetadata(config=cfg, status="committed").save()

        sessions = SessionMetadata.list_sessions()
        names = [s.config.name for s in sessions]
        assert "s1" in names
        assert "s2" in names
