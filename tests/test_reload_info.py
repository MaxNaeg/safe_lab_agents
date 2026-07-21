"""Tests for the reload_tools system-prompt guidance (reload_info.txt).

reload_tools is an MCP tool that only exists with --update-tools, so its prompt
guidance must be gated on that flag and must not linger on a resume that does
not enable it.
"""

from __future__ import annotations

from pathlib import Path

from safe_lab_agents.cli import _load_template, _sync_reload_info
from safe_lab_agents.config import SessionConfig


def _config(tmp_path: Path) -> SessionConfig:
    return SessionConfig(
        name="s", tools_file=tmp_path / "t.py", workspace_dir=tmp_path / "ws",
    )


def test_reload_info_text_is_user_driven() -> None:
    """The guidance frames the user as the editor and forbids self-initiated reloads."""
    reload_info = _load_template("reload_info.txt")
    assert "reload_tools" in reload_info
    assert "edited by the user" in reload_info
    assert "never on your own" in reload_info


def test_reload_info_written_when_available(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    _sync_reload_info(cfg, reload_available=True)
    path = cfg.workspace_dir / "reload_info.txt"
    assert path.exists()
    assert "reload_tools" in path.read_text(encoding="utf-8")


def test_reload_info_absent_when_unavailable(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    cfg.workspace_dir.mkdir(parents=True, exist_ok=True)
    assert not (cfg.workspace_dir / "reload_info.txt").exists()
    _sync_reload_info(cfg, reload_available=False)
    assert not (cfg.workspace_dir / "reload_info.txt").exists()


def test_reload_info_removed_on_stale_transition(tmp_path: Path) -> None:
    """A stale file from an --update-tools start is cleared when reload is unavailable."""
    cfg = _config(tmp_path)
    _sync_reload_info(cfg, reload_available=True)
    assert (cfg.workspace_dir / "reload_info.txt").exists()
    _sync_reload_info(cfg, reload_available=False)
    assert not (cfg.workspace_dir / "reload_info.txt").exists()
