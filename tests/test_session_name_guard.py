"""Tests for the duplicate-session-name guard in ``agent start``."""

from __future__ import annotations

from pathlib import Path

import pytest
import typer

from safe_lab_agents import cli


@pytest.fixture
def sessions_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point ``get_sessions_dir`` (as used by the cli module) at a tmp dir."""
    d = tmp_path / "sessions"
    d.mkdir()
    monkeypatch.setattr(cli, "get_sessions_dir", lambda: d)
    return d


def test_session_exists_detects_existing_dir(sessions_dir: Path) -> None:
    (sessions_dir / "calibration").mkdir()
    assert cli._session_exists("calibration") is True
    assert cli._session_exists("brand_new") is False


def test_reject_existing_session_exits(sessions_dir: Path) -> None:
    (sessions_dir / "calibration").mkdir()
    with pytest.raises(typer.Exit) as exc:
        cli._reject_existing_session("calibration")
    assert exc.value.exit_code == 1


def test_reject_existing_session_allows_free_name(sessions_dir: Path) -> None:
    # Does not raise for a name with no existing session directory.
    cli._reject_existing_session("brand_new")


def test_prompt_session_name_reasks_on_clash(
    sessions_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (sessions_dir / "calibration").mkdir()
    answers = iter(["calibration", "fresh_one"])
    monkeypatch.setattr(cli.Prompt, "ask", lambda *a, **k: next(answers))
    assert cli._prompt_session_name() == "fresh_one"
