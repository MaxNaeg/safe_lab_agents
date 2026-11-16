"""Tests for the --auto-log-prompt option (full / minimal / none / custom file)."""

from __future__ import annotations

from pathlib import Path

import pytest
import typer

from safe_lab_agents.cli import (
    _load_template,
    _resolve_auto_log_prompt,
    _write_auto_log_info,
)

LOG_DIR = "/agent/shared/auto_log"


def test_full_is_the_unchanged_bundled_prompt() -> None:
    setting, text = _resolve_auto_log_prompt("full")
    assert setting == "full"
    assert text == _load_template("auto_log_info.txt")


def test_minimal_is_short_and_generic() -> None:
    setting, text = _resolve_auto_log_prompt("minimal")
    assert setting == "minimal"
    assert text is not None
    assert len(text.split()) <= 160
    assert "start_batch" in text and "log_analysis" in text
    assert "voltage" not in text.lower()
    assert "ALWAYS" not in text


def test_none_has_no_text() -> None:
    assert _resolve_auto_log_prompt("none") == ("none", None)


def test_custom_file_stored_as_absolute_path(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "p.md").write_text("Logs are in __CONTAINER_LOG_DIR__.", encoding="utf-8")
    setting, text = _resolve_auto_log_prompt("p.md")
    assert setting == str((tmp_path / "p.md").resolve())
    assert text == "Logs are in __CONTAINER_LOG_DIR__."


def test_missing_file_is_error(tmp_path: Path) -> None:
    with pytest.raises(typer.Exit):
        _resolve_auto_log_prompt(str(tmp_path / "nope.md"))


def test_empty_file_is_error(tmp_path: Path) -> None:
    f = tmp_path / "empty.md"
    f.write_text("  \n", encoding="utf-8")
    with pytest.raises(typer.Exit):
        _resolve_auto_log_prompt(str(f))


def test_write_substitutes_log_dir(tmp_path: Path) -> None:
    _write_auto_log_info(tmp_path, LOG_DIR, "Logs are in __CONTAINER_LOG_DIR__/.")
    text = (tmp_path / "auto_log_info.txt").read_text(encoding="utf-8")
    assert text == f"Logs are in {LOG_DIR}/."


def test_write_none_removes_stale_file(tmp_path: Path) -> None:
    """The entrypoints inject auto_log_info.txt whenever it exists, so 'none'
    must remove a leftover copy."""
    (tmp_path / "auto_log_info.txt").write_text("stale", encoding="utf-8")
    _write_auto_log_info(tmp_path, LOG_DIR, None)
    assert not (tmp_path / "auto_log_info.txt").exists()
    _write_auto_log_info(tmp_path, LOG_DIR, None)  # no error when already absent
