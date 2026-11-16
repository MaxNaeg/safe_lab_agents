"""Tests for resolving the --system-prompt / --system-prompt-file CLI options."""

from __future__ import annotations

from pathlib import Path

import pytest
import typer

from safe_lab_agents.cli import _resolve_system_prompt, _write_system_prompt
from safe_lab_agents.agents import get_agent
from safe_lab_agents.config import SessionConfig


def test_none_means_builtin() -> None:
    assert _resolve_system_prompt(None, None) is None


def test_inline_prompt_stripped() -> None:
    assert _resolve_system_prompt("  Be terse.\n", None) == "Be terse."


def test_prompt_from_file(tmp_path: Path) -> None:
    f = tmp_path / "prompt.md"
    f.write_text("You review code.\n", encoding="utf-8")
    assert _resolve_system_prompt(None, f) == "You review code."


def test_both_given_is_error(tmp_path: Path) -> None:
    f = tmp_path / "prompt.md"
    f.write_text("x", encoding="utf-8")
    with pytest.raises(typer.Exit):
        _resolve_system_prompt("y", f)


def test_missing_file_is_error(tmp_path: Path) -> None:
    with pytest.raises(typer.Exit):
        _resolve_system_prompt(None, tmp_path / "nope.md")


def test_empty_prompt_is_error(tmp_path: Path) -> None:
    f = tmp_path / "empty.md"
    f.write_text("  \n", encoding="utf-8")
    with pytest.raises(typer.Exit):
        _resolve_system_prompt(None, f)
    with pytest.raises(typer.Exit):
        _resolve_system_prompt("", None)


def test_custom_prompt_written_to_workspace(tmp_path: Path) -> None:
    """The entrypoints read system_prompt.txt, so the custom text must land there."""
    cfg = SessionConfig(
        name="t", tools_file=tmp_path / "t.py", workspace_dir=tmp_path / "ws",
        system_prompt="You review code.",
    )
    _write_system_prompt(cfg, get_agent("claude-code"))
    text = (tmp_path / "ws" / "system_prompt.txt").read_text(encoding="utf-8")
    assert text.startswith("You review code.\n")
    assert "scientific experiment" not in text
    assert "/agent/workspace" in text
    assert "/agent/shared" in text
