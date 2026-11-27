"""Reading user-supplied text files (task, system prompt, auto-log prompt) in the
encodings Windows tools produce: UTF-8 with BOM (Notepad), UTF-16 with BOM
(PowerShell 5.1 ``>``), and legacy cp1252 ("ANSI")."""

from __future__ import annotations

import codecs
from pathlib import Path

import pytest
import typer

from safe_lab_agents.cli import (
    _read_user_text,
    _resolve_auto_log_prompt,
    _resolve_system_prompt,
    _resolve_task,
)

TEXT = "Measure the power at 5 µW, 20 °C.\nThen report."


@pytest.mark.parametrize(
    "data",
    [
        TEXT.encode("utf-8"),
        codecs.BOM_UTF8 + TEXT.encode("utf-8"),
        TEXT.encode("utf-16"),  # BOM + native order, like PowerShell 5.1 `>`
        codecs.BOM_UTF16_BE + TEXT.encode("utf-16-be"),
        TEXT.replace("\n", "\r\n").encode("utf-8"),  # Windows line endings
    ],
    ids=["utf8", "utf8-bom", "utf16", "utf16-be", "crlf"],
)
def test_read_user_text_accepts_common_encodings(tmp_path: Path, data: bytes) -> None:
    f = tmp_path / "t.txt"
    f.write_bytes(data)
    assert _read_user_text(f, "task file") == TEXT


def test_read_user_text_rejects_non_utf8_with_clear_message(tmp_path: Path, capsys) -> None:
    f = tmp_path / "ansi.txt"
    f.write_bytes(TEXT.encode("cp1252"))
    with pytest.raises(typer.Exit):
        _read_user_text(f, "task file")
    err = " ".join(capsys.readouterr().err.split())  # undo Rich line wrapping
    assert "Save it as UTF-8" in err


def test_all_three_options_use_it(tmp_path: Path) -> None:
    f = tmp_path / "p.txt"
    f.write_bytes(TEXT.encode("utf-16"))
    assert _resolve_task(None, f) == TEXT
    assert _resolve_system_prompt(None, f) == TEXT
    assert _resolve_auto_log_prompt(str(f))[1] == TEXT
