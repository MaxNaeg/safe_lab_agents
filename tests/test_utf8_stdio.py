"""Redirected stdout on Windows defaults to cp1252; the CLI must not crash on the
non-ASCII status glyphs it prints (▶ ⚠ ✓ ✗)."""

from __future__ import annotations

import io
import os
import sys

import pytest

from safe_lab_agents.cli import _ensure_utf8_stdio

GLYPHS = "▶ ⚠ ✓ ✗"


def _stream(encoding: str) -> io.TextIOWrapper:
    return io.TextIOWrapper(io.BytesIO(), encoding=encoding)


def test_cp1252_stdout_is_switched_to_utf8(monkeypatch) -> None:
    out, err = _stream("cp1252"), _stream("cp1252")
    with pytest.raises(UnicodeEncodeError):  # the bug this guards against
        _stream("cp1252").write(GLYPHS)
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    monkeypatch.delenv("PYTHONIOENCODING", raising=False)

    _ensure_utf8_stdio()

    out.write(GLYPHS)
    out.flush()
    assert out.buffer.getvalue().decode("utf-8") == GLYPHS
    assert err.encoding == "utf-8"
    # Exported so the spawned MCP server subprocess gets UTF-8 too.
    assert os.environ["PYTHONIOENCODING"] == "utf-8"


def test_utf8_stdout_and_user_setting_are_left_alone(monkeypatch) -> None:
    out = _stream("UTF-8")
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setenv("PYTHONIOENCODING", "utf-8:strict")

    _ensure_utf8_stdio()

    assert sys.stdout is out and out.errors == "strict"
    assert os.environ["PYTHONIOENCODING"] == "utf-8:strict"


def test_streams_without_reconfigure_are_skipped(monkeypatch) -> None:
    class Plain:
        encoding = "cp1252"

    monkeypatch.setattr(sys, "stdout", Plain())
    _ensure_utf8_stdio()  # must not raise
