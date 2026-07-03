"""Tests for the self-contained HTML conversation viewer."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from safe_lab_agents.agents.base import ConversationEntry
from safe_lab_agents.config import SessionConfig, SessionMetadata
from safe_lab_agents.history.html import build_conversation_html


def _ts(second: int = 0) -> datetime:
    return datetime(2026, 6, 26, 12, 0, second, tzinfo=timezone.utc)


def _sample_entries() -> list[ConversationEntry]:
    return [
        ConversationEntry(timestamp=_ts(0), role="user", content="Please measure the power."),
        ConversationEntry(
            timestamp=_ts(1),
            role="assistant",
            content="# Plan\n\nI will **measure** now.\n\n```python\nx = 1\n```",
        ),
        ConversationEntry(
            timestamp=_ts(2),
            role="tool_use",
            content="Calling tool: measure",
            tool_name="measure",
            tool_input={"channel": 1, "code": "import time\nx = 2"},
        ),
        ConversationEntry(
            timestamp=_ts(3),
            role="tool_result",
            content="",
            tool_name="measure",
            tool_output="3.14 mW",
        ),
        ConversationEntry(
            timestamp=_ts(4),
            role="tool_result",
            content="",
            tool_name="broken",
            tool_output="boom",
            metadata={"is_error": True},
        ),
        ConversationEntry(timestamp=_ts(5), role="system", content="Session ended."),
    ]


def test_renders_every_role_self_contained(tmp_path: Path) -> None:
    out = tmp_path / "conversation.html"
    result = build_conversation_html(_sample_entries(), None, out)

    assert result == out
    html = out.read_text(encoding="utf-8")

    # Self-contained: no external asset references.
    assert "http://" not in html
    assert "https://" not in html

    # Each role appears as a filter checkbox and a card.
    for role in ("user", "assistant", "tool_use", "tool_result", "system"):
        assert f"value='{role}'" in html
        assert f"data-kind='{role}'" in html

    # Tool names surface in tool entries.
    assert "measure" in html
    assert "broken" in html


def test_assistant_markdown_becomes_html(tmp_path: Path) -> None:
    out = tmp_path / "c.html"
    build_conversation_html(_sample_entries(), None, out)
    html = out.read_text(encoding="utf-8")

    assert "<h1>Plan</h1>" in html
    assert "<strong>measure</strong>" in html
    assert "<pre>" in html  # fenced code block rendered
    # The assistant card body is wrapped in the markdown container.
    assert "<div class='text md'>" in html


def test_error_tool_result_gets_accent(tmp_path: Path) -> None:
    out = tmp_path / "c.html"
    build_conversation_html(_sample_entries(), None, out)
    html = out.read_text(encoding="utf-8")
    assert "class='card error'" in html


def test_long_text_output_truncated(tmp_path: Path) -> None:
    big = "x" * 5000
    entries = [
        ConversationEntry(
            timestamp=_ts(0), role="tool_result", content="", tool_name="t", tool_output=big
        ),
    ]
    out = tmp_path / "c.html"
    build_conversation_html(entries, None, out)
    html = out.read_text(encoding="utf-8")

    assert "(truncated)" in html
    assert "x" * 5000 not in html


def test_image_output_inlined(tmp_path: Path) -> None:
    data = "A" * 4000
    img = (
        "{'type': 'image', 'source': {'type': 'base64', "
        f"'data': '{data}', 'media_type': 'image/png'}}}}"
    )
    entries = [
        ConversationEntry(
            timestamp=_ts(0), role="tool_result", content="", tool_name="read_img", tool_output=img
        ),
    ]
    out = tmp_path / "c.html"
    build_conversation_html(entries, None, out)
    html = out.read_text(encoding="utf-8")

    # The agent-read image is inlined as a data URI, not summarized.
    assert f"data:image/png;base64,{data}" in html
    assert "<img class='figure'" in html
    assert "[image" not in html


def test_empty_entries_writes_placeholder(tmp_path: Path) -> None:
    out = tmp_path / "c.html"
    build_conversation_html([], None, out)
    html = out.read_text(encoding="utf-8")
    assert "No conversation history found." in html


def test_blank_entries_are_skipped(tmp_path: Path) -> None:
    entries = [
        ConversationEntry(timestamp=_ts(0), role="user", content="   "),
        ConversationEntry(timestamp=_ts(1), role="user", content="real message"),
    ]
    out = tmp_path / "c.html"
    build_conversation_html(entries, None, out)
    html = out.read_text(encoding="utf-8")
    assert "(1 entries)" in html
    assert "real message" in html


def test_session_header_rendered_from_metadata(tmp_path: Path) -> None:
    cfg = SessionConfig(
        name="run-42",
        tools_file=tmp_path / "tools.py",
        workspace_dir=tmp_path / "ws",
        task="characterize the laser",
    )
    metadata = SessionMetadata(config=cfg, status="stopped")
    out = tmp_path / "c.html"
    build_conversation_html(_sample_entries(), metadata, out)
    html = out.read_text(encoding="utf-8")

    assert "Session info" in html
    assert "run-42" in html
    assert "characterize the laser" in html
    assert "stopped" in html
