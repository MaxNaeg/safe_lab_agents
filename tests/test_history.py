"""Tests for conversation history storage and display."""

from __future__ import annotations

from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from safe_lab_agents.agents.base import ConversationEntry, parse_iso_timestamp
from safe_lab_agents.history.display import display_history
from safe_lab_agents.history.store import HistoryStore


class TestParseIsoTimestamp:
    def test_zoned_is_aware(self) -> None:
        dt = parse_iso_timestamp("2026-04-13T10:00:00Z")
        assert dt.tzinfo is not None
        assert dt == datetime(2026, 4, 13, 10, 0, 0, tzinfo=timezone.utc)

    def test_naive_is_stamped_utc(self) -> None:
        dt = parse_iso_timestamp("2026-04-13T10:00:00")  # no zone
        assert dt.tzinfo is timezone.utc
        assert dt == datetime(2026, 4, 13, 10, 0, 0, tzinfo=timezone.utc)

    def test_empty_falls_back_to_aware_now(self) -> None:
        assert parse_iso_timestamp("").tzinfo is not None

    def test_garbage_falls_back_to_aware_now(self) -> None:
        assert parse_iso_timestamp("not-a-timestamp").tzinfo is not None

    def test_mixed_inputs_are_all_comparable(self) -> None:
        """The whole point: naive, zoned, and fallback values must sort together."""
        dts = [
            parse_iso_timestamp("2026-04-13T10:00:05"),   # naive
            parse_iso_timestamp("2026-04-13T10:00:00Z"),  # zoned
            parse_iso_timestamp(""),                       # fallback now()
        ]
        assert sorted(dts)  # no TypeError comparing naive vs aware


@pytest.fixture()
def sample_entries() -> list[ConversationEntry]:
    """A small set of sample conversation entries."""
    return [
        ConversationEntry(
            timestamp=datetime(2026, 4, 13, 10, 0, 0),
            role="user",
            content="Read the temperature.",
        ),
        ConversationEntry(
            timestamp=datetime(2026, 4, 13, 10, 0, 1),
            role="tool_use",
            content="Calling tool: read_temperature",
            tool_name="read_temperature",
            tool_input={},
        ),
        ConversationEntry(
            timestamp=datetime(2026, 4, 13, 10, 0, 2),
            role="tool_result",
            content="22.5",
            tool_name="read_temperature",
            tool_output="22.5",
        ),
        ConversationEntry(
            timestamp=datetime(2026, 4, 13, 10, 0, 3),
            role="assistant",
            content="The current temperature is **22.5 °C**.",
        ),
    ]


class TestHistoryStore:
    """Tests for :class:`HistoryStore`."""

    def test_save_and_load(self, tmp_path: Path, sample_entries, monkeypatch) -> None:
        """Entries round-trip through save/load."""
        monkeypatch.setattr(
            "safe_lab_agents.history.store.get_sessions_dir",
            lambda: tmp_path,
        )
        store = HistoryStore("test-session")
        store.save_entries(sample_entries)

        loaded = store.load_history()
        assert len(loaded) == len(sample_entries)
        assert loaded[0].role == "user"
        assert loaded[0].content == "Read the temperature."
        assert loaded[2].tool_name == "read_temperature"

    def test_append_entry(self, tmp_path: Path, sample_entries, monkeypatch) -> None:
        """append_entry adds to existing history."""
        monkeypatch.setattr(
            "safe_lab_agents.history.store.get_sessions_dir",
            lambda: tmp_path,
        )
        store = HistoryStore("test-session")
        store.save_entries(sample_entries[:2])
        store.append_entry(sample_entries[2])

        loaded = store.load_history()
        assert len(loaded) == 3

    def test_load_empty(self, tmp_path: Path, monkeypatch) -> None:
        """Loading a session with no history returns an empty list."""
        monkeypatch.setattr(
            "safe_lab_agents.history.store.get_sessions_dir",
            lambda: tmp_path,
        )
        store = HistoryStore("empty-session")
        assert store.load_history() == []


class TestDisplayHistory:
    """Tests for :func:`display_history`."""

    def test_renders_without_error(self, sample_entries) -> None:
        """display_history runs without exceptions."""
        buf = StringIO()
        test_console = Console(file=buf, force_terminal=True)
        display_history(sample_entries, console=test_console)
        output = buf.getvalue()
        assert "USER" in output
        assert "ASSISTANT" in output

    def test_empty_history(self) -> None:
        """Empty history prints a message."""
        buf = StringIO()
        test_console = Console(file=buf, force_terminal=True)
        display_history([], console=test_console)
        assert "No conversation history" in buf.getvalue()

    def test_limit(self, sample_entries) -> None:
        """The limit parameter restricts displayed entries."""
        buf = StringIO()
        test_console = Console(file=buf, force_terminal=True, no_color=True)
        display_history(sample_entries, limit=1, console=test_console)
        output = buf.getvalue()
        assert "1" in output and "entries" in output
