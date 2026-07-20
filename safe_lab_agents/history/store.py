"""Conversation history persistence.

Stores and loads conversation entries as JSON files alongside session
metadata.  Supports importing history from agent-native formats (delegated
to the agent's :meth:`~BaseAgent.parse_conversation_history` method) and
capturing ``stream-json`` output from autonomous Claude Code sessions in
real time.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from safe_lab_agents.agents.base import (
    BaseAgent,
    ConversationEntry,
    parse_iso_timestamp,
)
from safe_lab_agents.config import get_sessions_dir

logger = logging.getLogger(__name__)


class HistoryStore:
    """Read and write conversation history for experiment sessions.

    History is stored as a JSON array of :class:`ConversationEntry` objects in
    ``~/.safe_lab_agents/sessions/<name>/history.json``.
    """

    def __init__(self, session_name: str) -> None:
        self.session_name = session_name
        self._dir = get_sessions_dir() / session_name
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / "history.json"

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def save_entries(self, entries: list[ConversationEntry]) -> None:
        """Overwrite the history file with *entries*.

        Args:
            entries: Complete list of conversation entries to persist.
        """
        data = [_entry_to_dict(e) for e in entries]
        self._path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        logger.info("Saved %d history entries to %s", len(entries), self._path)

    def append_entry(self, entry: ConversationEntry) -> None:
        """Append a single entry to the existing history file."""
        entries = self.load_history()
        entries.append(entry)
        self.save_entries(entries)

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def load_history(self) -> list[ConversationEntry]:
        """Load all conversation entries from disk.

        Returns an empty list if no history file exists yet.
        """
        if not self._path.exists():
            return []
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            return [_dict_to_entry(d) for d in data]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning("Could not parse history at %s: %s", self._path, exc)
            return []

    # ------------------------------------------------------------------
    # Importing from agents
    # ------------------------------------------------------------------

    def import_from_agent(self, agent: BaseAgent, log_dir: Path) -> list[ConversationEntry]:
        """Import conversation history using the agent's native parser.

        Delegates to :meth:`~BaseAgent.parse_conversation_history` and merges
        the result with any entries already stored in ``history.json``.

        Args:
            agent: The agent backend instance.
            log_dir: Directory containing the agent's native logs.  After a
                session shuts down this is ``<session_dir>/logs/``; for Claude
                Code it must contain a ``projects/`` subdirectory, for OpenClaw
                a ``.openclaw/`` subdirectory.

        Returns:
            The merged list of entries.
        """
        imported = agent.parse_conversation_history(log_dir)

        if not imported:
            logger.info("No history found to import for session '%s'.", self.session_name)
            return self.load_history()

        imported.sort(key=lambda e: e.timestamp)
        self.save_entries(imported)
        logger.info("Saved %d entries for session '%s'.", len(imported), self.session_name)
        return imported


# ------------------------------------------------------------------
# Serialisation helpers
# ------------------------------------------------------------------


def _entry_to_dict(entry: ConversationEntry) -> dict:
    """Serialize a :class:`ConversationEntry` to a JSON-compatible dict."""
    return {
        "timestamp": entry.timestamp.isoformat(),
        "role": entry.role,
        "content": entry.content,
        "tool_name": entry.tool_name,
        "tool_input": entry.tool_input,
        "tool_output": entry.tool_output,
        "metadata": entry.metadata,
    }


def _dict_to_entry(data: dict) -> ConversationEntry:
    """Deserialize a dict back to a :class:`ConversationEntry`."""
    return ConversationEntry(
        timestamp=parse_iso_timestamp(data.get("timestamp", "")),
        role=data.get("role", "system"),
        content=data.get("content", ""),
        tool_name=data.get("tool_name"),
        tool_input=data.get("tool_input"),
        tool_output=data.get("tool_output"),
        metadata=data.get("metadata", {}),
    )
