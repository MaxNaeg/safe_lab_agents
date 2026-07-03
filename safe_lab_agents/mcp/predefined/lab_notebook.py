"""Predefined MCP server: Lab Notebook.

Provides tools for maintaining a simple Markdown-based lab notebook stored in
the shared directory.  Entries are timestamped and appended to a
``lab_notebook.md`` file.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Callable

from safe_lab_agents.mcp.predefined import PredefinedServer, register_server

# The notebook is written to the shared directory so both the agent and the
# host can access it.
_NOTEBOOK_DIR = Path(os.environ.get("NOTEBOOK_DIR", "/agent/shared"))


def add_entry(title: str, content: str) -> str:
    """Add a new timestamped entry to the lab notebook.

    Args:
        title: A short title for the notebook entry.
        content: The body text of the entry (Markdown is supported).

    Returns:
        Confirmation message with the timestamp.
    """
    notebook = _NOTEBOOK_DIR / "lab_notebook.md"
    timestamp = datetime.now().isoformat(timespec="seconds")
    entry = f"\n## {title}\n\n**Date:** {timestamp}\n\n{content}\n\n---\n"

    if not notebook.exists():
        notebook.write_text(f"# Lab Notebook\n{entry}", encoding="utf-8")
    else:
        with notebook.open("a", encoding="utf-8") as f:
            f.write(entry)

    return f"Entry '{title}' added at {timestamp}."


def search_entries(query: str) -> str:
    """Search the lab notebook for entries containing *query*.

    Args:
        query: Case-insensitive search string.

    Returns:
        Matching sections separated by newlines, or a message if nothing
        was found.
    """
    notebook = _NOTEBOOK_DIR / "lab_notebook.md"
    if not notebook.exists():
        return "Lab notebook is empty."

    text = notebook.read_text(encoding="utf-8")
    sections = text.split("\n## ")
    matches = [s for s in sections if query.lower() in s.lower()]
    if not matches:
        return f"No entries matching '{query}'."
    return "\n## ".join(matches)


def list_entries() -> str:
    """List all entry titles in the lab notebook.

    Returns:
        A newline-separated list of entry titles, or a message if the
        notebook is empty.
    """
    notebook = _NOTEBOOK_DIR / "lab_notebook.md"
    if not notebook.exists():
        return "Lab notebook is empty."

    titles: list[str] = []
    for line in notebook.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            titles.append(line.removeprefix("## ").strip())
    if not titles:
        return "No entries found."
    return "\n".join(f"- {t}" for t in titles)


@register_server("lab-notebook")
class LabNotebookServer(PredefinedServer):
    """Predefined MCP server providing a simple Markdown lab notebook."""

    def get_tools(self) -> list[Callable]:
        """Return the notebook tool functions."""
        return [add_entry, search_entries, list_entries]
