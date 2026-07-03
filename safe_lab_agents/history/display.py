"""Rich-based pretty-printing for conversation history."""

from __future__ import annotations

import json
import re
from typing import Optional

from rich.console import Console, RenderableType
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from safe_lab_agents.agents.base import ConversationEntry
from safe_lab_agents.config import SessionMetadata

_ROLE_STYLES = {
    "user": ("bold blue", "blue"),
    "assistant": ("bold green", "green"),
    "tool_use": ("bold yellow", "yellow"),
    "tool_result": ("bold cyan", "cyan"),
    "system": ("bold dim", "dim"),
}


def display_history(
    entries: list[ConversationEntry],
    metadata: Optional[SessionMetadata] = None,
    limit: Optional[int] = None,
    console: Optional[Console] = None,
) -> None:
    console = console or Console(stderr=True)

    if metadata is not None:
        _print_session_header(console, metadata)

    entries = [e for e in entries if e.content.strip() or e.tool_input or e.tool_output]

    if not entries:
        console.print("[dim]No conversation history found.[/dim]")
        return

    if limit is not None:
        entries = entries[-limit:]

    for entry in entries:
        _print_entry(console, entry)

    console.print()
    console.print(f"[dim]Showing {len(entries)} entries.[/dim]")


def print_history(
    session_name: str,
    last: Optional[int] = None,
    console: Optional[Console] = None,
) -> None:
    from safe_lab_agents.history.store import HistoryStore

    console = console or Console(stderr=True)

    metadata: Optional[SessionMetadata] = None
    try:
        metadata = SessionMetadata.load(session_name)
    except FileNotFoundError:
        pass

    store = HistoryStore(session_name)
    entries = store.load_history()
    display_history(entries, metadata=metadata, limit=last, console=console)


# ------------------------------------------------------------------
# Internal rendering helpers
# ------------------------------------------------------------------


def _print_session_header(console: Console, metadata: SessionMetadata) -> None:
    cfg = metadata.config
    table = Table(title="Session Info", show_header=False, expand=False)
    table.add_column("Key", style="bold")
    table.add_column("Value")

    table.add_row("Session", cfg.name)
    table.add_row("Agent", cfg.agent_type)
    table.add_row("Container", cfg.container_runtime)
    table.add_row("Status", metadata.status)
    table.add_row("Created", cfg.created_at.strftime("%Y-%m-%d %H:%M:%S"))
    if metadata.started_at:
        table.add_row("Started", metadata.started_at.strftime("%Y-%m-%d %H:%M:%S"))
    if metadata.stopped_at:
        table.add_row("Stopped", metadata.stopped_at.strftime("%Y-%m-%d %H:%M:%S"))
    table.add_row("Tools file", str(cfg.tools_file))
    if cfg.task:
        table.add_row("Task", cfg.task)

    console.print(table)
    console.print()


def _print_entry(console: Console, entry: ConversationEntry) -> None:
    title_style, border_style = _ROLE_STYLES.get(entry.role, ("bold", "white"))
    role_label = entry.role.upper().replace("_", " ")
    timestamp = entry.timestamp.strftime("%H:%M:%S")

    if entry.role in ("tool_use", "tool_result") and entry.tool_name:
        title = f"{role_label}: {entry.tool_name} [{timestamp}]"
    else:
        title = f"{role_label} [{timestamp}]"

    body: RenderableType
    if entry.role == "assistant":
        body = Markdown(entry.content)
    elif entry.role == "tool_use":
        body = _render_tool_input(entry.tool_input)
    elif entry.role == "tool_result":
        body = _render_tool_output(entry.tool_output or entry.content)
    else:
        body = Text(entry.content)

    panel = Panel(
        body,
        title=title,
        title_align="left",
        border_style=border_style,
        expand=True,
        padding=(0, 1),
    )
    console.print(panel)


def _render_tool_input(tool_input: Optional[dict]) -> RenderableType:
    """Render tool input arguments as a labelled key/value table."""
    if not tool_input:
        return Text("(no arguments)", style="dim italic")

    table = Table(show_header=False, box=None, padding=(0, 1, 0, 0), expand=True)
    table.add_column("key", style="bold dim", justify="left", no_wrap=True)
    table.add_column("value")

    for key, value in tool_input.items():
        if isinstance(value, str) and "\n" in value:
            lang = _guess_language(key, value)
            table.add_row(key, Syntax(value.strip(), lang, theme="ansi_dark", line_numbers=False, word_wrap=False))
        elif isinstance(value, (dict, list)):
            table.add_row(key, Text(json.dumps(value, indent=2)))
        else:
            table.add_row(key, Text(str(value)))

    return table


def _render_tool_output(content: str) -> RenderableType:
    """Render tool output, with special handling for image data."""
    if _is_image_content(content):
        m = re.search(r"['\"]data['\"]\s*:\s*['\"]([A-Za-z0-9+/]{20,})", content)
        kb = len(m.group(1)) * 3 // 4 // 1024 if m else 0
        size_str = f"~{kb} KB" if kb else "binary"
        return Text(f"[image {size_str}]", style="dim italic")

    if len(content) > 2000:
        content = content[:2000] + "\n… (truncated)"
    return Text(content)


def _is_image_content(content: str) -> bool:
    return bool(re.search(r"['\"]type['\"]\s*:\s*['\"]image['\"]", content[:200]))


def _guess_language(key: str, value: str) -> str:
    if key == "command":
        return "bash"
    if key in ("code", "script", "source"):
        return "python"
    stripped = value.strip()
    if stripped.startswith(("import ", "from ", "def ", "class ", "#!/usr/bin/env python")):
        return "python"
    return "text"
