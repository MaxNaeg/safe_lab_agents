"""Claude Code agent backend.

Runs the ``@anthropic-ai/claude-code`` CLI inside a Docker container.
The entrypoint script configures the MCP connection and launches Claude Code
in either interactive or autonomous mode.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from rich.markup import escape

from safe_lab_agents.agents.base import (
    AgentArg,
    BaseAgent,
    ConversationEntry,
    parse_iso_timestamp,
    register_agent,
)
from safe_lab_agents.config import SessionConfig

logger = logging.getLogger(__name__)


@register_agent("claude-code")
class ClaudeCodeAgent(BaseAgent):
    """Agent backend for Anthropic's Claude Code CLI."""

    def get_agent_type(self) -> str:
        """Return ``'claude-code'``."""
        return "claude-code"

    def get_agent_args(self) -> list[AgentArg]:
        return [
            AgentArg("model", str, False, False,
                     "Claude model alias or full ID (e.g. sonnet, opus, claude-sonnet-4-6).",
                     default=None),
            AgentArg("effort", str, False, False,
                     "Effort level (low/medium/high/xhigh/max).",
                     default="low", choices=["low", "medium", "high", "xhigh", "max"]),
            AgentArg("copy-host-credentials", bool, False, False,
                     "Copy Claude login credentials from the host into the "
                     "container (off by default; without it, log in inside the "
                     "container or pass oauth-token).",
                     default=False),
            AgentArg("oauth-token", str, False, False,
                     "Claude OAuth token (sk-ant-oat…) to authenticate with directly, "
                     "skipping host-credential copy and in-container login. "
                     "Not stored in session metadata.",
                     default=None, is_secret=True),
            AgentArg("dangerously-skip-permissions", bool, False, False,
                     "Pass --dangerously-skip-permissions to Claude Code.",
                     default=False),
        ]

    def get_dockerfile_name(self) -> str:
        """Return the Dockerfile template for Claude Code."""
        return "Dockerfile.claude-code"

    def get_environment_variables(self, config: SessionConfig, mcp_port: int) -> dict[str, str]:
        """Return environment variables for the Claude Code container.

        Claude Code is paid via subscription — no API key is needed.
        Only the MCP port and session configuration are passed.
        """
        env: dict[str, str] = {
            "MCP_PORT": str(mcp_port),
            "MODE": "autonomous" if config.task else "interactive",
        }

        if config.task:
            env["TASK_PROMPT"] = config.task

        if config.context_dir is not None:
            env["CONTEXT_DIR"] = "/agent/context"

        if config.no_web:
            env["NO_WEB"] = "true"

        model = config.agent_args.get("model")
        if model:
            env["CLAUDE_MODEL"] = model

        effort = config.agent_args.get("effort")
        if effort:
            env["CLAUDE_EFFORT"] = str(effort)

        if config.agent_args.get("dangerously-skip-permissions"):
            env["SKIP_PERMISSIONS"] = "true"

        return env

    def get_secret_env_keys(self) -> list[str]:
        """Blank the OAuth credential env vars when committing the container.

        ``CLAUDE_CREDENTIALS_JSON`` (host-copied credentials) and
        ``CLAUDE_CODE_OAUTH_TOKEN`` (a directly-supplied token) are both
        injected as env vars, so ``docker commit`` would otherwise bake them
        into the image config.  The entrypoint additionally scrubs the
        ``~/.claude/.credentials.json`` file it writes, so no credential remains
        in the committed image.
        """
        return super().get_secret_env_keys() + [
            "CLAUDE_CREDENTIALS_JSON",
            "CLAUDE_CODE_OAUTH_TOKEN",
        ]

    def resume_credential_env(self, config: SessionConfig) -> dict[str, str]:
        """Re-inject a directly-supplied OAuth token on resume, if given.

        A token passed via ``--agent-args oauth-token=…`` is re-injected (and
        popped so it is never persisted).  When absent nothing is injected: the
        entrypoint scrubbed ``~/.claude/.credentials.json`` before commit, so the
        resumed (interactive) session re-authenticates via the in-container
        login flow.
        """
        token = config.agent_args.pop("oauth-token", None)
        return {"CLAUDE_CODE_OAUTH_TOKEN": token} if token else {}

    def get_entrypoint_command(self) -> list[str]:
        """Return the entrypoint command.

        The actual logic is in ``entrypoint.claude-code.sh`` which reads the
        ``MODE`` and ``TASK_PROMPT`` environment variables.
        """
        return ["/entrypoint.sh"]

    def get_resume_command(self) -> list[str]:
        """Return the command to resume the most recent conversation."""
        return ["--resume"]

    def get_login_command(self) -> list[str]:
        """Return the entrypoint command that runs an interactive login then exits.

        Used by the autonomous login-bootstrap when the host has no Claude
        credentials: the container runs ``claude setup-token`` over a TTY and
        writes the resulting OAuth token to a file the host harvests.
        """
        return ["--login"]

    def parse_conversation_history(self, log_dir: Path) -> list[ConversationEntry]:
        """Parse Claude Code's JSONL conversation logs.

        Claude Code stores conversation history as JSONL files under
        ``~/.claude/projects/``.  After extraction via
        :meth:`~DockerManager.copy_agent_logs` the layout is
        ``log_dir/projects/<hash>/*.jsonl``.
        """
        entries: list[ConversationEntry] = []

        projects_dir = log_dir / "projects"
        if not projects_dir.exists():
            logger.info("No Claude Code history directory found under %s", log_dir)
            return entries

        # Find all JSONL files
        for jsonl_file in sorted(projects_dir.rglob("*.jsonl")):
            entries.extend(self._parse_jsonl(jsonl_file))

        entries.sort(key=lambda e: e.timestamp)
        return entries

    def _parse_jsonl(self, jsonl_path: Path) -> list[ConversationEntry]:
        """Parse a single JSONL conversation file from Claude Code."""
        entries: list[ConversationEntry] = []
        tool_name_by_id: dict[str, str] = {}
        try:
            for line in jsonl_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    entries.extend(self._jsonl_record_to_entries(data, tool_name_by_id))
                except json.JSONDecodeError:
                    continue
        except OSError as exc:
            logger.warning("Could not read %s: %s", jsonl_path, exc)
        return entries

    @staticmethod
    def _jsonl_record_to_entries(
        data: dict, tool_name_by_id: dict[str, str]
    ) -> list[ConversationEntry]:
        """Convert a single JSONL record to zero or more :class:`ConversationEntry` items.

        Claude Code embeds tool_use blocks inside assistant messages and
        tool_result blocks inside user messages, so one record may produce
        multiple entries.  ``tool_name_by_id`` is updated in-place as
        tool_use blocks are seen so that matching tool_result entries can
        carry the tool name.
        """
        msg_type = data.get("type", "")
        timestamp = parse_iso_timestamp(data.get("timestamp", ""))

        if msg_type in ("human", "user"):
            message = data.get("message", {})
            content = message.get("content", []) if isinstance(message, dict) else message

            if isinstance(content, str):
                return [ConversationEntry(timestamp=timestamp, role="user", content=content)] if content.strip() else []

            entries: list[ConversationEntry] = []
            text_parts: list[str] = []
            for block in content if isinstance(content, list) else []:
                if isinstance(block, str):
                    text_parts.append(block)
                elif isinstance(block, dict):
                    btype = block.get("type", "")
                    if btype == "tool_result":
                        result_content = block.get("content", "")
                        if isinstance(result_content, list):
                            parts: list[str] = []
                            for b in result_content:
                                if not isinstance(b, dict):
                                    parts.append(str(b))
                                elif b.get("type") == "tool_reference":
                                    parts.append(b.get("tool_name", str(b)))
                                elif "text" in b:
                                    parts.append(b["text"])
                                else:
                                    parts.append(str(b))
                            result_content = "\n".join(parts)
                        tool_use_id = block.get("tool_use_id", "")
                        entries.append(ConversationEntry(
                            timestamp=timestamp,
                            role="tool_result",
                            content=str(result_content),
                            tool_name=tool_name_by_id.get(tool_use_id, tool_use_id),
                            tool_output=str(result_content),
                            metadata={"is_error": block.get("is_error", False)},
                        ))
                    elif "text" in block:
                        text_parts.append(block["text"])

            if text_parts:
                entries.insert(0, ConversationEntry(timestamp=timestamp, role="user", content="\n".join(text_parts)))
            return entries

        if msg_type == "assistant":
            message = data.get("message", {})
            content = message.get("content", []) if isinstance(message, dict) else []

            if isinstance(content, str):
                return [ConversationEntry(timestamp=timestamp, role="assistant", content=content)] if content.strip() else []

            entries = []
            text_parts = []
            for block in content if isinstance(content, list) else []:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type", "")
                if btype == "text":
                    text = block.get("text", "").strip()
                    if text:
                        text_parts.append(text)
                elif btype == "tool_use":
                    if text_parts:
                        entries.append(ConversationEntry(timestamp=timestamp, role="assistant", content="\n".join(text_parts)))
                        text_parts = []
                    tool_id = block.get("id", "")
                    tool_name = block.get("name", "?")
                    if tool_id:
                        tool_name_by_id[tool_id] = tool_name
                    tool_input = block.get("input")
                    entries.append(ConversationEntry(
                        timestamp=timestamp,
                        role="tool_use",
                        content=f"Calling tool: {tool_name}",
                        tool_name=tool_name,
                        tool_input=tool_input if isinstance(tool_input, dict) else {},
                        metadata={"tool_use_id": tool_id},
                    ))

            if text_parts:
                entries.append(ConversationEntry(timestamp=timestamp, role="assistant", content="\n".join(text_parts)))
            return entries

        # Skip metadata records (permission-mode, file-history-snapshot, etc.)
        return []

    def format_autonomous_line(self, line: str) -> Optional[str]:
        """Parse a Claude Code ``stream-json`` line and return Rich-formatted text.

        Returns ``None`` for lines that should be suppressed (e.g. metadata,
        blank lines, unrecognised types).
        """
        line = line.strip()
        if not line:
            return None
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            return escape(line)

        if not isinstance(data, dict):
            return escape(line)

        msg_type = data.get("type", "")

        if msg_type == "system" and data.get("subtype") == "init":
            session_id = data.get("session_id", "")[:8]
            model = escape(data.get("model", "?"))
            tools = data.get("tools", [])
            return f"[dim]▶ Session {session_id} — model: {model}, tools: {len(tools)}[/dim]"

        if msg_type == "assistant":
            message = data.get("message", {})
            content = message.get("content", [])
            parts: list[str] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text":
                    text = block.get("text", "").strip()
                    if text:
                        parts.append(escape(text))
                elif block.get("type") == "tool_use":
                    name = escape(block.get("name", "?"))
                    inp = block.get("input") or {}
                    kv_parts = []
                    for k, v in list(inp.items())[:3]:
                        v_str = str(v)
                        if len(v_str) > 40:
                            v_str = v_str[:40] + "…"
                        kv_parts.append(f"{escape(k)}={escape(v_str)}")
                    kv = ", ".join(kv_parts)
                    parts.append(f"[dim cyan]  ⚙ {name}({kv})[/dim cyan]")
            return "\n".join(parts) if parts else None

        if msg_type == "result":
            is_error = data.get("is_error", False)
            result_text = escape(data.get("result", ""))
            if is_error:
                return f"[bold red]✗ Error:[/bold red] {result_text}"
            return "[bold green]✓ Done[/bold green]"

        return None
