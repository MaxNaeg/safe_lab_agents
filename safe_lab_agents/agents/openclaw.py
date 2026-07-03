"""OpenClaw agent backend.

Runs the OpenClaw autonomous agent framework inside a Docker container.
The entrypoint script configures the MCP connection and launches OpenClaw
in either interactive (``openclaw tui --local``) or autonomous
(``openclaw agent --local``) mode.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from rich.markup import escape

from safe_lab_agents.agents.base import (
    AgentArg,
    BaseAgent,
    ConversationEntry,
    register_agent,
)
from safe_lab_agents.config import SessionConfig

logger = logging.getLogger(__name__)

_PROVIDER_KEY_MAP = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}


@register_agent("openclaw")
class OpenClawAgent(BaseAgent):
    """Agent backend for the OpenClaw autonomous agent framework."""

    def get_agent_type(self) -> str:
        return "openclaw"

    def get_dockerfile_name(self) -> str:
        return "Dockerfile.openclaw"

    def get_agent_args(self) -> list[AgentArg]:
        return [
            AgentArg(
                "api-key",
                str,
                required=True,
                required_for_autonomous=True,
                description="API key for the LLM provider.",
                is_secret=True,
            ),
            AgentArg(
                "provider",
                str,
                required=True,
                required_for_autonomous=True,
                description="LLM provider (anthropic, openai, google, openrouter).",
                choices=["anthropic", "openai", "google", "openrouter"],
            ),
            AgentArg(
                "model",
                str,
                required=True,
                required_for_autonomous=True,
                description="Model name (e.g. gpt-4o, claude-sonnet-4-6).",
            ),
        ]

    def get_environment_variables(self, config: SessionConfig, mcp_port: int) -> dict[str, str]:
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

        api_key: str = config.agent_args.get("api-key", "")
        provider: str = config.agent_args.get("provider", "")

        if api_key:
            env["LLM_API_KEY"] = api_key
            provider_env_key = _PROVIDER_KEY_MAP.get(provider)
            if provider_env_key:
                env[provider_env_key] = api_key

        if provider:
            env["LLM_PROVIDER"] = provider

        model: str = config.agent_args.get("model", "")
        if model:
            env["LLM_MODEL"] = f"{provider}/{model}" if provider else model

        return env

    def get_entrypoint_command(self) -> list[str]:
        return ["/entrypoint.sh"]

    def get_system_prompt(self, config: SessionConfig) -> str:
        """Environment prompt for OpenClaw (written to system_prompt.txt by the host).

        Adds OpenClaw-specific guidance to prefer the MCP tools, and uses
        OpenClaw's own (soft) ``--no-web`` wording, since it has no CLI-level
        web block.  See :meth:`BaseAgent.get_system_prompt`.
        """
        parts = [
            "You are an AI agent controlling a scientific experiment.",
            "Your workspace is /agent/workspace/ — use it for scripts and analysis files.",
            "You have access to experiment control tools via the MCP server named "
            "'experiment-tools'. Always prefer these tools over shell workarounds when "
            "controlling instruments or retrieving data.",
        ]
        if config.context_dir is not None:
            parts.append(
                "Experiment context files are at /agent/context/ (read-only). Read them "
                "to understand the experiment before taking action."
            )
        parts.append(
            "Large data exchange happens via /agent/shared/ (read-write). Instruments "
            "write data there; you can read it and also write results there."
        )
        if config.no_web:
            parts.append(
                "RESTRICTION: Do NOT use any web search or web fetch tools under any "
                "circumstances. All internet access is forbidden for this session."
            )
        return "\n".join(parts)

    def get_resume_command(self) -> list[str]:
        return ["--resume"]

    def parse_conversation_history(self, log_dir: Path) -> list[ConversationEntry]:
        """Parse OpenClaw session JSONL logs.

        After ``copy_agent_logs`` the layout is:
        ``log_dir/.openclaw/agents/<id>/sessions/<session-id>.jsonl``.
        """
        entries: list[ConversationEntry] = []

        openclaw_dir = log_dir / ".openclaw"
        if not openclaw_dir.exists():
            logger.info("No OpenClaw history directory found under %s", log_dir)
            return entries

        for jsonl_file in sorted(self._find_session_logs(openclaw_dir)):
            entries.extend(self._parse_jsonl(jsonl_file))

        entries.sort(key=lambda e: e.timestamp)
        return entries

    # Directory basenames belonging to OpenClaw's bundled plugin/npm install
    # trees, never to session logs. Pruning them is a correctness fix on
    # Windows: paths like
    # ``agents/main/agent/codex-home/.tmp/plugins/plugins/.../skills/.../scripts``
    # exceed the 260-char MAX_PATH, so a naive ``rglob("*.jsonl")`` raises
    # ``WinError 3`` (path not found) and aborts the whole history import.
    _PRUNE_DIRS = frozenset({"codex-home", "node_modules", ".tmp", "plugins", "npm"})

    def _find_session_logs(self, openclaw_dir: Path) -> list[Path]:
        """Collect ``*.jsonl`` session logs under ``openclaw_dir``.

        Walks manually so we can prune OpenClaw's bundled plugin/npm trees
        (see :attr:`_PRUNE_DIRS`) — keeping the traversal off the deeply
        nested paths that overflow Windows' MAX_PATH — and tolerate a
        per-entry ``OSError`` instead of failing the entire import.
        """
        logs: list[Path] = []
        for root, dirs, files in os.walk(openclaw_dir, onerror=lambda _e: None):
            dirs[:] = [d for d in dirs if d not in self._PRUNE_DIRS]
            logs.extend(Path(root) / name for name in files if name.endswith(".jsonl"))
        return logs

    def _parse_jsonl(self, path: Path) -> list[ConversationEntry]:
        entries: list[ConversationEntry] = []
        try:
            for raw in path.read_text(encoding="utf-8").splitlines():
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                if not isinstance(data, dict) or data.get("type") != "message":
                    continue

                msg = data.get("message", {})
                role = msg.get("role", "")

                timestamp_str = data.get("timestamp", "")
                try:
                    ts = (
                        datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                        if timestamp_str
                        else datetime.now(tz=timezone.utc)
                    )
                except (ValueError, TypeError):
                    ts = datetime.now(tz=timezone.utc)

                if role == "user":
                    content = _clean_user_message(_extract_text(msg.get("content", [])))
                    if content:
                        entries.append(ConversationEntry(timestamp=ts, role="user", content=content))

                elif role == "assistant":
                    content_blocks = msg.get("content", [])
                    text_parts: list[str] = []
                    for block in content_blocks if isinstance(content_blocks, list) else []:
                        if not isinstance(block, dict):
                            continue
                        btype = block.get("type", "")
                        if btype == "text":
                            text = block.get("text", "").strip()
                            if text:
                                text_parts.append(text)
                        elif btype == "toolCall":
                            if text_parts:
                                entries.append(ConversationEntry(timestamp=ts, role="assistant", content="\n".join(text_parts)))
                                text_parts = []
                            raw_name = block.get("name", "?")
                            tool_name = _strip_server_prefix(raw_name)
                            args = block.get("arguments") or block.get("input") or {}
                            if isinstance(args, str):
                                try:
                                    args = json.loads(args)
                                except (json.JSONDecodeError, TypeError):
                                    args = {}
                            entries.append(ConversationEntry(
                                timestamp=ts,
                                role="tool_use",
                                content=f"Calling tool: {tool_name}",
                                tool_name=tool_name,
                                tool_input=args if isinstance(args, dict) else {},
                                metadata={"tool_call_id": block.get("id", ""), "raw_name": raw_name},
                            ))
                    if text_parts:
                        entries.append(ConversationEntry(timestamp=ts, role="assistant", content="\n".join(text_parts)))

                elif role == "toolResult":
                    raw_name = msg.get("toolName", "?")
                    tool_name = _strip_server_prefix(raw_name)
                    output = _extract_tool_result_text(msg.get("content", [])).strip()
                    if output:
                        entries.append(ConversationEntry(
                            timestamp=ts,
                            role="tool_result",
                            content=output,
                            tool_name=tool_name,
                            tool_output=output,
                            metadata={"is_error": msg.get("isError", False)},
                        ))

        except OSError as exc:
            logger.warning("Could not read %s: %s", path, exc)
        return entries

    def format_autonomous_line(self, line: str) -> Optional[str]:
        """Format a line streamed from the tailed session JSONL file."""
        stripped = line.strip()
        if not stripped:
            return None

        # Entrypoint debug markers
        if stripped.startswith("OPENCLAW_JSONL:"):
            path = stripped[len("OPENCLAW_JSONL:"):].strip()
            if path == "not found":
                return "[dim yellow]⚠ session JSONL not found — showing raw output[/dim yellow]"
            return f"[dim]▶ streaming {escape(path)}[/dim]"
        if stripped.startswith("OPENCLAW_DEBUG:"):
            info = stripped[len("OPENCLAW_DEBUG:"):].strip()
            return f"[dim yellow]debug: {escape(info)}[/dim yellow]"

        # Filter openclaw config/startup noise that leaks into stdout before
        # the agent process takes over.
        if (
            stripped.startswith("Saved MCP server")
            or stripped.startswith("Updated ")
            or stripped.startswith("Config overwrite:")
            or "Restart the gateway to apply." in stripped
        ):
            return None

        # Parse session JSONL records
        if stripped.startswith("{"):
            try:
                data = json.loads(stripped)
                if not isinstance(data, dict):
                    return escape(stripped)

                record_type = data.get("type", "")

                # Skip streaming/partial records — openclaw emits both an
                # intermediate "streaming" update and a final "committed"
                # record; only show the committed one to avoid duplicates.
                if data.get("status") == "streaming":
                    return None

                if record_type == "message":
                    msg = data.get("message", {})
                    role = msg.get("role", "")
                    if role == "assistant":
                        return self._format_assistant_message(msg)
                    if role == "toolResult":
                        return _format_tool_result(msg)
                    # user messages suppressed
                    return None

                # All other record types (session, model_change, …) suppressed.
                return None
            except json.JSONDecodeError:
                pass

        return escape(stripped)

    def _format_assistant_message(self, msg: dict) -> Optional[str]:
        content = msg.get("content", [])
        if isinstance(content, str):
            text = content.strip()
            return escape(text) if text else None

        parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                text = block.get("text", "").strip()
                if text:
                    parts.append(escape(text))
            elif btype == "toolCall":
                # openclaw uses "toolCall" (camelCase), not "tool_use"
                parts.append(_format_tool_block(block))
            # thinking / thinkingSignature blocks intentionally skipped
        return "\n".join(parts) if parts else None


# OpenClaw frames each incoming user message with a "Sender (untrusted
# metadata)" JSON preamble and a "[<weekday> <date> <time> UTC]" prefix.  Strip
# both so the conversation reads as what the user actually typed.
_SENDER_META_RE = re.compile(
    r"^Sender \(untrusted metadata\):\s*```json\s*\{.*?\}\s*```\s*",
    re.DOTALL,
)
_MSG_TIMESTAMP_RE = re.compile(r"^\[[^\]]*\d{4}[^\]]*\]\s*")


def _clean_user_message(text: str) -> str:
    """Strip OpenClaw's sender-metadata preamble and ``[timestamp]`` prefix."""
    cleaned = _SENDER_META_RE.sub("", text, count=1)
    cleaned = _MSG_TIMESTAMP_RE.sub("", cleaned, count=1)
    cleaned = cleaned.strip()
    return cleaned or text


def _strip_server_prefix(name: str) -> str:
    """Remove MCP server prefix from tool names (e.g. 'experiment-tools__foo' → 'foo')."""
    return name.split("__", 1)[-1] if "__" in name else name


def _format_tool_block(block: dict) -> str:
    """Format a toolCall content block as a Rich cyan ⚙ line.

    openclaw uses {"type":"toolCall","name":"server__tool","arguments":{...}}.
    """
    raw_name = block.get("name", "?")
    name = escape(_strip_server_prefix(raw_name))
    inp = block.get("arguments") or block.get("input") or {}
    if isinstance(inp, str):
        try:
            inp = json.loads(inp)
        except (json.JSONDecodeError, TypeError):
            inp = {}

    kv_parts = []
    if isinstance(inp, dict):
        for k, v in list(inp.items())[:3]:
            v_str = str(v)
            if len(v_str) > 40:
                v_str = v_str[:40] + "…"
            kv_parts.append(f"{escape(k)}={escape(v_str)}")
    return f"[dim cyan]  ⚙ {name}({', '.join(kv_parts)})[/dim cyan]"


def _format_tool_result(msg: dict) -> Optional[str]:
    """Format a toolResult message record.

    Format: {"role":"toolResult","toolName":"server__tool","content":[...],"isError":bool}
    """
    raw_name = msg.get("toolName", "?")
    name = escape(_strip_server_prefix(raw_name))
    is_error = msg.get("isError", False)

    content = msg.get("content", [])
    if isinstance(content, str):
        value = content.strip()
    elif isinstance(content, list):
        parts = [
            b["text"] for b in content
            if isinstance(b, dict) and b.get("type") == "text" and "text" in b
        ]
        value = " ".join(parts).strip()
    else:
        value = str(content)

    if not value:
        return None

    if is_error:
        return f"[red]  ✗ {name} → {escape(value)}[/red]"
    return f"[dim cyan]  ← {name} = {escape(value)}[/dim cyan]"


def _extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            block["text"]
            for block in content
            if isinstance(block, dict)
            and block.get("type") == "text"
            and "text" in block
        ]
        return "\n".join(parts)
    return str(content)


def _extract_tool_result_text(content) -> str:
    """Pull readable text out of an OpenClaw ``toolResult`` message content.

    Unlike user/assistant text, tool-result blocks are shaped
    ``{"type": "toolResult", "content": "<json>"}`` where the string is itself
    an MCP result envelope
    (``{"content": [{"type": "text", "text": ...}], "structuredContent": {...}}``).
    Plain ``type: "text"`` blocks are still honoured; anything unrecognised
    falls back to its rawest string form so output is never silently dropped.
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)

    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text" and "text" in block:
            parts.append(str(block["text"]))
            continue
        inner = block.get("content")
        if isinstance(inner, str):
            parts.append(_unwrap_mcp_envelope(inner))
        elif inner is not None:
            parts.append(_extract_text(inner) or str(inner))
    return "\n".join(p for p in parts if p)


def _unwrap_mcp_envelope(raw: str) -> str:
    """Return the human-readable text inside a stringified MCP result envelope.

    Falls back to the raw string when it is not JSON or lacks a text payload.
    """
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw
    if isinstance(data, dict) and data.get("content") is not None:
        text = _extract_text(data["content"])
        if text:
            return text
    return raw
