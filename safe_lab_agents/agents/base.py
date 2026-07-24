"""Abstract base class and registry for agent backends.

Every supported agent (Claude Code, OpenClaw, etc.) subclasses
:class:`BaseAgent` and registers itself with ``@register_agent("name")``.
The CLI looks up agents by name and delegates all agent-specific behaviour
to the concrete class.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from safe_lab_agents.config import SessionConfig

# ---------------------------------------------------------------------------
# Agent arg descriptor
# ---------------------------------------------------------------------------


@dataclass
class AgentArg:
    """Descriptor for an agent-specific CLI argument."""

    name: str
    type: type
    required: bool
    required_for_autonomous: bool
    description: str
    default: Any = None
    is_secret: bool = False
    choices: list[str] | None = None


# ---------------------------------------------------------------------------
# Agent registry
# ---------------------------------------------------------------------------

AGENT_REGISTRY: dict[str, type[BaseAgent]] = {}


def register_agent(name: str):
    """Class decorator that registers a :class:`BaseAgent` subclass.

    Args:
        name: Short identifier used on the CLI (e.g. ``"claude-code"``).
    """

    def decorator(cls: type[BaseAgent]) -> type[BaseAgent]:
        AGENT_REGISTRY[name] = cls
        return cls

    return decorator


def get_agent(name: str) -> BaseAgent:
    """Instantiate and return the agent registered under *name*.

    Raises:
        ValueError: If no agent with that name has been registered.
    """
    if name not in AGENT_REGISTRY:
        available = ", ".join(sorted(AGENT_REGISTRY)) or "(none)"
        raise ValueError(f"Unknown agent '{name}'. Available: {available}")
    return AGENT_REGISTRY[name]()


def list_agents() -> list[str]:
    """Return the names of all registered agent backends."""
    return sorted(AGENT_REGISTRY)


# ---------------------------------------------------------------------------
# Conversation history data model
# ---------------------------------------------------------------------------


@dataclass
class ConversationEntry:
    """A single entry in the conversation history."""

    timestamp: datetime
    role: str  # "user", "assistant", "tool_use", "tool_result", "system"
    content: str
    tool_name: Optional[str] = None
    tool_input: Optional[dict] = None
    tool_output: Optional[str] = None
    metadata: dict = field(default_factory=dict)


def parse_iso_timestamp(timestamp_str: str) -> datetime:
    """Parse an ISO-8601 timestamp into a **timezone-aware** UTC datetime.

    Log lines may carry a zoned timestamp (``...Z`` / ``+00:00``), a *naive*
    one (no offset), or none at all.  Every parsed timestamp is normalized to
    aware-UTC — naive values are assumed to be UTC and stamped accordingly — so
    that mixing sources never yields a naive/aware mix.  Such a mix makes
    ``sorted(entries, key=lambda e: e.timestamp)`` raise ``TypeError: can't
    compare offset-naive and offset-aware datetimes`` and abort the whole
    history import.  Empty or unparseable input falls back to ``now`` in UTC.
    """
    try:
        dt = (
            datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            if timestamp_str
            else datetime.now(tz=timezone.utc)
        )
    except (ValueError, TypeError, AttributeError):
        dt = datetime.now(tz=timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class BaseAgent(ABC):
    """Abstract interface that every agent backend must implement.

    The :class:`DockerManager` and CLI delegate all agent-specific decisions
    (Dockerfile choice, environment variables, entrypoint commands, history
    parsing) to the concrete agent subclass.
    """

    @abstractmethod
    def get_agent_type(self) -> str:
        """Return the short identifier for this agent (e.g. ``'claude-code'``)."""

    @abstractmethod
    def get_dockerfile_name(self) -> str:
        """Return the filename of the Dockerfile template (e.g. ``'Dockerfile.claude-code'``)."""

    @abstractmethod
    def get_environment_variables(self, config: SessionConfig, mcp_port: int) -> dict[str, str]:
        """Return environment variables to inject into the container.

        Args:
            config: The current session configuration.
            mcp_port: The TCP port the MCP server is listening on.
        """

    @abstractmethod
    def get_entrypoint_command(self) -> list[str]:
        """Return the container entrypoint command.

        The interactive/autonomous switch is read from the ``MODE`` environment
        variable inside the entrypoint script, so no mode argument is needed.
        """

    @abstractmethod
    def get_resume_command(self) -> list[str]:
        """Return the command to resume a previous conversation."""

    @abstractmethod
    def parse_conversation_history(self, log_dir: Path) -> list[ConversationEntry]:
        """Parse conversation history from the agent's native log format.

        Args:
            log_dir: Directory containing the agent's extracted native logs
                (the session ``logs/`` directory populated by
                :meth:`DockerManager.copy_agent_logs`).  For Claude Code it
                contains a ``projects/`` subdirectory; for OpenClaw a
                ``.openclaw/`` subdirectory.

        Returns:
            Chronologically ordered list of conversation entries.
        """

    def get_login_command(self) -> list[str]:
        """Return the entrypoint command for an interactive login bootstrap.

        Override in agents that support logging in inside a throwaway container
        when the host has no credentials.  Raises by default.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support in-container login."
        )

    # ------------------------------------------------------------------
    # Credential hygiene
    #
    # Secrets must never be baked into the committed session image or written
    # to ``metadata.json``.  These three hooks keep them out: ``pop_secret_env``
    # strips credential agent-args into runtime env at ``start`` (so they never
    # reach metadata); ``get_secret_env_keys`` lists the env keys the host
    # blanks when it commits the container; ``resume_credential_env`` re-obtains
    # any credential that a resumed image therefore no longer carries.
    # ------------------------------------------------------------------

    def get_secret_env_keys(self) -> list[str]:
        """Return the env-var names to blank when committing the container.

        ``docker commit`` bakes the container's environment into the image
        config, so any secret passed as an env var would otherwise be readable
        (via ``docker inspect``/``save``) by anyone with access to the container
        runtime.  The host blanks these keys at commit time.  The per-session
        MCP auth token is always included; subclasses extend the list with their
        provider/credential keys.
        """
        return ["MCP_AUTH_TOKEN"]

    def pop_secret_env(self, config: SessionConfig) -> dict[str, str]:
        """Pop credential agent-args out of ``config.agent_args`` into env vars.

        Called on ``start``.  Popping (rather than reading) keeps the secret out
        of ``metadata.json`` — and therefore out of any later resumed run.  The
        returned mapping is merged into the container environment.  The base
        implementation pops nothing.
        """
        return {}

    def resume_credential_env(self, config: SessionConfig) -> dict[str, str]:
        """Return credential env vars to inject on ``resume``.

        Because secrets are never persisted, an agent whose credentials cannot
        be recovered from the committed image must re-obtain them here (e.g. by
        prompting the user).  The base implementation requires nothing.
        """
        return {}

    def get_system_prompt(self, config: SessionConfig) -> str:
        """Return the base *environment* system prompt for the agent.

        This is the single source of truth for the prompt prose that describes
        the container environment.  The host writes the result to
        ``/agent/workspace/system_prompt.txt`` and both entrypoint scripts read
        that file, so the text is defined here once instead of being duplicated
        in each entrypoint.  Subclasses override this to add agent-specific
        guidance (see :class:`OpenClawAgent`).

        The entrypoint owns everything that depends on interactive/autonomous/
        resume branching — the autonomous-mode instruction and the appended
        tool-info files — so those are intentionally *not* included here.
        """
        parts = [
            "You are an AI agent controlling a scientific experiment.",
            "Your workspace is /agent/workspace/ — use it for scripts and analysis files.",
        ]
        if config.context_dir is not None:
            parts.append(
                "Experiment context files are at /agent/context/ (read-only). "
                "Read them to understand the experiment before taking action."
            )
        parts.append(
            "Large data exchange happens via /agent/shared/ (read-write). "
            "Instruments write data there; you can read it and also write results there."
        )
        if config.no_web:
            # Prompt-level (soft) restriction. The container keeps network egress
            # (needed for the agent's own model API), so for vectors other than
            # the hard-blocked WebSearch/WebFetch tools this instruction — not a
            # network block — is what discourages web access.
            parts.append(
                "RESTRICTION: All internet and web access is strictly forbidden. Do not "
                "access the web by any means — including built-in tools, shell commands "
                "(curl, wget, etc.), Python libraries, or any other method."
            )
        return "\n".join(parts)

    def get_agent_args(self) -> list[AgentArg]:
        """Return descriptors for agent-specific CLI arguments.

        Override to declare args that users can pass via ``--agent-args``.
        """
        return []

    def format_autonomous_line(self, line: str) -> Optional[str]:
        """Format one line of autonomous container output for display.

        Return the formatted string to print, or ``None`` to suppress the line.
        The default implementation passes non-empty lines through unchanged —
        suitable for agents that emit plain text.  Override for structured
        formats like Claude Code's ``stream-json``.
        """
        return line if line.strip() else None
