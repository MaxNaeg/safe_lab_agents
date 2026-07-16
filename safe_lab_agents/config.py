"""Configuration models for safe_lab_agents sessions."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


def get_base_dir() -> Path:
    """Return the base directory for safe_lab_agents data.

    Defaults to ``~/.safe_lab_agents``.  The directory is created if it does
    not already exist.
    """
    base = Path.home() / ".safe_lab_agents"
    base.mkdir(parents=True, exist_ok=True)
    return base


def get_sessions_dir() -> Path:
    """Return the directory where session data is stored.

    Creates the directory tree if necessary.
    """
    sessions = get_base_dir() / "sessions"
    sessions.mkdir(parents=True, exist_ok=True)
    return sessions


class SessionConfig(BaseModel):
    """All user-provided configuration for a single experiment session."""

    name: str = Field(description="Unique session name")
    agent_type: str = Field(
        default="claude-code",
        description="Agent backend to use (e.g. 'claude-code', 'openclaw')",
    )
    tools_file: Path = Field(description="Path to the Python file defining MCP tools")
    context_dir: Optional[Path] = Field(
        default=None,
        description="Directory with experiment context (mounted read-only)",
    )
    shared_dir: Optional[Path] = Field(
        default=None,
        description="Shared directory for data exchange (mounted read-write)",
    )
    workspace_dir: Path = Field(
        description="Workspace directory visible to user and agent (auto-created)"
    )
    requirements_file: Optional[Path] = Field(
        default=None,
        description="Path to a requirements.txt for additional Python packages in Docker",
    )
    mcp_port: int = Field(
        default=0,
        description="Port for the MCP server on the host (0 = auto-select)",
    )
    task: Optional[str] = Field(
        default=None,
        description="Initial task for autonomous mode (None = interactive)",
    )
    predefined_servers: list[str] = Field(
        default_factory=list,
        description="Names of predefined MCP servers to enable",
    )
    auto_log_dir: Optional[Path] = Field(
        default=None,
        description="Host path for auto-log output (set when --auto-log is active)",
    )
    kadi4mat_project: Optional[str] = Field(
        default=None,
        description="Kadi4Mat project name (required when kadi4mat server is enabled)",
    )
    kadi4mat_max_per_minute: int = Field(
        default=10,
        description="Kadi4Mat: max records created per minute (rate limit)",
    )
    kadi4mat_max_per_session: int = Field(
        default=500,
        description="Kadi4Mat: max records per session (0 = unlimited)",
    )
    container_runtime: Literal["docker", "podman"] = Field(
        default="docker",
        description="Container runtime to use: 'docker' or 'podman'",
    )
    no_web: bool = Field(
        default=False,
        description=(
            "Disable web tools. This is a SOFT restriction for both agents: it removes "
            "the dedicated web tools but does not block network access. For Claude Code "
            "the built-in web tools are disabled via --disallowedTools, "
            "but Bash is still allowed so curl/wget/python can reach the network. For "
            "OpenClaw there is no CLI flag, so only a system-prompt instruction is injected."
        ),
    )
    update_tools: bool = Field(
        default=False,
        description="Watch tools file for changes and automatically reload the MCP server.",
    )
    agent_args: dict[str, Any] = Field(
        default_factory=dict,
        description="Agent-specific arguments passed via --agent-args.",
    )
    created_at: datetime = Field(default_factory=datetime.now)


class SessionMetadata(BaseModel):
    """Persisted metadata for a session, stored alongside session data."""

    config: SessionConfig
    container_id: Optional[str] = None
    image_tag: Optional[str] = None
    status: str = Field(
        default="created",
        description="Session status: created, running, stopped, committed",
    )
    started_at: Optional[datetime] = None
    stopped_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def session_dir(self) -> Path:
        """Return the directory for this session's data."""
        return get_sessions_dir() / self.config.name

    def save(self) -> None:
        """Persist the metadata to ``<session_dir>/metadata.json``."""
        directory = self.session_dir()
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "metadata.json"
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, session_name: str) -> SessionMetadata:
        """Load metadata for *session_name* from disk.

        Raises ``FileNotFoundError`` if the session does not exist.
        """
        path = get_sessions_dir() / session_name / "metadata.json"
        if not path.exists():
            raise FileNotFoundError(f"No session metadata found at {path}")
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    @classmethod
    def list_sessions(cls) -> list[SessionMetadata]:
        """Return metadata for every session that has been saved to disk."""
        sessions_dir = get_sessions_dir()
        results: list[SessionMetadata] = []
        if not sessions_dir.exists():
            return results
        for entry in sorted(sessions_dir.iterdir()):
            meta_path = entry / "metadata.json"
            if meta_path.exists():
                try:
                    results.append(cls.model_validate_json(meta_path.read_text(encoding="utf-8")))
                except (json.JSONDecodeError, ValueError):
                    continue
        return results
