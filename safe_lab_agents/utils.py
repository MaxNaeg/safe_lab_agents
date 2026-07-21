"""Shared utility functions for safe_lab_agents."""

from __future__ import annotations

import logging
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


def configure_logging(level: str | int | None) -> None:
    """Install a stderr log handler at *level* (a name like ``"DEBUG"`` or a number).

    A no-op when *level* is ``None``/empty or an unrecognised name, so callers can
    forward an unset ``--log-level``/``LOG_LEVEL`` verbatim. The package logs
    through module loggers but never configured a sink, so without this only
    ``WARNING``+ surfaced (via logging's last-resort handler) and every
    ``debug``/``info`` line was dropped. Logs go to stderr to stay clear of the
    agent's stdout. Called once on the host (CLI callback) and once in the MCP
    server subprocess, which starts fresh under ``spawn``.
    """
    if level is None or level == "":
        return
    if isinstance(level, str):
        resolved = logging.getLevelName(level.strip().upper())
        if not isinstance(resolved, int):  # unknown name → "Level FOO" string
            return
        level = resolved
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    logging.getLogger().setLevel(level)  # in case a handler already existed


def utc_now() -> datetime:
    """Return the current time as a timezone-aware UTC datetime.

    Preferred over the naive ``datetime.now()`` everywhere a timestamp is
    stored or compared: naive local datetimes are ambiguous (they silently
    shift with the host timezone and DST) and cannot be compared against the
    timezone-aware timestamps parsed from agent logs, which raises
    ``TypeError`` on a mixed sort. Using UTC everywhere keeps every stored
    timestamp unambiguous and mutually comparable.
    """
    return datetime.now(timezone.utc)


def find_free_port() -> int:
    """Find and return a free TCP port on localhost.

    Binds to port 0 to let the OS assign a free port, then releases it.
    ``SO_REUSEADDR`` lets the eventual server bind the same port immediately
    (e.g. if it is briefly in ``TIME_WAIT``).  A small time-of-check-to-time-of-
    use window remains — another process could claim the port before the server
    binds it — so callers verify the server actually came up via
    :func:`wait_for_server` and retry as needed (see the reload path).
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("", 0))
        return s.getsockname()[1]


def generate_session_name() -> str:
    """Generate a unique session name based on the current timestamp.

    Returns a string like ``session-20260413-153042``.
    """
    return f"session-{datetime.now().strftime('%Y%m%d-%H%M%S')}"


def wait_for_server(port: int, host: str = "127.0.0.1", timeout: float = 30.0) -> bool:
    """Block until an HTTP server at *host*:*port* responds, or *timeout* expires.

    Polls the ``/health`` endpoint every 0.5 seconds.

    Returns:
        ``True`` if the server responded within the timeout, ``False`` otherwise.
    """
    url = f"http://{host}:{port}/health"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except (URLError, OSError, ConnectionError):
            pass
        time.sleep(0.5)
    return False


def safe_under(base: Path, name: str) -> Path | None:
    """Resolve *name* against *base*, returning it only if it stays inside *base*.

    Guards against path traversal from untrusted record data (e.g. figure names
    read from agent-written auto-log records).  Joining ``base / name`` with
    ``pathlib`` does *not* keep the result inside ``base``: an absolute *name*
    (``/etc/passwd``) discards ``base`` entirely, and ``../`` segments escape it.

    Returns the resolved absolute path if it is contained in ``base``, otherwise
    ``None``.  ``resolve()`` follows symlinks, so a symlink planted inside
    ``base`` that points outside is also rejected.
    """
    base_resolved = base.resolve()
    candidate = (base_resolved / name).resolve()
    if candidate == base_resolved or candidate.is_relative_to(base_resolved):
        return candidate
    return None
