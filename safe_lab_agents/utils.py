"""Shared utility functions for safe_lab_agents."""

from __future__ import annotations

import socket
import time
from datetime import datetime
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


def find_free_port() -> int:
    """Find and return a free TCP port on localhost.

    Binds to port 0 to let the OS assign a free port, then immediately
    releases it.  There is a small race window, but in practice this is
    reliable for our use case.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
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


def resolve_path(path_str: str) -> Path:
    """Resolve a user-supplied path string to an absolute ``Path``.

    Expands ``~`` and resolves relative paths against the current working
    directory.
    """
    return Path(path_str).expanduser().resolve()


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
