"""Tests for utility functions."""

from __future__ import annotations

from safe_lab_agents.utils import find_free_port, generate_session_name


def test_find_free_port() -> None:
    """find_free_port returns a valid port number."""
    port = find_free_port()
    assert isinstance(port, int)
    assert 1024 <= port <= 65535


def test_find_free_port_unique() -> None:
    """Two calls return different ports (high probability)."""
    ports = {find_free_port() for _ in range(5)}
    # At least 2 distinct ports in 5 tries.
    assert len(ports) >= 2


def test_find_free_port_is_bindable() -> None:
    """The returned port can actually be bound (SO_REUSEADDR lets us rebind it
    immediately, as the server does after the probe socket closes)."""
    import socket

    port = find_free_port()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("", port))  # must not raise


def test_generate_session_name() -> None:
    """Session names follow the expected format."""
    name = generate_session_name()
    assert name.startswith("session-")
    assert len(name) > len("session-")
