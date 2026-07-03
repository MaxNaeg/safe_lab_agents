"""Tests for utility functions."""

from __future__ import annotations

from safe_lab_agents.utils import find_free_port, generate_session_name, resolve_path


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


def test_generate_session_name() -> None:
    """Session names follow the expected format."""
    name = generate_session_name()
    assert name.startswith("session-")
    assert len(name) > len("session-")


def test_resolve_path_tilde() -> None:
    """resolve_path expands ~ to the home directory."""
    p = resolve_path("~/Documents")
    assert "~" not in str(p)
    assert p.is_absolute()


def test_resolve_path_relative() -> None:
    """resolve_path converts relative paths to absolute."""
    p = resolve_path("some/relative/path")
    assert p.is_absolute()
