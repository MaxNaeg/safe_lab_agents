"""Tests for the generated tools client header."""

from __future__ import annotations

import io
import json
import urllib.error

import pytest

from safe_lab_agents.mcp.client_generator import _CLIENT_HEADER, _format_tools_info


def _exec_header(monkeypatch, *, port: str, host: str | None) -> dict:
    """Exec the generated client header with the given env and return its namespace."""
    monkeypatch.setenv("MCP_PORT", port)
    if host is None:
        monkeypatch.delenv("MCP_HOST", raising=False)
    else:
        monkeypatch.setenv("MCP_HOST", host)
    namespace: dict = {}
    exec(compile(_CLIENT_HEADER, "<client-header>", "exec"), namespace)
    return namespace


def test_client_url_defaults_to_host_docker_internal(monkeypatch) -> None:
    """Without MCP_HOST, the client targets host.docker.internal (Docker default)."""
    ns = _exec_header(monkeypatch, port="5000", host=None)
    assert ns["_URL"] == "http://host.docker.internal:5000/invoke"


def test_client_url_honours_mcp_host_override(monkeypatch) -> None:
    """When MCP_HOST is set (Podman/Windows), the client targets that address."""
    ns = _exec_header(monkeypatch, port="5000", host="172.26.80.1")
    assert ns["_URL"] == "http://172.26.80.1:5000/invoke"


def test_client_adds_authorization_header_when_token_set(monkeypatch) -> None:
    """With MCP_AUTH_TOKEN set, the client sends a bearer Authorization header."""
    monkeypatch.setenv("MCP_AUTH_TOKEN", "s3cr3t")
    ns = _exec_header(monkeypatch, port="5000", host=None)
    assert ns["_HEADERS"]["Authorization"] == "Bearer s3cr3t"


def test_client_omits_authorization_header_without_token(monkeypatch) -> None:
    """Without MCP_AUTH_TOKEN, no Authorization header is added (backward compatible)."""
    monkeypatch.delenv("MCP_AUTH_TOKEN", raising=False)
    ns = _exec_header(monkeypatch, port="5000", host=None)
    assert "Authorization" not in ns["_HEADERS"]


def _a_tool(channel: int) -> float:
    """A sample tool."""
    return 1.0


def test_tools_info_has_no_reload_text() -> None:
    """Python-tools info carries no reload guidance (reload_tools is an MCP tool,
    documented separately via reload_info.txt)."""
    assert "reload_tools" not in _format_tools_info([_a_tool])


def _patch_urlopen(monkeypatch, ns, *, code: int, body: bytes) -> None:
    """Make the client's urlopen raise an HTTPError with the given code/body."""

    def fake_urlopen(req):
        raise urllib.error.HTTPError(ns["_URL"], code, "error", {}, io.BytesIO(body))

    monkeypatch.setattr(ns["urllib"].request, "urlopen", fake_urlopen)


def test_invoke_surfaces_host_exception(monkeypatch) -> None:
    """A 500 from the host is raised as a RuntimeError carrying the real traceback."""
    ns = _exec_header(monkeypatch, port="5000", host=None)
    body = json.dumps(
        {
            "error": "ValueError: bad value",
            "traceback": "Traceback (most recent call last):\nValueError: bad value",
        }
    ).encode()
    _patch_urlopen(monkeypatch, ns, code=500, body=body)

    with pytest.raises(RuntimeError) as exc:
        ns["_invoke"]("my_tool")
    msg = str(exc.value)
    assert "my_tool" in msg
    assert "bad value" in msg
    assert "Traceback" in msg


def test_invoke_type_error_still_maps_to_type_error(monkeypatch) -> None:
    """A 422 (arg validation) is still surfaced as a TypeError, not the new 500 path."""
    ns = _exec_header(monkeypatch, port="5000", host=None)
    body = json.dumps({"error": "missing required argument 'x'"}).encode()
    _patch_urlopen(monkeypatch, ns, code=422, body=body)

    with pytest.raises(TypeError) as exc:
        ns["_invoke"]("my_tool")
    assert "missing required argument" in str(exc.value)
