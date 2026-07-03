"""Tests for the host-side MCP server helpers."""

from __future__ import annotations

import asyncio
import inspect

import pytest
from fastmcp.exceptions import ToolError

from safe_lab_agents.mcp.server import _BearerAuthMiddleware, _wrap_mcp_tool_errors


def _run_middleware(token: str, *, path: str, header: str | None) -> tuple[bool, int | None]:
    """Drive the ASGI middleware once.

    Returns ``(downstream_called, response_status)`` where ``response_status`` is
    the status the middleware itself sent (only set when it short-circuits).
    """
    downstream_called = {"hit": False}

    async def downstream(scope, receive, send):
        downstream_called["hit"] = True
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    mw = _BearerAuthMiddleware(downstream, token)
    headers = [(b"authorization", header.encode())] if header is not None else []
    scope = {"type": "http", "path": path, "headers": headers}
    sent: list[dict] = []

    async def send(message):
        sent.append(message)

    async def receive():
        return {"type": "http.request"}

    asyncio.run(mw(scope, receive, send))
    status = next((m["status"] for m in sent if m["type"] == "http.response.start"), None)
    return downstream_called["hit"], status


def test_auth_middleware_rejects_missing_token() -> None:
    """A request without an Authorization header is refused with 401."""
    hit, status = _run_middleware("sekret", path="/mcp", header=None)
    assert not hit
    assert status == 401


def test_auth_middleware_rejects_wrong_token() -> None:
    """A request with the wrong bearer token is refused with 401."""
    hit, status = _run_middleware("sekret", path="/invoke", header="Bearer nope")
    assert not hit
    assert status == 401


def test_auth_middleware_accepts_correct_token() -> None:
    """A request carrying the correct bearer token reaches the downstream app."""
    hit, status = _run_middleware("sekret", path="/mcp", header="Bearer sekret")
    assert hit
    assert status == 200


def test_auth_middleware_exempts_health() -> None:
    """GET /health is reachable without a token (host liveness probe)."""
    hit, status = _run_middleware("sekret", path="/health", header=None)
    assert hit
    assert status == 200


def test_wrap_mcp_tool_errors_includes_type_and_traceback() -> None:
    """A tool exception is re-raised as a ToolError with type, message, and traceback."""

    def boom(x: int) -> int:
        raise ValueError("bad value")

    wrapped = _wrap_mcp_tool_errors(boom)
    with pytest.raises(ToolError) as exc:
        wrapped(1)
    msg = str(exc.value)
    assert "boom raised ValueError: bad value" in msg
    assert "Traceback" in msg


def test_wrap_mcp_tool_errors_passes_through_tool_error() -> None:
    """An explicit ToolError is forwarded unchanged (no traceback wrapping)."""

    def deliberate() -> None:
        raise ToolError("please pass a positive number")

    wrapped = _wrap_mcp_tool_errors(deliberate)
    with pytest.raises(ToolError) as exc:
        wrapped()
    assert str(exc.value) == "please pass a positive number"


def test_wrap_mcp_tool_errors_preserves_metadata_and_result() -> None:
    """The wrapper preserves name/signature (for FastMCP schemas) and passes results through."""

    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    wrapped = _wrap_mcp_tool_errors(add)
    assert wrapped(2, 3) == 5
    assert wrapped.__name__ == "add"
    assert wrapped.__doc__ == "Add two numbers."
    assert list(inspect.signature(wrapped).parameters) == ["a", "b"]
