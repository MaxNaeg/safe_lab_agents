"""Tests for the generated tools client header."""

from __future__ import annotations

import inspect
import io
import json
import urllib.error
from enum import Enum
from pathlib import Path

import pytest

from safe_lab_agents.mcp.client_generator import (
    _CLIENT_HEADER,
    _format_tools_info,
    _render_function,
)


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


# --- default-value rendering (non-literal reprs must not break the client) ---


class _Mode(Enum):
    FAST = 1
    SLOW = 2


def _tricky_tool(
    p: Path = Path("."),
    x: float = float("inf"),
    mode: _Mode = _Mode.FAST,
    n: int = 5,
    label: str = "run",
) -> dict:
    """A tool whose defaults have non-literal reprs."""
    return {}


def _scan(start: float, stop: float, steps: int = 100):
    """A tool with required args."""
    return []


def _exec_generated(monkeypatch, funcs):
    """Compile+exec the full generated client (header + funcs); return its namespace."""
    monkeypatch.setenv("MCP_PORT", "5000")
    src = _CLIENT_HEADER + "".join(_render_function(f) for f in funcs)
    ns: dict = {}
    exec(compile(src, "<client>", "exec"), ns)  # must not raise
    return ns, src


def test_client_with_nonliteral_defaults_is_importable(monkeypatch) -> None:
    """Path()/float('inf')/enum defaults used to emit PosixPath('.')/inf/<Mode.FAST:1>
    — a NameError/SyntaxError on import. Now the module compiles and exec's cleanly."""
    ns, _ = _exec_generated(monkeypatch, [_tricky_tool])
    fn = ns["_tricky_tool"]
    # Every optional param defaults to the sentinel in the executable signature.
    for param in inspect.signature(fn).parameters.values():
        assert param.default is ns["_MISSING"]


def test_client_type_hints_preserved_in_signature(monkeypatch) -> None:
    _, src = _exec_generated(monkeypatch, [_tricky_tool])
    assert "p: Path = _MISSING" in src
    assert "x: float = _MISSING" in src
    assert "mode: _Mode = _MISSING" in src


def test_client_docstring_shows_true_defaults(monkeypatch) -> None:
    """The real defaults survive as documentation (source-faithful, not repr)."""
    _, src = _exec_generated(monkeypatch, [_tricky_tool])
    assert "p: Path = Path('.')" in src
    assert "x: float = float('inf')" in src
    assert "mode: _Mode = _Mode.FAST" in src
    assert "n: int = 5" in src


def test_client_omits_missing_args_so_host_applies_defaults(monkeypatch) -> None:
    ns, _ = _exec_generated(monkeypatch, [_tricky_tool])
    sent: dict = {}
    ns["_invoke"] = lambda name, **kw: sent.update(name=name, kw=kw)

    ns["_tricky_tool"](n=7)
    assert sent["kw"] == {"n": 7}  # only the supplied arg is sent

    sent.clear()
    ns["_tricky_tool"]()
    assert sent["kw"] == {}  # nothing sent → host uses every real default


def test_client_required_args_always_sent(monkeypatch) -> None:
    ns, _ = _exec_generated(monkeypatch, [_scan])
    sent: dict = {}
    ns["_invoke"] = lambda name, **kw: sent.update(name=name, kw=kw)
    ns["_scan"](1.0, 2.0)
    assert sent["kw"] == {"start": 1.0, "stop": 2.0}  # required args sent, steps omitted


def test_tools_info_shows_true_defaults() -> None:
    info = _format_tools_info([_tricky_tool])
    assert "Path('.')" in info
    assert "float('inf')" in info
    assert "_Mode.FAST" in info


# --- Quantity reconstruction on the client side ---


def test_invoke_rebuilds_scalar_quantity(monkeypatch) -> None:
    """A bare {value, unit} result comes back as a Quantity, not a plain dict."""
    ns = _exec_header(monkeypatch, port="5000", host=None)
    import pickle

    monkeypatch.setattr(
        ns["urllib"].request,
        "urlopen",
        lambda req: io.BytesIO(pickle.dumps({"value": 2.5, "unit": "W"})),
    )
    result = ns["_invoke"]("read_power")
    assert isinstance(result, ns["Quantity"])
    assert result.value == 2.5
    assert result.unit == "W"
    assert result.term is None
    assert float(result) == 2.5


def test_invoke_rebuilds_nested_quantities(monkeypatch) -> None:
    """Quantities nested inside dicts/lists are rebuilt; plain values are untouched."""
    ns = _exec_header(monkeypatch, port="5000", host=None)
    import pickle

    payload = {
        "power": {"value": 2.5, "unit": "W", "term": "http://qudt.org/vocab/unit/W"},
        "readings": [{"value": 1, "unit": "V"}, {"value": 2, "unit": "V"}],
        "label": "run-1",
    }
    monkeypatch.setattr(
        ns["urllib"].request,
        "urlopen",
        lambda req: io.BytesIO(pickle.dumps(payload)),
    )
    result = ns["_invoke"]("read_all")
    assert isinstance(result["power"], ns["Quantity"])
    assert result["power"].term == "http://qudt.org/vocab/unit/W"
    assert all(isinstance(r, ns["Quantity"]) for r in result["readings"])
    assert result["label"] == "run-1"  # plain values pass through unchanged


def test_invoke_does_not_rebuild_ndarray_ref_dict(monkeypatch) -> None:
    """An ndarray reference dict (has a 'unit' key) must not be mistaken for a
    quantity."""
    ns = _exec_header(monkeypatch, port="5000", host=None)
    import pickle

    ref = {"_type": "ndarray", "value": "scan.x", "unit": "V"}
    monkeypatch.setattr(
        ns["urllib"].request,
        "urlopen",
        lambda req: io.BytesIO(pickle.dumps(ref)),
    )
    result = ns["_invoke"]("read_trace")
    assert result == ref  # left as-is, not wrapped in Quantity


def test_quantity_type_annotation_surfaces_in_signature(monkeypatch) -> None:
    """A '-> Quantity' return annotation is preserved in the generated stub, and
    the stub is importable because Quantity is defined in the header."""

    def read_power() -> Quantity:  # noqa: F821 - annotation is a string (PEP 563)
        """Read power."""
        return {}

    ns, src = _exec_generated(monkeypatch, [read_power])
    assert "-> Quantity" in src
    assert "read_power" in ns


def test_tools_info_explains_quantity() -> None:
    """The agent-facing info explains the Quantity object it will receive."""
    info = _format_tools_info([_a_tool])
    assert "Quantity" in info
    assert ".value" in info
    assert ".unit" in info
