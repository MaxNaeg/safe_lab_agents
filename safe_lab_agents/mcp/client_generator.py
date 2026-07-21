"""Generate a Python client for experiment tools.

Reads ``PYTHON_TOOLS`` from the user's tools file and writes two files into
the workspace directory:

- ``tools_client.py``: callable wrappers using urllib + pickle (stdlib only).
  Both arguments and return values are pickle-serialized, so any Python object
  (numpy arrays, dicts, custom classes, …) can cross the boundary.
- ``python_tools_info.txt``: human-readable documentation injected into the
  agent's system prompt by the entrypoint script.
"""

from __future__ import annotations

import ast
import inspect
import logging
import textwrap
from pathlib import Path
from typing import Callable

from safe_lab_agents.mcp.loader import load_tools_from_file

logger = logging.getLogger(__name__)

_CLIENT_HEADER = '''\
"""Auto-generated Python client for experiment tools.

Import this to call experiment tools from Python scripts inside Docker.
Inputs are sent as JSON (numpy arrays encoded as .npy byte streams).
Outputs are received as pickle from the trusted host.
"""
from __future__ import annotations
import base64
import io
import json
import os
import pickle
import urllib.error
import urllib.request

try:
    import numpy as np
    _NUMPY = True
except ImportError:
    _NUMPY = False

_HOST = os.environ.get("MCP_HOST", "host.docker.internal")
_PORT = os.environ["MCP_PORT"]
_URL  = f"http://{_HOST}:{_PORT}/invoke"
_HEADERS = {"Content-Type": "application/json"}
_TOKEN = os.environ.get("MCP_AUTH_TOKEN", "")
if _TOKEN:
    _HEADERS["Authorization"] = f"Bearer {_TOKEN}"


class _MissingType:
    """Sentinel for an omitted optional argument.

    Optional parameters default to _MISSING; the wrapper simply doesn't send
    them, so the real tool on the host applies its own default. The true
    default value is shown in each function's docstring / signature.
    """
    __slots__ = ()

    def __repr__(self):
        return "<use tool default>"


_MISSING = _MissingType()


class Quantity:
    """A measurement value carrying a unit, as returned by a tool.

    ``value`` is the underlying number or numpy array, ``unit`` is a string
    (e.g. "W"), and ``term`` is an optional ontology IRI.  A tool annotated
    ``-> Quantity`` (or e.g. ``-> dict[str, Quantity]``) returns these; use it
    like its value in arithmetic (``float(q)``, ``np.asarray(q)``) — the unit is
    carried on the object, not through numeric operations.
    """
    __slots__ = ("value", "unit", "term")

    def __init__(self, value, unit, term=None):
        self.value = value
        self.unit = unit
        self.term = term

    def __repr__(self):
        if self.term:
            return f"Quantity({self.value!r}, {self.unit!r}, term={self.term!r})"
        return f"Quantity({self.value!r}, {self.unit!r})"

    def __str__(self):
        return f"{self.value} {self.unit}"

    def __float__(self):
        return float(self.value)

    def __int__(self):
        return int(self.value)

    def __array__(self, dtype=None):
        if not _NUMPY:
            raise TypeError("numpy is not available to convert this Quantity")
        arr = np.asarray(self.value)
        return arr.astype(dtype) if dtype is not None else arr

    def __eq__(self, other):
        if isinstance(other, Quantity):
            return self.value == other.value and self.unit == other.unit
        return NotImplemented

    __hash__ = None


def _is_quantity_dict(obj):
    """Mirror of ``records.is_quantity``: a {value, unit} dict that is not an
    ndarray reference dict."""
    return (
        isinstance(obj, dict)
        and "value" in obj
        and "unit" in obj
        and obj.get("_type") != "ndarray"
    )


def _rebuild(obj):
    """Recursively turn quantity dicts in a tool result into Quantity objects."""
    if _is_quantity_dict(obj):
        return Quantity(_rebuild(obj["value"]), obj["unit"], obj.get("term"))
    if isinstance(obj, dict):
        return {k: _rebuild(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_rebuild(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_rebuild(v) for v in obj)
    return obj


def _encode_arg(obj):
    """Encode one argument for JSON transport. Numpy arrays use the .npy byte stream."""
    if _NUMPY and isinstance(obj, np.ndarray):
        buf = io.BytesIO()
        np.save(buf, obj)
        return {"__type__": "ndarray", "data": base64.b64encode(buf.getvalue()).decode()}
    if isinstance(obj, (list, tuple)):
        return [_encode_arg(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _encode_arg(v) for k, v in obj.items()}
    return obj  # scalars, str, bool, None — natively JSON-safe


def _invoke(tool_name: str, **kwargs):
    body = json.dumps(
        {"tool": tool_name, "args": {k: _encode_arg(v) for k, v in kwargs.items()}}
    ).encode()
    req = urllib.request.Request(_URL, data=body, headers=_HEADERS)
    try:
        with urllib.request.urlopen(req) as r:
            return _rebuild(pickle.loads(r.read()))
    except urllib.error.HTTPError as e:
        if e.code == 422:
            msg = json.loads(e.read().decode()).get("error", "type validation failed")
            raise TypeError(f"Tool '{tool_name}': {msg}") from None
        if e.code == 500:
            try:
                payload = json.loads(e.read().decode())
            except Exception:
                raise
            detail = payload.get("traceback") or payload.get("error", "tool raised an exception")
            raise RuntimeError(
                f"Tool '{tool_name}' raised an error on the host:\\n{detail}"
            ) from None
        raise


'''


def generate_client_files(tools_file: Path, workspace_dir: Path) -> None:
    """Generate ``tools_client.py`` and ``python_tools_info.txt`` in *workspace_dir*.

    Does nothing if ``PYTHON_TOOLS`` is not declared in *tools_file*.
    """
    _, python_tools = load_tools_from_file(tools_file)
    if not python_tools:
        logger.info("No PYTHON_TOOLS declared; skipping Python client generation.")
        return

    _write_client_py(python_tools, workspace_dir / "tools_client.py")
    _write_tools_info(python_tools, workspace_dir / "python_tools_info.txt")
    logger.info(
        "Generated tools_client.py and python_tools_info.txt in %s", workspace_dir
    )


def _write_client_py(tools: list[Callable], output: Path) -> None:
    lines = [_CLIENT_HEADER]
    for func in tools:
        lines.append(_render_function(func))
    output.write_text("".join(lines), encoding="utf-8")


def _render_function(func: Callable) -> str:
    sig = inspect.signature(func)
    source_anns = _source_annotations(func)
    source_defaults = _source_defaults(func)
    ret_str = _render_return(sig, source_anns)

    defaulted = [
        n for n, p in sig.parameters.items()
        if p.default is not inspect.Parameter.empty
    ]
    # Executable signature: every default becomes the _MISSING sentinel so the
    # stub is always importable (a non-literal repr like PosixPath('.') is not).
    exec_defaults = {n: "_MISSING" for n in defaulted}
    params_str = _render_params(sig, source_anns, exec_defaults)

    # Documented signature: the real defaults (from source), shown so the agent
    # still knows the true type hints AND default values.
    true_sig = (
        f"{func.__name__}({_render_params(sig, source_anns, source_defaults)}){ret_str}"
    )

    lines = [f"def {func.__name__}({params_str}){ret_str}:\n"]

    # Docstring: the true signature first, then the user's own docstring.
    doc = true_sig
    if func.__doc__:
        doc += "\n\n" + textwrap.dedent(func.__doc__).strip()
    lines.append(f'    """{doc}\n    """\n')

    # Body: forward only the args the caller actually supplied, so any omitted
    # optional argument is resolved to its real default on the host.
    if sig.parameters:
        lines.append("    _kw = {}\n")
        for name, param in sig.parameters.items():
            if param.default is inspect.Parameter.empty:
                lines.append(f"    _kw[{name!r}] = {name}\n")
            else:
                lines.append(f"    if {name} is not _MISSING:\n")
                lines.append(f"        _kw[{name!r}] = {name}\n")
        lines.append(f"    return _invoke({func.__name__!r}, **_kw)\n\n")
    else:
        lines.append(f"    return _invoke({func.__name__!r})\n\n")
    return "".join(lines)


def _find_funcdef(func: Callable) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    """Return the AST definition node for *func*, or ``None`` if unavailable."""
    try:
        source = textwrap.dedent(inspect.getsource(func))
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == func.__name__
            ):
                return node
    except Exception:
        pass
    return None


def _source_annotations(func: Callable) -> dict[str, str]:
    """Return annotations as written in source (e.g. 'np.ndarray', not 'ndarray').

    Falls back to an empty dict if the source is unavailable.
    """
    node = _find_funcdef(func)
    if node is None:
        return {}
    result: dict[str, str] = {}
    all_args = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
    for arg in all_args:
        if arg.annotation:
            result[arg.arg] = ast.unparse(arg.annotation)
    if node.returns:
        result["return"] = ast.unparse(node.returns)
    return result


def _source_defaults(func: Callable) -> dict[str, str]:
    """Return default values as written in source (e.g. 'Path(".")', 'float("inf")').

    Reproducing a default's ``repr()`` in generated code is unsafe — non-literal
    reprs (``PosixPath('.')``, ``<Color.RED: 1>``, ``inf``) are unimportable — so
    the executable stub uses a sentinel and the *real* default is surfaced as
    documentation text via this map.  Falls back to an empty dict if unavailable.
    """
    node = _find_funcdef(func)
    if node is None:
        return {}
    result: dict[str, str] = {}
    positional = [*node.args.posonlyargs, *node.args.args]
    defaults = node.args.defaults  # right-aligned over posonly + positional
    offset = len(positional) - len(defaults)
    for i, arg in enumerate(positional):
        if i >= offset:
            result[arg.arg] = ast.unparse(defaults[i - offset])
    for arg, default in zip(node.args.kwonlyargs, node.args.kw_defaults):
        if default is not None:
            result[arg.arg] = ast.unparse(default)
    return result


def _render_params(
    sig: inspect.Signature,
    source_anns: dict[str, str] | None = None,
    default_render: dict[str, str] | None = None,
) -> str:
    """Render a parameter list.

    *default_render* maps a parameter name to the exact text to emit after
    ``=``.  A defaulted parameter absent from the map falls back to ``repr()``
    (safe only for literals — callers emitting executable code pass an explicit
    map for every defaulted parameter).
    """
    parts = []
    for name, param in sig.parameters.items():
        part = name
        if param.annotation is not inspect.Parameter.empty:
            ann_str = (source_anns or {}).get(name) or _annotation_str(param.annotation)
            part += f": {ann_str}"
        if param.default is not inspect.Parameter.empty:
            rendered = (default_render or {}).get(name)
            if rendered is None:
                rendered = repr(param.default)
            part += f" = {rendered}"
        parts.append(part)
    return ", ".join(parts)


def _render_return(sig: inspect.Signature, source_anns: dict[str, str] | None = None) -> str:
    if sig.return_annotation is inspect.Parameter.empty:
        return ""
    ann_str = (source_anns or {}).get("return") or _annotation_str(sig.return_annotation)
    return f" -> {ann_str}"


def _annotation_str(ann) -> str:
    if isinstance(ann, str):
        return ann
    if hasattr(ann, "__name__"):
        return ann.__name__
    return str(ann).replace("typing.", "")


def _format_tools_info(tools: list[Callable]) -> str:
    names = ", ".join(f.__name__ for f in tools)
    lines = [
        "Python tools are available for use in scripts at /agent/workspace/tools_client.py.",
        "Both inputs and outputs support any Python object (numpy arrays, dicts, etc.).",
        "",
        "Import with:",
        '    import sys; sys.path.insert(0, "/agent/workspace")',
        f"    from tools_client import {names}",
        "",
        "Measurement values carry units as Quantity objects. A tool that returns a",
        "measurement (annotated '-> Quantity', or a dict/list containing quantities)",
        "gives back Quantity objects with .value (the number or numpy array), .unit",
        "(a string like 'W'), and .term (an optional ontology IRI, may be None). Read",
        "the number via q.value or float(q); np.asarray(q) gives the array. Import the",
        "type from tools_client if you need it: 'from tools_client import Quantity'.",
        "",
        "Available Python tools:",
    ]
    for func in tools:
        sig = inspect.signature(func)
        source_anns = _source_annotations(func)
        source_defaults = _source_defaults(func)
        sig_str = f"{func.__name__}({_render_params(sig, source_anns, source_defaults)}){_render_return(sig, source_anns)}"
        lines.append(f"  {sig_str}")
        if func.__doc__:
            for line in textwrap.dedent(func.__doc__).strip().splitlines():
                lines.append(f"      {line}")
        lines.append("")
    return "\n".join(lines) + "\n"


def _write_tools_info(tools: list[Callable], output: Path) -> None:
    output.write_text(_format_tools_info(tools), encoding="utf-8")
