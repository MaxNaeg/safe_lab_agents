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
            return pickle.loads(r.read())
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
    params_str = _render_params(sig, source_anns)
    ret_str = _render_return(sig, source_anns)
    kwargs_str = ", ".join(f"{n}={n}" for n in sig.parameters)
    invoke_args = repr(func.__name__)
    if kwargs_str:
        invoke_args += f", {kwargs_str}"

    lines = [f"def {func.__name__}({params_str}){ret_str}:\n"]
    if func.__doc__:
        doc = textwrap.dedent(func.__doc__).strip()
        if "\n" in doc:
            lines.append(f'    """{doc}\n    """\n')
        else:
            lines.append(f'    """{doc}"""\n')
    lines.append(f"    return _invoke({invoke_args})\n\n")
    return "".join(lines)


def _source_annotations(func: Callable) -> dict[str, str]:
    """Return annotations as written in source (e.g. 'np.ndarray', not 'ndarray').

    Falls back to an empty dict if the source is unavailable.
    """
    try:
        source = textwrap.dedent(inspect.getsource(func))
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == func.__name__
            ):
                result: dict[str, str] = {}
                all_args = [
                    *node.args.posonlyargs,
                    *node.args.args,
                    *node.args.kwonlyargs,
                ]
                for arg in all_args:
                    if arg.annotation:
                        result[arg.arg] = ast.unparse(arg.annotation)
                if node.returns:
                    result["return"] = ast.unparse(node.returns)
                return result
    except Exception:
        pass
    return {}


def _render_params(sig: inspect.Signature, source_anns: dict[str, str] | None = None) -> str:
    parts = []
    for name, param in sig.parameters.items():
        part = name
        if param.annotation is not inspect.Parameter.empty:
            ann_str = (source_anns or {}).get(name) or _annotation_str(param.annotation)
            part += f": {ann_str}"
        if param.default is not inspect.Parameter.empty:
            part += f" = {param.default!r}"
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
        "Available Python tools:",
    ]
    for func in tools:
        sig = inspect.signature(func)
        source_anns = _source_annotations(func)
        sig_str = f"{func.__name__}({_render_params(sig, source_anns)}){_render_return(sig, source_anns)}"
        lines.append(f"  {sig_str}")
        if func.__doc__:
            for line in textwrap.dedent(func.__doc__).strip().splitlines():
                lines.append(f"      {line}")
        lines.append("")
    return "\n".join(lines) + "\n"


def _write_tools_info(tools: list[Callable], output: Path) -> None:
    output.write_text(_format_tools_info(tools), encoding="utf-8")
