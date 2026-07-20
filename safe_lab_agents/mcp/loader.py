"""Dynamically load user-defined Python functions and expose them as MCP tools.

The user provides a plain ``.py`` file containing public functions and two
module-level lists:

- ``MCP_TOOLS``: functions exposed as MCP tools (AI tool-calling interface).
- ``PYTHON_TOOLS``: functions exposed via the Python client inside Docker.

Each list is optional; a warning is emitted when both are absent or empty.
A function may appear in both lists.

The file may also define ``GRACEFUL_EXPERIMENT_SHUTDOWN``, a callable invoked when
the MCP server process stops (both on ``--update-tools`` reload and on final
shutdown).  ``load_module_exports`` loads the tools and this hook from a *single*
module evaluation, so a hook closing over a stateful experiment (e.g. one built
via ``safe_lab_agents.experiment``) sees the same instance the tools used.
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import logging
import sys
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)


def load_tools_from_file(
    file_path: Path, shared_dir: Optional[Path] = None
) -> tuple[list[Callable] | None, list[Callable] | None]:
    """Load MCP_TOOLS and PYTHON_TOOLS from a Python file.

    Args:
        file_path: Absolute or relative path to a ``.py`` file that defines
            the experiment's tool functions, plus optional ``MCP_TOOLS`` and
            ``PYTHON_TOOLS`` module-level lists.
        shared_dir: If provided, injected as ``SHARED_DATA_DIR`` into the
            loaded module so tools can reference it without hardcoding paths.

    Returns:
        ``(mcp_tools, python_tools)`` where each element is the declared list
        or ``None`` if the corresponding list was not declared.

    Raises:
        FileNotFoundError: If *file_path* does not exist.
        ValueError: If the file is not a ``.py`` file, or if a declared list
            contains non-callable entries.
    """
    file_path = Path(file_path).resolve()

    if not file_path.exists():
        raise FileNotFoundError(f"Tools file not found: {file_path}")
    if file_path.suffix != ".py":
        raise ValueError(f"Tools file must be a .py file, got: {file_path.suffix}")

    module = _load_module(file_path)

    # Inject SHARED_DATA_DIR so user tools can reference it without hardcoding.
    module.SHARED_DATA_DIR = str(shared_dir.resolve()) if shared_dir is not None else None

    mcp_tools = _resolve_tool_list(module, "MCP_TOOLS", file_path)
    python_tools = _resolve_tool_list(module, "PYTHON_TOOLS", file_path)

    total = len(mcp_tools or []) + len(python_tools or [])
    if total == 0:
        logger.warning(
            "%s defines neither MCP_TOOLS nor PYTHON_TOOLS (or both are empty) "
            "— no tools will be registered.",
            file_path,
        )

    seen: set[int] = set()
    for func in (mcp_tools or []) + (python_tools or []):
        if id(func) not in seen:
            _warn_if_missing_metadata(func)
            seen.add(id(func))

    logger.info(
        "Loaded tools from %s: %d MCP tool(s), %d Python tool(s)",
        file_path,
        len(mcp_tools or []),
        len(python_tools or []),
    )
    return mcp_tools, python_tools


def load_shutdown_hook(
    file_path: Path, shared_dir: Optional[Path] = None
) -> Optional[Callable]:
    """Return the ``GRACEFUL_EXPERIMENT_SHUTDOWN`` callable from *file_path*, or ``None``.

    Raises:
        FileNotFoundError: If *file_path* does not exist.
        ValueError: If the file is not a ``.py`` file, or if
            ``GRACEFUL_EXPERIMENT_SHUTDOWN`` exists but is not callable.
    """
    file_path = Path(file_path).resolve()

    if not file_path.exists():
        raise FileNotFoundError(f"Tools file not found: {file_path}")
    if file_path.suffix != ".py":
        raise ValueError(f"Tools file must be a .py file, got: {file_path.suffix}")

    module = _load_module(file_path)
    module.SHARED_DATA_DIR = str(shared_dir.resolve()) if shared_dir is not None else None
    return _extract_shutdown_hook(module, file_path)


def load_module_exports(
    file_path: Path, shared_dir: Optional[Path] = None
) -> tuple[list[Callable] | None, list[Callable] | None, Optional[Callable]]:
    """Load tools and the shutdown hook from a *single* evaluation of *file_path*.

    Unlike calling ``load_tools_from_file`` and ``load_shutdown_hook`` separately
    (each of which re-evaluates the module), this loads the module once so the
    returned ``GRACEFUL_EXPERIMENT_SHUTDOWN`` closes over the *same* module state —
    including any stateful experiment — that the returned tools use.

    Returns:
        ``(mcp_tools, python_tools, shutdown_hook)``; each tool list is the declared
        list or ``None``, and ``shutdown_hook`` is the callable or ``None``.

    Raises:
        FileNotFoundError: If *file_path* does not exist.
        ValueError: If the file is not a ``.py`` file, if a declared tool list
            contains non-callables, or if ``GRACEFUL_EXPERIMENT_SHUTDOWN`` is not
            callable.
    """
    file_path = Path(file_path).resolve()

    if not file_path.exists():
        raise FileNotFoundError(f"Tools file not found: {file_path}")
    if file_path.suffix != ".py":
        raise ValueError(f"Tools file must be a .py file, got: {file_path.suffix}")

    module = _load_module(file_path)
    module.SHARED_DATA_DIR = str(shared_dir.resolve()) if shared_dir is not None else None

    mcp_tools = _resolve_tool_list(module, "MCP_TOOLS", file_path)
    python_tools = _resolve_tool_list(module, "PYTHON_TOOLS", file_path)
    shutdown_hook = _extract_shutdown_hook(module, file_path)
    return mcp_tools, python_tools, shutdown_hook


def _extract_shutdown_hook(module, file_path: Path) -> Optional[Callable]:
    """Return the validated ``GRACEFUL_EXPERIMENT_SHUTDOWN`` from *module*, or ``None``."""
    hook = getattr(module, "GRACEFUL_EXPERIMENT_SHUTDOWN", None)
    if hook is None:
        return None
    if not callable(hook):
        raise ValueError(
            f"'GRACEFUL_EXPERIMENT_SHUTDOWN' in {file_path} exists but is not callable."
        )
    logger.info("Loaded GRACEFUL_EXPERIMENT_SHUTDOWN from %s", file_path)
    return hook


def _load_module(file_path: Path):
    """Import a Python file as a module using importlib.

    The module is added to ``sys.modules`` so that relative imports inside
    the user file work correctly.
    """
    # Include a hash of the absolute path so two tool files that share a basename
    # (e.g. projA/tools.py and projB/tools.py) get distinct sys.modules keys and
    # don't clobber each other; the same file always maps to the same key.
    path_hash = hashlib.sha1(
        str(file_path.resolve()).encode("utf-8")
    ).hexdigest()[:8]
    module_name = f"_user_tools_{file_path.stem}_{path_hash}"
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot create module spec from {file_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _resolve_tool_list(module, attr: str, source_file: Path) -> list[Callable] | None:
    """Return the declared tool list for *attr*, or ``None`` if not declared."""
    declared = getattr(module, attr, None)
    if declared is None:
        return None
    if not isinstance(declared, list) or not all(callable(f) for f in declared):
        raise ValueError(
            f"'{attr}' in {source_file} must be a list of callables. "
            f"Example: {attr} = [my_function, another_function]"
        )
    return declared


def _warn_if_missing_metadata(func: Callable) -> None:
    """Log warnings when a tool function lacks type hints or a docstring.

    FastMCP relies on type hints to generate the JSON schema and on docstrings
    for tool descriptions.  Missing metadata degrades the agent's ability to
    use the tool correctly.
    """
    sig = inspect.signature(func)
    hints = func.__annotations__

    if not func.__doc__:
        logger.warning(
            "Tool '%s' has no docstring — the agent will not see a description for this tool.",
            func.__name__,
        )

    for param_name, param in sig.parameters.items():
        if param_name not in hints and param_name != "return":
            logger.warning(
                "Tool '%s', parameter '%s' has no type hint — "
                "the agent may not know what type to pass.",
                func.__name__,
                param_name,
            )
