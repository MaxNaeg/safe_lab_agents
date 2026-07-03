"""Registry for predefined MCP server modules.

Predefined servers bundle a fixed set of MCP tools that can be enabled alongside
the user's own tools.  Each server is a class decorated with
:func:`register_server` that implements :meth:`get_tools`.

Example usage::

    from safe_lab_agents.mcp.predefined import register_server, PredefinedServer

    @register_server("lab-notebook")
    class LabNotebookServer(PredefinedServer):
        def get_tools(self):
            return [add_entry, search_entries, list_entries]
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

PREDEFINED_SERVERS: dict[str, type[PredefinedServer]] = {}


class PredefinedServer(ABC):
    """Base class for a predefined MCP server module."""

    @abstractmethod
    def get_tools(self) -> list[Callable]:
        """Return MCP tools exposed to the agent via tool-calling."""
        ...

    def get_python_tools(self) -> list[Callable]:
        """Return functions added to the Python /invoke registry only.

        These are callable from Python scripts inside Docker via the generated
        client (e.g. ``auto_log_client.py``) but are not exposed as MCP tools
        and therefore not visible to the agent in conversation.
        Override to provide server-side functions that are only useful in code.
        """
        return []


def register_server(name: str):
    """Class decorator that registers a :class:`PredefinedServer` under *name*.

    Args:
        name: Short identifier used on the CLI (e.g. ``"lab-notebook"``).
    """

    def decorator(cls: type[PredefinedServer]) -> type[PredefinedServer]:
        PREDEFINED_SERVERS[name] = cls
        return cls

    return decorator


def get_predefined_server(name: str) -> PredefinedServer:
    """Instantiate and return the predefined server registered under *name*.

    Raises:
        ValueError: If no server with that name has been registered.
    """
    if name not in PREDEFINED_SERVERS:
        available = ", ".join(sorted(PREDEFINED_SERVERS)) or "(none)"
        raise ValueError(
            f"Unknown predefined server '{name}'. Available: {available}"
        )
    return PREDEFINED_SERVERS[name]()


def list_predefined_servers() -> list[str]:
    """Return the names of all registered predefined servers."""
    return sorted(PREDEFINED_SERVERS)


# ------------------------------------------------------------------
# Auto-import predefined server modules so they self-register.
# ------------------------------------------------------------------
from safe_lab_agents.mcp.predefined import lab_notebook as _  # noqa: F401, E402

try:
    from safe_lab_agents.mcp.predefined import kadi4mat as _kadi  # noqa: F401, E402
except (ImportError, ValueError):
    # kadi-apy not installed or KADI4MAT_PROJECT not set — kadi4mat server
    # simply won't appear in the registry.
    pass
