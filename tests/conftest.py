"""Shared pytest fixtures for safe_lab_agents tests."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest


@pytest.fixture()
def tmp_tools_file(tmp_path: Path) -> Path:
    """Create a temporary Python file with sample MCP tool functions."""
    tools = tmp_path / "tools.py"
    tools.write_text(
        textwrap.dedent("""\
        def read_temperature() -> float:
            \"\"\"Read the current temperature in Celsius.\"\"\"
            return 22.5

        def set_voltage(voltage: float) -> str:
            \"\"\"Set the output voltage.

            Args:
                voltage: Target voltage in volts.
            \"\"\"
            return f"Voltage set to {voltage} V."

        def _private_helper():
            \"\"\"This should not be loaded as a tool.\"\"\"
            pass

        MCP_TOOLS = [read_temperature, set_voltage]
        PYTHON_TOOLS = [read_temperature]
        """)
    )
    return tools


@pytest.fixture()
def tmp_tools_file_no_public(tmp_path: Path) -> Path:
    """Create a tools file with no MCP_TOOLS or PYTHON_TOOLS declared."""
    tools = tmp_path / "tools_private.py"
    tools.write_text(
        textwrap.dedent("""\
        def _internal():
            pass
        """)
    )
    return tools


@pytest.fixture()
def tmp_tools_file_no_hints(tmp_path: Path) -> Path:
    """Create a tools file with functions missing type hints."""
    tools = tmp_path / "tools_nohints.py"
    tools.write_text(
        textwrap.dedent("""\
        def measure(channel):
            return 42.0

        MCP_TOOLS = [measure]
        PYTHON_TOOLS = [measure]
        """)
    )
    return tools


@pytest.fixture()
def tmp_requirements_file(tmp_path: Path) -> Path:
    """Create a temporary requirements.txt."""
    req = tmp_path / "requirements.txt"
    req.write_text("requests>=2.28.0\nscikit-learn>=1.0.0\n")
    return req


@pytest.fixture()
def tmp_context_dir(tmp_path: Path) -> Path:
    """Create a temporary context directory with a sample file."""
    ctx = tmp_path / "context"
    ctx.mkdir()
    (ctx / "experiment.md").write_text("# Experiment\nMeasure temperature vs voltage.")
    return ctx


@pytest.fixture()
def tmp_shared_dir(tmp_path: Path) -> Path:
    """Create a temporary shared directory."""
    shared = tmp_path / "shared"
    shared.mkdir()
    return shared
