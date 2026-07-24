"""Tests for the MCP dynamic tool loader."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

import textwrap

from safe_lab_agents.mcp.loader import load_module_exports, load_tools_from_file


class TestLoadToolsFromFile:
    """Tests for :func:`load_tools_from_file`."""

    def test_loads_declared_lists(self, tmp_tools_file: Path) -> None:
        """MCP_TOOLS and PYTHON_TOOLS are loaded from the declared lists."""
        mcp_tools, python_tools = load_tools_from_file(tmp_tools_file)
        mcp_names = [f.__name__ for f in mcp_tools]
        python_names = [f.__name__ for f in python_tools]
        assert set(mcp_names) == {"read_temperature", "set_voltage"}
        assert python_names == ["read_temperature"]
        assert "_private_helper" not in mcp_names
        assert "_private_helper" not in python_names

    def test_preserves_signatures(self, tmp_tools_file: Path) -> None:
        """Loaded functions retain their type hints and docstrings."""
        mcp_tools, _ = load_tools_from_file(tmp_tools_file)
        temp_func = next(f for f in mcp_tools if f.__name__ == "read_temperature")
        assert temp_func.__doc__ is not None
        assert "temperature" in temp_func.__doc__.lower()
        assert temp_func.__annotations__.get("return") is float

    def test_functions_are_callable(self, tmp_tools_file: Path) -> None:
        """Loaded functions can be called and return correct values."""
        mcp_tools, python_tools = load_tools_from_file(tmp_tools_file)
        temp_func = next(f for f in mcp_tools if f.__name__ == "read_temperature")
        assert temp_func() == 22.5

        voltage_func = next(f for f in mcp_tools if f.__name__ == "set_voltage")
        assert voltage_func(3.3) == "Voltage set to 3.3 V."

        py_temp = next(f for f in python_tools if f.__name__ == "read_temperature")
        assert py_temp() == 22.5

    def test_raises_on_missing_file(self, tmp_path: Path) -> None:
        """FileNotFoundError when the file does not exist."""
        with pytest.raises(FileNotFoundError):
            load_tools_from_file(tmp_path / "nonexistent.py")

    def test_raises_on_non_py_file(self, tmp_path: Path) -> None:
        """ValueError when the file is not a .py file."""
        txt = tmp_path / "tools.txt"
        txt.write_text("not python")
        with pytest.raises(ValueError, match=r"\.py"):
            load_tools_from_file(txt)

    def test_warns_on_no_tools(self, tmp_tools_file_no_public: Path, caplog) -> None:
        """Warning logged when neither MCP_TOOLS nor PYTHON_TOOLS is declared."""
        with caplog.at_level(logging.WARNING):
            mcp_tools, python_tools = load_tools_from_file(tmp_tools_file_no_public)
        assert mcp_tools is None
        assert python_tools is None
        assert "no tools will be registered" in caplog.text.lower()

    def test_warns_on_missing_hints(self, tmp_tools_file_no_hints: Path, caplog) -> None:
        """Warning logged when a function parameter lacks type hints."""
        with caplog.at_level(logging.WARNING):
            mcp_tools, python_tools = load_tools_from_file(tmp_tools_file_no_hints)
        assert len(mcp_tools) == 1
        assert len(python_tools) == 1
        assert "no type hint" in caplog.text.lower() or "no docstring" in caplog.text.lower()

    def test_raises_on_invalid_list(self, tmp_path: Path) -> None:
        """ValueError when MCP_TOOLS contains non-callable entries."""
        tools = tmp_path / "tools.py"
        tools.write_text(
            "def foo() -> int:\n    \"\"\"Foo.\"\"\"\n    return 1\n\nMCP_TOOLS = [foo, 42]\nPYTHON_TOOLS = [foo]\n"
        )
        with pytest.raises(ValueError, match="MCP_TOOLS"):
            load_tools_from_file(tools)

    def test_none_when_list_absent(self, tmp_path: Path) -> None:
        """None returned for a list that is not declared; the other list works fine."""
        tools = tmp_path / "tools.py"
        tools.write_text(
            "def foo() -> int:\n    \"\"\"Foo.\"\"\"\n    return 1\n\nMCP_TOOLS = [foo]\n"
        )
        mcp_tools, python_tools = load_tools_from_file(tools)
        assert mcp_tools is not None and len(mcp_tools) == 1
        assert python_tools is None


class TestLoadModuleExports:
    """Tests for :func:`load_module_exports`."""

    def test_returns_tools_and_no_hook(self, tmp_tools_file: Path) -> None:
        """Tools load and the hook is None when not declared."""
        mcp_tools, python_tools, hook = load_module_exports(tmp_tools_file)
        assert {f.__name__ for f in mcp_tools} == {"read_temperature", "set_voltage"}
        assert [f.__name__ for f in python_tools] == ["read_temperature"]
        assert hook is None

    def test_hook_shares_module_state_with_tools(self, tmp_path: Path) -> None:
        """The hook and the tools come from a single module evaluation.

        A tool that mutates module state must be observed by the hook — proving
        they close over the same module (the bug the combined loader fixes).
        """
        tools = tmp_path / "tools.py"
        tools.write_text(
            textwrap.dedent("""\
            _state = {"open": False}

            def acquire() -> str:
                \"\"\"Open the resource.\"\"\"
                _state["open"] = True
                return "opened"

            def GRACEFUL_EXPERIMENT_SHUTDOWN():
                # records what it saw so the test can assert on it
                results.append(_state["open"])

            results = []
            MCP_TOOLS = [acquire]
            """)
        )
        mcp_tools, _, hook = load_module_exports(tools)
        acquire = mcp_tools[0]
        acquire()        # mutate module state via the tool
        hook()           # hook must see the mutation
        # Read the module's results list via the tool's own globals (its module
        # namespace) — robust to the stem+hash module name and test isolation.
        assert acquire.__globals__["results"] == [True]

    def test_same_stem_files_do_not_collide(self, tmp_path: Path) -> None:
        """Two tools files sharing a basename load as distinct modules (no
        sys.modules key collision), so one does not clobber the other."""
        a = tmp_path / "projA" / "tools.py"
        b = tmp_path / "projB" / "tools.py"
        a.parent.mkdir()
        b.parent.mkdir()
        a.write_text(
            'def fa() -> int:\n    """A."""\n    return 1\n\nMCP_TOOLS = [fa]\n'
        )
        b.write_text(
            'def fb() -> int:\n    """B."""\n    return 2\n\nMCP_TOOLS = [fb]\n'
        )
        (ta, _, _), (tb, _, _) = load_module_exports(a), load_module_exports(b)
        # Each file kept its own tool — no cross-contamination.
        assert ta[0].__name__ == "fa"
        assert tb[0].__name__ == "fb"
        assert ta[0].__module__ != tb[0].__module__  # distinct module names

    def test_raises_on_non_callable_hook(self, tmp_path: Path) -> None:
        """ValueError when GRACEFUL_EXPERIMENT_SHUTDOWN is not callable."""
        tools = tmp_path / "tools.py"
        tools.write_text(
            "def foo() -> int:\n    \"\"\"Foo.\"\"\"\n    return 1\n\n"
            "MCP_TOOLS = [foo]\nGRACEFUL_EXPERIMENT_SHUTDOWN = 42\n"
        )
        with pytest.raises(ValueError, match="GRACEFUL_EXPERIMENT_SHUTDOWN"):
            load_module_exports(tools)
