"""Tests for the experiment() lazy-experiment helper."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from safe_lab_agents import experiment
from safe_lab_agents.mcp import tool_utils
from safe_lab_agents.mcp.serialization import validate_and_coerce_args
from safe_lab_agents.mcp.tool_utils import results_to_shared


class FakeSetup:
    """A stand-in instrument used to exercise experiment()."""

    instances = 0

    def __init__(self, port: str = "sim", baud: int = 9600):
        FakeSetup.instances += 1
        self.port = port
        self.baud = baud
        self.closed = False

    def get_power(self, channel: int) -> float:
        """Read the power on a channel, in watts.

        Args:
            channel: Channel index to read.
        """
        return float(channel) + 0.5

    @staticmethod
    def units() -> str:
        """Return the measurement units."""
        return "W"

    @property
    def is_connected(self) -> bool:
        return not self.closed

    def close(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def _reset_instances():
    """Reset the construction counter before every test (function or method)."""
    FakeSetup.instances = 0
    yield


class TestLazyConstruction:
    def test_accessing_method_does_not_construct(self) -> None:
        """Referencing a method (for registration) must not build the object."""
        exp = experiment(FakeSetup)
        tool = exp.get_power          # the expression used in PYTHON_TOOLS/MCP_TOOLS
        assert FakeSetup.instances == 0
        assert callable(tool)

    def test_first_call_constructs_and_caches(self) -> None:
        """The first call constructs the object; later calls reuse it."""
        exp = experiment(FakeSetup)
        assert exp.get_power(3) == 3.5
        assert exp.get_power(4) == 4.5
        assert FakeSetup.instances == 1

    def test_forwards_constructor_args(self) -> None:
        """*args/**kwargs reach the factory."""
        exp = experiment(FakeSetup, "ttyUSB0", baud=115200)
        exp.get_power(1)              # trigger construction
        assert exp.port == "ttyUSB0"
        assert exp.baud == 115200

    def test_lambda_factory(self) -> None:
        """A non-class factory still works via construct-on-access."""
        exp = experiment(lambda: FakeSetup(port="cfg"))
        assert exp.get_power(2) == 2.5
        assert exp.port == "cfg"

    def test_property_read_returns_value(self) -> None:
        """Reading a property constructs the object and returns the real value."""
        exp = experiment(FakeSetup)
        assert exp.is_connected is True
        assert FakeSetup.instances == 1


class TestDirectRegistrationMetadata:
    def test_instance_method_signature_strips_self(self) -> None:
        """A method accessed off the proxy is registrable: clean name/doc/signature."""
        exp = experiment(FakeSetup)
        tool = exp.get_power
        assert tool.__name__ == "get_power"
        assert "power" in tool.__doc__.lower()
        params = list(inspect.signature(tool).parameters)
        assert params == ["channel"]            # self stripped
        assert FakeSetup.instances == 0          # still not constructed

    def test_staticmethod_keeps_signature(self) -> None:
        """A staticmethod keeps its (self-less) signature and is callable."""
        exp = experiment(FakeSetup)
        tool = exp.units
        assert list(inspect.signature(tool).parameters) == []
        assert tool() == "W"

    def test_coercion_uses_stripped_signature(self) -> None:
        """The arg-coercion layer validates against the stripped signature."""
        exp = experiment(FakeSetup)
        coerced = validate_and_coerce_args(exp.get_power, {"channel": 2})
        assert coerced == {"channel": 2}


class TestShutdownPattern:
    def test_close_via_hook(self) -> None:
        """The documented GRACEFUL_EXPERIMENT_SHUTDOWN pattern closes the object."""
        exp = experiment(FakeSetup)
        exp.get_power(1)              # construct
        exp.close()
        assert exp.is_connected is False


class TestResultsToShared:
    def test_no_return_statement_raises_clear_error(self) -> None:
        """A function with no value-returning `return` fails clearly at decoration."""
        def f(x: int):
            """Does nothing."""
            pass

        with pytest.raises(ValueError, match="return <names>"):
            results_to_shared()(f)

    def test_decorates_indented_function_and_saves(self, tmp_path, monkeypatch) -> None:
        """An indented (nested) function decorates without IndentationError (dedent).

        Also exercises the multi-value path: status passed through, array saved.
        """
        monkeypatch.setattr(tool_utils, "SHARED_DATA_DIR", str(tmp_path))

        @results_to_shared(results_to_save=[False, True])
        def measure(channel: int):
            """Return a status and an array."""
            return "ok", np.arange(3)

        status, saved_msg = measure(1)
        assert status == "ok"
        assert saved_msg.startswith("Saved result")
        assert any(tmp_path.iterdir())  # the array was written to the shared dir

    def test_requires_shared_dir_at_call_time(self, monkeypatch) -> None:
        """Calling a decorated tool without a shared dir raises a clear RuntimeError."""
        monkeypatch.setattr(tool_utils, "SHARED_DATA_DIR", None)

        @results_to_shared()
        def f(x: int):
            """Return x."""
            return x

        with pytest.raises(RuntimeError, match="requires a shared directory"):
            f(1)
