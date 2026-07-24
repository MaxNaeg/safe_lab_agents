"""Tests for safe_lab_agents.mcp.serialization."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pytest

from safe_lab_agents.mcp.serialization import validate_and_coerce_args


def _make_func(**hints):
    """Build a dummy function with given parameter type hints."""
    params = ", ".join(hints)
    src = f"def f({params}): pass\n"
    annotations = {k: v for k, v in hints.items()}
    ns: dict = {}
    exec(src, ns)  # noqa: S102
    ns["f"].__annotations__ = annotations
    return ns["f"]


class TestValidateAndCoerceArgs:

    def test_passes_matching_float(self):
        def f(x: float): pass
        assert validate_and_coerce_args(f, {"x": 3.14}) == {"x": 3.14}

    def test_coerces_int_to_float(self):
        def f(voltage: float): pass
        result = validate_and_coerce_args(f, {"voltage": 5})
        assert result["voltage"] == 5.0
        assert isinstance(result["voltage"], float)

    def test_passes_matching_int(self):
        def f(n: int): pass
        assert validate_and_coerce_args(f, {"n": 42}) == {"n": 42}

    def test_rejects_float_for_int(self):
        def f(n: int): pass
        with pytest.raises(TypeError, match="'n'"):
            validate_and_coerce_args(f, {"n": 3.14})

    def test_rejects_bool_for_int(self):
        """A JSON bool must not satisfy an int hint (bool subclasses int)."""
        def f(n: int): pass
        with pytest.raises(TypeError, match="got bool"):
            validate_and_coerce_args(f, {"n": True})

    def test_bool_still_accepted_for_bool(self):
        def f(flag: bool): pass
        assert validate_and_coerce_args(f, {"flag": True}) == {"flag": True}

    def test_passes_matching_str(self):
        def f(s: str): pass
        assert validate_and_coerce_args(f, {"s": "hello"}) == {"s": "hello"}

    def test_rejects_int_for_str(self):
        def f(s: str): pass
        with pytest.raises(TypeError, match="'s'"):
            validate_and_coerce_args(f, {"s": 42})

    def test_passes_numpy_array(self):
        def f(data: np.ndarray): pass
        arr = np.array([1.0, 2.0])
        assert validate_and_coerce_args(f, {"data": arr})["data"] is arr

    def test_rejects_list_for_ndarray(self):
        def f(data: np.ndarray): pass
        with pytest.raises(TypeError, match="'data'"):
            validate_and_coerce_args(f, {"data": [1.0, 2.0]})

    def test_optional_accepts_none(self):
        def f(x: Optional[float]): pass
        result = validate_and_coerce_args(f, {"x": None})
        assert result["x"] is None

    def test_optional_accepts_value(self):
        def f(x: Optional[float]): pass
        result = validate_and_coerce_args(f, {"x": 1.5})
        assert result["x"] == 1.5

    def test_optional_coerces_int_to_float(self):
        def f(x: Optional[float]): pass
        result = validate_and_coerce_args(f, {"x": 3})
        assert isinstance(result["x"], float)

    def test_optional_rejects_wrong_type(self):
        def f(x: Optional[float]): pass
        with pytest.raises(TypeError, match="'x'"):
            validate_and_coerce_args(f, {"x": "bad"})

    def test_skips_params_without_hints(self):
        def f(x, y: int): pass
        result = validate_and_coerce_args(f, {"x": "anything", "y": 1})
        assert result["x"] == "anything"

    def test_skips_missing_params(self):
        def f(x: float, y: float = 1.0): pass
        result = validate_and_coerce_args(f, {"x": 2.0})
        assert result == {"x": 2.0}

    def test_passes_list_hint(self):
        def f(items: list): pass
        assert validate_and_coerce_args(f, {"items": [1, 2, 3]})["items"] == [1, 2, 3]

    def test_rejects_non_list_for_list(self):
        def f(items: list): pass
        with pytest.raises(TypeError, match="'items'"):
            validate_and_coerce_args(f, {"items": "not a list"})

    def test_bool_not_coerced_to_float(self):
        def f(x: float): pass
        with pytest.raises(TypeError, match="'x'"):
            validate_and_coerce_args(f, {"x": True})

    # --- First-level-only (shallow) generic checks ---

    def test_generic_list_checks_container_only(self):
        """list[int] validates the container but NOT element types (lenient)."""
        def f(xs: list[int]): pass
        # Mismatched element types pass — only "is a list" is checked.
        assert validate_and_coerce_args(f, {"xs": [1, "two", 3.0]})["xs"] == [1, "two", 3.0]

    def test_generic_list_rejects_non_list(self):
        def f(xs: list[int]): pass
        with pytest.raises(TypeError, match="'xs'"):
            validate_and_coerce_args(f, {"xs": "nope"})

    def test_tuple_hint_accepts_json_list_and_coerces(self):
        """JSON has no tuple; a list is accepted for tuple[int] and coerced."""
        def f(pt: tuple[int, int]): pass
        result = validate_and_coerce_args(f, {"pt": [1, 2]})
        assert result["pt"] == (1, 2)
        assert isinstance(result["pt"], tuple)

    def test_set_hint_accepts_json_list_and_coerces(self):
        def f(tags: set[str]): pass
        result = validate_and_coerce_args(f, {"tags": ["a", "b", "a"]})
        assert result["tags"] == {"a", "b"}
        assert isinstance(result["tags"], set)

    def test_generic_dict_checks_container_only(self):
        """dict[str, int] validates the container but NOT key/value types."""
        def f(d: dict[str, int]): pass
        payload = {"a": 1, "b": "not-an-int"}
        assert validate_and_coerce_args(f, {"d": payload})["d"] == payload
