"""Safe deserialization and validation of tool call arguments for the /invoke endpoint.

The agent sends JSON; numpy arrays are encoded as
{"__type__": "ndarray", "data": <base64 of np.save() bytes>}.
np.load(..., allow_pickle=False) is used — cannot execute arbitrary code.
"""

from __future__ import annotations

import base64
import inspect
import io
import json
import types as _types
import typing
from typing import Any, Callable

import numpy as np


def _is_union(expected: Any) -> bool:
    """Return True for both typing.Union/Optional and Python 3.10+ X | Y unions."""
    if typing.get_origin(expected) is typing.Union:
        return True
    if hasattr(_types, "UnionType") and isinstance(expected, _types.UnionType):
        return True
    return False


def decode_args(obj: Any) -> Any:
    """Recursively reconstruct Python objects from the JSON-encoded args."""
    if isinstance(obj, dict) and obj.get("__type__") == "ndarray":
        return np.load(io.BytesIO(base64.b64decode(obj["data"])), allow_pickle=False)
    if isinstance(obj, list):
        return [decode_args(x) for x in obj]
    if isinstance(obj, dict):
        return {k: decode_args(v) for k, v in obj.items()}
    return obj


def loads_args(data: bytes) -> dict:
    """Parse a JSON request body, decoding any embedded numpy arrays."""
    return decode_args(json.loads(data.decode()))


def validate_and_coerce_args(func: Callable, args: dict) -> dict:
    """Validate args against the function's type hints and coerce where safe.

    Rules:
    - Parameters without type hints are passed through unchanged.
    - Parameters absent from *args* (will use default) are skipped.
    - ``int`` is coerced to ``float`` when the hint is ``float``.
    - ``Optional[X]`` / ``Union[X, None]`` accepts ``None`` or ``X``.
    - Only the FIRST level of a hint is validated: generic containers
      (``list[int]``, ``tuple[int]``, ``dict[str, float]``) are checked at the
      container level only — element types are never inspected. JSON arrays are
      accepted for any sequence hint (list/tuple/set) and coerced to that type.

    Raises:
        TypeError: with a human-readable message if a value doesn't match.
    """
    try:
        hints = typing.get_type_hints(func)
    except Exception:
        return args  # unresolvable hints (e.g. forward refs) — skip

    result = dict(args)
    for name in inspect.signature(func).parameters:
        if name not in args or name not in hints:
            continue
        result[name] = _coerce_or_raise(name, args[name], hints[name])
    return result


def _coerce_or_raise(name: str, value: Any, expected: Any) -> Any:
    origin = typing.get_origin(expected)

    # Union[X, Y, ...] — includes Optional[X] and Python 3.10+ X | Y
    if _is_union(expected):
        for t in typing.get_args(expected):
            try:
                return _coerce_or_raise(name, value, t)
            except TypeError:
                continue
        type_names = " | ".join(
            "None" if t is type(None) else getattr(t, "__name__", str(t))
            for t in typing.get_args(expected)
        )
        raise TypeError(f"'{name}': expected {type_names}, got {type(value).__name__}")

    # NoneType
    if expected is type(None):
        if value is None:
            return value
        raise TypeError(f"'{name}': expected None, got {type(value).__name__}")

    # Coerce int → float (JSON sends bare integers; hint says float)
    if expected is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)

    # A JSON boolean must not satisfy an ``int`` hint: bool is a subclass of int,
    # so the generic isinstance check below would otherwise accept true/false.
    if expected is int and isinstance(value, bool):
        raise TypeError(f"'{name}': expected int, got bool")

    # Only the FIRST level of a hint is validated — element types inside a
    # generic (the ``int`` in ``list[int]`` / ``dict[str, int]``) are never
    # inspected. This is deliberate leniency; see the safety note in the README
    # ("Python tool client"). ``container`` is the concrete type to check
    # against, whether the hint is a bare ``list`` or a generic ``list[int]``.
    container = origin if origin is not None else expected

    # JSON has no tuple/set type — every array arrives as a list. Accept a list
    # for any sequence hint and coerce it to the declared container type.
    if container in (list, tuple, set, frozenset):
        if not isinstance(value, list):
            raise TypeError(
                f"'{name}': expected {container.__name__}, got {type(value).__name__}"
            )
        return value if container is list else container(value)

    if not isinstance(value, container):
        type_name = getattr(expected, "__name__", str(expected))
        raise TypeError(
            f"'{name}': expected {type_name}, got {type(value).__name__}"
        )
    return value
