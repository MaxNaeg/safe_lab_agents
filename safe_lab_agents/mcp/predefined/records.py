"""Format-neutral serialization helpers shared by auto-log, Kadi4Mat, and the
``.eln`` exporter.

This module has **no dependency on kadi-apy** and is safe to import from the
local-logging path even when Kadi4Mat is not configured.  It owns three things:

* :func:`json_safe` — coerce arbitrary values to JSON-serializable ones.
* the **quantity** convention — :func:`quantity` / :func:`is_quantity` /
  :func:`split_quantity` — a ``{"value": …, "unit": …}`` dict that lets a tool
  attach a unit of measurement to a numeric (or array) result.
* the single canonical numpy-array extractor :func:`extract_arrays`, which
  writes arrays to HDF5 (with a NeXus-style ``units`` attribute) and replaces
  them with a reference dict.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import h5py
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JSON coercion
# ---------------------------------------------------------------------------


def json_safe(value: Any) -> Any:
    """Convert a value to something JSON-serializable."""
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return str(value)


# ---------------------------------------------------------------------------
# Quantities — a measurement value carrying a unit
# ---------------------------------------------------------------------------


def quantity(value: Any, unit: str, term: str | None = None) -> dict[str, Any]:
    """Wrap a measurement *value* with its *unit* (and optional ontology *term*).

    Tools opt a numeric or array result into carrying a unit by returning this
    dict, e.g. ``{"power": quantity(2.5, "W")}``.  ``term`` is an optional IRI
    (e.g. a QUDT unit URI) for full semantic annotation; it is propagated to
    Kadi4Mat's ``term`` field and the ``.eln`` ``unitCode`` when present.
    """
    q: dict[str, Any] = {"value": value, "unit": unit}
    if term:
        q["term"] = term
    return q


def is_quantity(value: Any) -> bool:
    """Return ``True`` if *value* is a quantity dict (``value`` + ``unit``).

    An ndarray *reference* dict (``_type == "ndarray"``) is **not** a quantity,
    even though it may carry a ``unit`` key.
    """
    return (
        isinstance(value, dict)
        and "value" in value
        and "unit" in value
        and value.get("_type") != "ndarray"
    )


def split_quantity(value: dict[str, Any]) -> tuple[Any, str | None, str | None]:
    """Return ``(value, unit, term)`` for a quantity dict."""
    return value.get("value"), value.get("unit"), value.get("term")


# ---------------------------------------------------------------------------
# numpy array extraction → HDF5
# ---------------------------------------------------------------------------


def _array_ref(
    h5_filename: str, dataset: str, arr: np.ndarray, unit: str | None
) -> dict[str, Any]:
    """Build the canonical ndarray reference dict for an extracted array."""
    ref: dict[str, Any] = {
        "_type": "ndarray",
        "file": h5_filename,
        "dataset": dataset,
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
    }
    if unit:
        ref["unit"] = unit
    return ref


def _array_and_unit(value: Any) -> tuple[np.ndarray, str | None] | None:
    """If *value* is an array or an array-valued quantity, return (array, unit)."""
    if isinstance(value, np.ndarray):
        return value, None
    if is_quantity(value) and isinstance(value.get("value"), np.ndarray):
        return value["value"], value.get("unit")
    return None


def extract_arrays(result: Any, h5_path: Path, group: str) -> Any:
    """Extract numpy arrays from *result* into *h5_path* under *group*.

    Returns a modified copy of *result* with arrays replaced by reference
    dicts.  Handles a bare ndarray, a bare array-valued quantity, and the
    top-level values of a dict (plain arrays or array-valued quantities).
    A unit (from an array-valued quantity) is written as the HDF5 dataset's
    ``units`` attribute (NeXus convention) and carried onto the reference dict.
    Scalar quantities and deeper nesting are left untouched.
    """
    h5_filename = h5_path.name

    # Bare ndarray or bare array-valued quantity.
    bare = _array_and_unit(result)
    if bare is not None:
        arr, unit = bare
        dataset = f"{group}/data"
        with h5py.File(str(h5_path), "a") as f:
            dset = f.create_dataset(dataset.lstrip("/"), data=arr)
            if unit:
                dset.attrs["units"] = unit
        return _array_ref(h5_filename, dataset, arr, unit)

    if isinstance(result, dict):
        to_extract: dict[str, tuple[np.ndarray, str | None]] = {}
        for key, value in result.items():
            arr_unit = _array_and_unit(value)
            if arr_unit is not None:
                to_extract[key] = arr_unit
        if not to_extract:
            return result
        with h5py.File(str(h5_path), "a") as f:
            for key, (arr, unit) in to_extract.items():
                dset = f.create_dataset(f"{group}/{key}".lstrip("/"), data=arr)
                if unit:
                    dset.attrs["units"] = unit
        modified: dict[str, Any] = {}
        for key, value in result.items():
            if key in to_extract:
                arr, unit = to_extract[key]
                modified[key] = _array_ref(h5_filename, f"{group}/{key}", arr, unit)
            else:
                modified[key] = value
        return modified

    return result
