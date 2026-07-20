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

import contextlib
import logging
import threading
from pathlib import Path
from typing import Any

import h5py
import numpy as np

logger = logging.getLogger(__name__)

# libhdf5 is not safe for concurrent access to a file: in batch mode every tool
# call appends to the same ``.h5``, and tool calls can run on parallel threads
# (FastMCP worker pool + the ``/invoke`` HTTP endpoint).  Serialize all HDF5
# writes process-wide.  Held only around the short ``h5py.File`` open/write, so
# it does not add meaningful latency to the tool-return critical path.
_h5_lock = threading.Lock()


@contextlib.contextmanager
def _h5_target(h5_path: Path, h5_file: Any):
    """Yield an HDF5 handle for writing, serialized process-wide by ``_h5_lock``.

    If *h5_file* is a caller-managed open handle — used by batch logging to keep
    one file open across many tool calls and avoid a per-call open/close — write
    into it and leave it open (the caller closes it when the batch stops).  A
    batch's JSON metadata is buffered in memory until ``stop_batch`` anyway, so
    there is deliberately no per-call flush here: it would add cost without
    changing the batch's all-or-nothing-at-stop durability.  If *h5_file* is
    ``None`` (individual records), open *h5_path* for append and close it.
    """
    with _h5_lock:
        if h5_file is not None:
            yield h5_file
        else:
            with h5py.File(str(h5_path), "a") as f:
                yield f


# ---------------------------------------------------------------------------
# JSON coercion
# ---------------------------------------------------------------------------


def to_native_scalar(value: Any) -> Any:
    """Convert a numpy scalar to its Python-native equivalent; pass others through.

    numpy's ``int64``/``float32``/``bool_`` are **not** subclasses of Python's
    ``int``/``float``/``bool`` (only ``float64`` subclasses ``float``), so
    without this they would be stringified by :func:`json_safe` and misclassified
    as ``str`` downstream — losing the numeric type and any attached unit.
    """
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def json_safe(value: Any) -> Any:
    """Convert a value to something JSON-serializable."""
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    native = to_native_scalar(value)
    if isinstance(native, (bool, int, float)):
        return native
    return str(native)


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


def has_arrays(value: Any) -> bool:
    """Return ``True`` if :func:`extract_arrays` would write an HDF5 dataset for
    *value* (a bare array/array-quantity, or a dict with such a top-level value).

    Lets a caller skip opening an HDF5 file for array-free (scalar-only) results.
    """
    if _array_and_unit(value) is not None:
        return True
    if isinstance(value, dict):
        return any(_array_and_unit(v) is not None for v in value.values())
    return False


def extract_arrays(
    result: Any, h5_path: Path, group: str, h5_file: Any = None
) -> Any:
    """Extract numpy arrays from *result* into *h5_path* under *group*.

    Returns a modified copy of *result* with arrays replaced by reference
    dicts.  Handles a bare ndarray, a bare array-valued quantity, and the
    top-level values of a dict (plain arrays or array-valued quantities).
    A unit (from an array-valued quantity) is written as the HDF5 dataset's
    ``units`` attribute (NeXus convention) and carried onto the reference dict.
    Scalar quantities and deeper nesting are left untouched.

    *h5_file* is an optional caller-managed open HDF5 handle (used by batch
    logging to keep one file open across many calls); when given, arrays are
    written into it instead of reopening *h5_path* each call.  The reference
    dicts always name the file by ``h5_path.name`` regardless.
    """
    h5_filename = h5_path.name

    # Bare ndarray or bare array-valued quantity.
    bare = _array_and_unit(result)
    if bare is not None:
        arr, unit = bare
        dataset = f"{group}/data"
        with _h5_target(h5_path, h5_file) as f:
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
        with _h5_target(h5_path, h5_file) as f:
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
