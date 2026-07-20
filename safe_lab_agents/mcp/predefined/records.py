"""Format-neutral serialization helpers shared by auto-log, Kadi4Mat, and the
``.eln`` exporter.

This module has **no dependency on kadi-apy** and is safe to import from the
local-logging path even when Kadi4Mat is not configured.  It owns three things:

* :func:`json_safe` — coerce arbitrary values to JSON-serializable ones — and
  :func:`json_safe_keep_refs`, its variant that leaves ndarray reference dicts
  intact for consumers that resolve them later.
* the **quantity** convention — :func:`quantity` / :func:`is_quantity` /
  :func:`split_quantity` — a ``{"value": …, "unit": …}`` dict that lets a tool
  attach a unit of measurement to a numeric (or array) result.
* the single canonical numpy-array extractor :func:`extract_arrays`, which
  writes arrays to HDF5 (with a NeXus-style ``units`` attribute) and replaces
  them with a reference dict — together with the reference-dict vocabulary
  every consumer shares: :func:`is_array_ref`, :func:`array_shape_str`, and
  :func:`ndarray_summary`.
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
    if isinstance(native, np.ndarray):
        # extract_arrays should have replaced every array with a reference dict
        # before json_safe ever sees the value; if one slips through it would be
        # stringified (truncated, unrecoverable), so surface it loudly.
        logger.warning(
            "json_safe: a numpy array reached json_safe un-extracted and was "
            "stringified (shape=%s dtype=%s) — data lost", native.shape, native.dtype
        )
    return str(native)


def json_safe_keep_refs(value: Any) -> Any:
    """:func:`json_safe`, but ndarray reference dicts pass through untouched.

    The post-:func:`extract_arrays` serializer: arrays have already been written
    to HDF5 and replaced by reference dicts, which downstream consumers (report,
    ``.eln``, Kadi push) must still recognize as arrays rather than as ordinary
    nested dicts.  Recurses through dicts and lists so a reference nested at any
    depth survives.
    """
    if is_array_ref(value):
        return value
    if isinstance(value, dict):
        return {k: json_safe_keep_refs(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe_keep_refs(v) for v in value]
    return json_safe(value)


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


def _is_leaf(value: Any) -> bool:
    """A value that :func:`flatten_record` should NOT descend into.

    Scalars, quantities, and ndarray reference dicts are leaves; plain dicts and
    lists are containers to flatten.
    """
    if is_quantity(value) or is_array_ref(value):
        return True
    return not isinstance(value, (dict, list, tuple))


def flatten_record(mapping: dict | None) -> dict[str, Any]:
    """Flatten a params/result mapping into ``{dotted_key: leaf}`` pairs.

    Nested dicts and lists are flattened with dotted / indexed keys that mirror
    :func:`extract_arrays`'s dataset naming (``scan.x``, ``traces.0``), so each
    nested array or scalar becomes its own leaf.  Quantities and ndarray
    reference dicts are kept whole (treated as leaves).  Consumers that produce
    one metadata entry per value (the ``.eln`` exporter, the Kadi push) call this
    so arrays nested below the top level get first-class treatment instead of
    being embedded as raw reference dicts.
    """
    out: dict[str, Any] = {}

    def _walk(value: Any, path: str) -> None:
        if _is_leaf(value):
            out[path] = value
        elif isinstance(value, dict):
            for key, sub in value.items():
                _walk(sub, f"{path}.{key}" if path else str(key))
        else:  # list / tuple
            for i, sub in enumerate(value):
                _walk(sub, f"{path}.{i}" if path else str(i))

    for key, value in (mapping or {}).items():
        _walk(value, str(key))
    return out


# ---------------------------------------------------------------------------
# numpy array extraction → HDF5
# ---------------------------------------------------------------------------


def is_array_ref(value: Any) -> bool:
    """Return ``True`` if *value* is an ndarray reference dict from :func:`extract_arrays`."""
    return isinstance(value, dict) and value.get("_type") == "ndarray"


def array_shape_str(ref: dict[str, Any]) -> str:
    """Render a reference dict's shape as ``2×3`` (empty string for a 0-d array)."""
    return "×".join(str(s) for s in ref.get("shape", []))


def ndarray_summary(ref: dict[str, Any], *, include_unit: bool = True) -> str:
    """Render a reference dict as ``ndarray[2×3] float64 (W)``.

    The one canonical human-readable form for an array that cannot be inlined:
    used by the Kadi push (string extras), the ``.eln`` export, and the HTML
    report.  Pass ``include_unit=False`` where the unit is carried in a
    dedicated field instead (the ``.eln`` ``unitText``), so it is not repeated
    inside the value text.
    """
    summary = f"ndarray[{array_shape_str(ref)}] {ref.get('dtype', '')}".strip()
    unit = ref.get("unit")
    if include_unit and unit:
        summary += f" ({unit})"
    return summary


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
    """Return ``True`` if :func:`extract_arrays` would write any HDF5 dataset for
    *value* — i.e. *value* contains a numpy array (or array-valued quantity)
    anywhere, at any nesting depth in its dicts/lists.

    Lets a caller skip opening an HDF5 file for array-free (scalar-only) results.
    Must match :func:`extract_arrays`'s recursion so callers that keep one HDF5
    file open (batch logging) open it whenever — and only when — an array exists.
    """
    if _array_and_unit(value) is not None:
        return True
    if isinstance(value, dict):
        return any(has_arrays(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(has_arrays(v) for v in value)
    return False


def _write_array(value: Any, dataset: str, f: Any, h5_filename: str) -> dict[str, Any]:
    """Write an array (or array-quantity) *value* to *dataset* in *f*; return its ref."""
    arr, unit = _array_and_unit(value)  # type: ignore[misc]  # caller guarantees not None
    dset = f.create_dataset(dataset.lstrip("/"), data=arr)
    if unit:
        dset.attrs["units"] = unit
    return _array_ref(h5_filename, dataset, arr, unit)


def _extract_walk(value: Any, path: str, f: Any, h5_filename: str) -> Any:
    """Recursively replace arrays in *value* with refs, writing them under *path*.

    Dicts descend as ``<path>/<key>`` and lists/tuples as ``<path>/<index>`` so
    every nested array gets a unique HDF5 dataset name.
    """
    if _array_and_unit(value) is not None:
        return _write_array(value, path, f, h5_filename)
    if isinstance(value, dict):
        return {k: _extract_walk(v, f"{path}/{k}", f, h5_filename) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_extract_walk(v, f"{path}/{i}", f, h5_filename) for i, v in enumerate(value)]
    return value


def extract_arrays(
    result: Any, h5_path: Path, group: str, h5_file: Any = None
) -> Any:
    """Extract numpy arrays from *result* into *h5_path* under *group*.

    Returns a modified copy of *result* with every numpy array replaced by a
    reference dict, recursing through nested dicts and lists so arrays at any
    depth are captured (``{"traces": [arr, arr]}``, ``{"scan": {"x": arr}}``).
    A unit (from an array-valued quantity) is written as the HDF5 dataset's
    ``units`` attribute (NeXus convention) and carried onto the reference dict.
    A bare top-level array keeps the historical ``<group>/data`` dataset name;
    nested arrays get unique ``<group>/<key>`` / ``<group>/<index>`` names.
    Scalar quantities and other (non-array) values are left untouched.

    *h5_file* is an optional caller-managed open HDF5 handle (used by batch
    logging to keep one file open across many calls); when given, arrays are
    written into it instead of reopening *h5_path* each call.  The reference
    dicts always name the file by ``h5_path.name`` regardless.
    """
    h5_filename = h5_path.name

    # Skip opening a file at all for array-free (e.g. scalar-only) results.
    if not has_arrays(result):
        return result

    with _h5_target(h5_path, h5_file) as f:
        # A bare top-level array/quantity keeps the historical dataset name.
        if _array_and_unit(result) is not None:
            return _write_array(result, f"{group}/data", f, h5_filename)
        return _extract_walk(result, group, f, h5_filename)
