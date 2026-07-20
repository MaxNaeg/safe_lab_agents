"""Auto-log: wraps every user tool call and records inputs, outputs, and timestamps
as structured ELN entries (JSON + HDF5 for arrays).  Activated with ``--auto-log``
on the CLI.

Individual records
------------------
Each call outside a batch creates two files in the output directory::

    {exp_id}-{tool_name}.json   – metadata, scalars, array references
    {exp_id}-{tool_name}.h5     – HDF5 file with one dataset per array
                                   (only written when arrays are present)

Array references in JSON look like::

    {"_type": "ndarray", "file": "…h5", "dataset": "/spectrum",
     "shape": [1024], "dtype": "float64"}

Reload: ``h5py.File(ref["file"], "r")[ref["dataset"]][:]``

Batch records
-------------
``start_batch(label)`` / ``stop_batch()`` group multiple tool calls into a
single merged record::

    {batch_id}.json   – all experiments inline
    {batch_id}.h5     – all arrays in one file, grouped by experiment ID
                        Dataset paths: /{exp_id}/{key}

``stop_batch()`` writes the JSON; the HDF5 file is built incrementally
during the batch, so array data is safe even if the session crashes before
``stop_batch()`` is called.

Tool return values
------------------
Tools should return a ``dict`` when ``--auto-log`` is enabled.  Non-dict
return values (bare scalars, tuples, lists) are recorded but produce a
less structured JSON entry.
"""

from __future__ import annotations

import functools
import inspect
import json
import logging
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import h5py

from safe_lab_agents.mcp.predefined.records import (
    extract_arrays,
    flatten_record,
    has_arrays,
    json_safe,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level state (set by make_autolog_wrapper in the server subprocess)
# ---------------------------------------------------------------------------

_output_dir: Path | None = None
# ``_Batch`` is defined below; ``from __future__ import annotations`` keeps this a
# forward reference so the annotation resolves without a runtime NameError.
_current_batch: _Batch | None = None
_kadi_client: Any = (
    None  # KadiClient; set by make_autolog_wrapper when KADI4MAT_PROJECT is set
)

# Guards the mutable batch state above.  Tools can run on parallel threads
# (FastMCP worker pool + the ``/invoke`` HTTP endpoint), so the check-then-set
# in ``start_batch``/``stop_batch`` and the ``batch.experiments.append`` in
# ``_record_call`` would otherwise race.  Held only around the in-memory
# pointer/list ops — never across HDF5 writes or the Kadi push.
_state_lock = threading.Lock()


def no_autolog(func: Callable) -> Callable:
    """Decorator to opt a specific tool out of auto-log recording."""
    func._no_autolog = True  # type: ignore[attr-defined]
    return func


def _int_env(name: str, default: int) -> int:
    """Read an integer env var, falling back to *default* (with a warning) on a
    missing/empty/unparseable value — a bad value must not crash server startup."""
    raw = os.environ.get(name, "")
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning(
            "Invalid %s=%r (expected an integer); using default %d",
            name, raw, default,
        )
        return default


@dataclass
class _Batch:
    id: str
    label: str
    description: str
    started_at: str
    h5_path: Path
    experiments: list[dict] = field(default_factory=list)
    # Open HDF5 handle, lazily opened on the first array write and kept open for
    # the batch's lifetime so a sweep of thousands of calls doesn't reopen the
    # same file every time; closed in ``stop_batch``.
    h5_file: Any = None


# ---------------------------------------------------------------------------
# Batch tools — registered in _python_registry by server.py when AUTO_LOG_DIR
# is set; callable from auto_log_client.py inside Docker via /invoke.
# ---------------------------------------------------------------------------


def start_batch(label: str, description: str = "") -> str:
    """Start collecting experiment results into a single ELN batch record.

    All MCP and Python tool calls made after start_batch() and before
    stop_batch() are grouped into one merged ELN record instead of
    creating individual records.

    Use this when running:
    - Parameter sweeps (iterating a tool over a range of values)
    - Optimisation loops (e.g. Bayesian optimisation, gradient descent
      on instrument settings)
    - Multi-step protocols where several tool calls form one logical
      experiment (calibrate → acquire → verify)
    - Any experiment you would describe as "N runs as part of one study"

    Args:
        label: Short label for this batch, e.g. "Voltage sweep 0–5 V".
        description: Optional longer description.
    """
    global _current_batch
    if _output_dir is None:
        return "Auto-log is not initialised (AUTO_LOG_DIR not set)."
    with _state_lock:
        if _current_batch is not None:
            return (
                f"A batch is already active: '{_current_batch.label}'. "
                "Call stop_batch() first."
            )
        batch_id = f"batch_{datetime.now(timezone.utc):%Y%m%d_%H%M%S_%f}"
        _current_batch = _Batch(
            id=batch_id,
            label=label,
            description=description,
            started_at=datetime.now(timezone.utc).isoformat(),
            h5_path=_output_dir / f"{batch_id}.h5",
        )
    return (
        f"Batch '{label}' started. All subsequent tool calls will be "
        "collected into a single ELN record until stop_batch() is called."
    )


def stop_batch() -> str:
    """Finalise the active batch and write a merged ELN record to disk.

    Returns a summary with the output file path and experiment count.
    """
    global _current_batch
    with _state_lock:
        if _current_batch is None:
            return "No batch is active. Call start_batch() first."
        batch = _current_batch
        _current_batch = None

    # Close the shared HDF5 handle kept open across the batch; this flushes the
    # arrays to disk so the ``h5_path.exists()`` check below (and any downstream
    # reader) sees a complete file.
    if batch.h5_file is not None:
        try:
            batch.h5_file.close()
        except Exception:
            logger.warning(
                "auto-log: failed to close batch HDF5 file %s",
                batch.h5_path,
                exc_info=True,
            )
        batch.h5_file = None

    completed_at = datetime.now(timezone.utc).isoformat()
    record = {
        "id": batch.id,
        "type": "batch",
        "label": batch.label,
        "description": batch.description,
        "started_at": batch.started_at,
        "completed_at": completed_at,
        "experiment_count": len(batch.experiments),
        "experiments": batch.experiments,
    }
    if batch.h5_path.exists():
        record["h5_file"] = batch.h5_path.name

    out_dir = _output_dir or Path(".")
    json_path = out_dir / f"{batch.id}.json"
    try:
        json_path.write_text(
            json.dumps(record, indent=2, default=str), encoding="utf-8"
        )
    except Exception:
        logger.warning(
            "auto-log: failed to write batch record %s", json_path, exc_info=True
        )
        return f"Error: failed to write batch record to {json_path}"

    _push_to_kadi(record, out_dir)

    count = len(batch.experiments)
    msg = (
        f"Batch '{batch.label}' saved "
        f"({count} experiment{'s' if count != 1 else ''}) → {json_path}"
    )
    if batch.h5_path.exists():
        msg += f"\n   Arrays → {batch.h5_path.name}"
    return msg


def flush_active_batch() -> str | None:
    """Persist a batch the agent left open, if any (used at server shutdown/reload).

    A batch's experiments live only in memory until ``stop_batch`` writes them,
    and that state lives in the MCP server subprocess — so if the agent forgets
    to stop the batch, it must be flushed here (in the subprocess) before exit,
    or the experiments are dropped and the arrays already in the batch ``.h5``
    are orphaned.  Returns the ``stop_batch`` summary if a batch was flushed, or
    ``None`` if none was active.
    """
    if _current_batch is None:
        return None
    logger.info("auto-log: flushing active batch '%s' at shutdown", _current_batch.label)
    return stop_batch()


def log_analysis(
    title: str,
    text: str = "",
    data: dict | None = None,
    references: list | None = None,
    script: str = "",
    figures: list | None = None,
    kind: str = "analysis",
) -> str:
    """Record analysis results as a structured ELN entry and push to Kadi4Mat.

    Called from Python scripts inside Docker via ``auto_log_client.log_analysis()``.
    This function runs on the **host** MCP server, so the following constraints apply:

    - ``data`` values must be JSON-serializable or ``numpy.ndarray``. Other
      types (pandas DataFrames, arbitrary objects) are not supported.
    - ``figures`` must be filenames of files already saved to ``AUTO_LOG_DIR``
      inside Docker (the shared bind mount). Full paths elsewhere in the
      container are inaccessible to the host.

    Args:
        title: Short title, e.g. "Linear fit of voltage sweep".
        text: Free-text narrative, observations, or conclusions (markdown OK).
        data: Dict of analysis results. numpy arrays are saved to HDF5;
              scalars and strings are stored as JSON metadata.
        references: List of exp_*/batch_*/analysis_* IDs this analysis is
                    based on.
        script: Python source code used to produce this analysis.
        figures: Filenames (not full paths) of figures already saved to
                 AUTO_LOG_DIR. E.g. ``["fit.png"]``.
        kind: What sort of record this is. One of "analysis" (default,
              a successful result/conclusion), "hypothesis" (intent before a
              measurement), "decision" (rationale for a choice), "debug" (a
              debugging step or failed-then-fixed iteration), "failed" (an
              attempt that did not succeed), or "observation" (anomaly,
              negative result, or next-step note). Record failures too — a
              failed attempt is data, not noise.
    """
    if _output_dir is None:
        return "Auto-log is not initialised (AUTO_LOG_DIR not set)."

    now = datetime.now(timezone.utc)
    entry_id = f"analysis_{now:%Y%m%d_%H%M%S_%f}"
    h5_path = _output_dir / f"{entry_id}.h5"

    data_out: dict[str, Any] = {}
    for k, v in (data or {}).items():
        extracted = extract_arrays(v, h5_path, f"/{k}")
        if isinstance(extracted, dict) and extracted.get("_type") == "ndarray":
            data_out[k] = extracted
        else:
            data_out[k] = json_safe(extracted)

    figure_refs = [{"_type": "figure", "file": Path(f).name} for f in (figures or [])]

    record: dict[str, Any] = {
        "type": "analysis",
        "id": entry_id,
        "title": title,
        "kind": kind,
        "timestamp": now.isoformat(),
        "text": text,
        "data": data_out,
        "references": list(references or []),
        "script": script,
        "figures": figure_refs,
    }
    if h5_path.exists():
        record["h5_file"] = h5_path.name

    json_path = _output_dir / f"{entry_id}.json"
    try:
        json_path.write_text(
            json.dumps(record, indent=2, default=str), encoding="utf-8"
        )
    except Exception:
        logger.warning("auto-log: failed to write analysis record", exc_info=True)
        return f"Error: failed to write analysis record to {json_path}"

    _push_to_kadi(record, _output_dir)
    return f"Analysis '{title}' saved → {json_path.name}"


# ---------------------------------------------------------------------------
# Kadi4Mat push (used when KADI4MAT_PROJECT is set alongside AUTO_LOG_DIR)
# ---------------------------------------------------------------------------


def _push_to_kadi(entry: dict, output_dir: Path) -> None:
    """Push a single auto-log entry to Kadi4Mat (fire-and-forget).

    Handles experiment records (have ``result`` + ``parameters``), batch
    records (have ``label``), and analysis records (have ``data`` + ``text``).
    """
    if _kadi_client is None:
        return
    title = entry.get("label") or entry.get("title", "unknown")
    call_args = {
        k.removeprefix("param_"): v
        for k, v in flatten_record(entry.get("parameters", {})).items()
    }
    # Analysis records use "data"; experiment/batch records use "result".
    # Flatten first so an array nested in a list/dict becomes its own extra
    # (dotted key, e.g. scan.x) instead of a stringified reference dict.
    raw_result = entry.get("result") or entry.get("data", {})
    result: dict[str, Any] = {}
    flat_result = flatten_record(raw_result) if isinstance(raw_result, dict) else {}
    for k, v in flat_result.items():
        if isinstance(v, dict) and v.get("_type") == "ndarray":
            # Arrays become a string summary; Kadi forbids a unit on a string
            # extra, so the unit rides along in the summary text (the machine-
            # readable unit lives in the attached HDF5 file's ``units`` attr).
            shape = "×".join(str(s) for s in v.get("shape", []))
            summary = f"ndarray[{shape}] {v.get('dtype', '')}".strip()
            if v.get("unit"):
                summary += f" ({v['unit']})"
            result[k] = summary
        else:
            # Plain values and scalar quantity dicts ({"value","unit"}) pass
            # through untouched; create_record turns quantities into a numeric
            # extra with Kadi's native ``unit`` field.
            result[k] = v
    if entry.get("text"):
        result["text"] = entry["text"]
    if entry.get("kind"):
        result["kind"] = entry["kind"]
    if entry.get("duration_ms") is not None:
        result["duration_ms"] = entry["duration_ms"]
    # Batch records carry no result/data — expose batch-level scalars as extras
    # so the Kadi record is searchable (per-experiment data stays in the files).
    if entry.get("type") == "batch":
        for key in ("experiment_count", "started_at", "completed_at", "description"):
            if entry.get(key) not in (None, ""):
                result[key] = entry[key]
    files: list[Path] = []
    # Always attach the full JSON record so every field is available in Kadi.
    entry_id = entry.get("id", "")
    if entry.get("type") == "individual":
        json_path = output_dir / f"{entry_id}-{entry.get('title', '')}.json"
    else:
        json_path = output_dir / f"{entry_id}.json"
    if json_path.exists():
        files.append(json_path)
    if h5_file := entry.get("h5_file"):
        files.append(output_dir / h5_file)
    for fig in entry.get("figures", []):
        fig_path = output_dir / (fig["file"] if isinstance(fig, dict) else fig)
        if fig_path.exists():
            files.append(fig_path)
    try:
        _kadi_client.create_record(
            title=title, call_args=call_args, result=result, files=files
        )
    except Exception:
        logger.warning(
            "auto-log: failed to push '%s' to Kadi4Mat", title, exc_info=True
        )


# ---------------------------------------------------------------------------
# Wrapper factory — called by server.py when AUTO_LOG_DIR is set
# ---------------------------------------------------------------------------


def make_autolog_wrapper() -> Callable[[Callable], Callable]:
    """Return a tool wrapper that logs every call to the auto-log output dir.

    Also initialises ``_output_dir`` (module-level) so that ``start_batch``
    and ``stop_batch`` (registered separately in the Python registry) share
    the same output directory.
    """
    global _output_dir, _kadi_client
    auto_log_dir = os.environ.get("AUTO_LOG_DIR", "")
    if not auto_log_dir:
        raise ValueError(
            "AUTO_LOG_DIR environment variable is required. "
            "Use --auto-log when starting a session."
        )
    _output_dir = Path(auto_log_dir)
    _output_dir.mkdir(parents=True, exist_ok=True)
    # The agent writes figures/data into this dir from *inside* the container
    # (auto_log_client saves files to AUTO_LOG_DIR), but the dir is created here
    # on the host and owned by the host user. The container's non-root `agent`
    # user falls in the "other" permission class, so the default 0775 blocks its
    # writes. Widen to 0777 (ownership unchanged, so the host-side logger keeps
    # writing too); a bind mount shares the inode, so the container sees it.
    try:
        _output_dir.chmod(0o777)
    except OSError as exc:  # pragma: no cover - depends on host fs/ownership
        logger.warning("Could not widen permissions on %s: %s", _output_dir, exc)

    if os.environ.get("KADI4MAT_PROJECT"):
        from safe_lab_agents.mcp.predefined.kadi4mat import KadiClient

        max_per_session_raw = _int_env("KADI4MAT_MAX_PER_SESSION", 500)
        _kadi_client = KadiClient(
            project=os.environ["KADI4MAT_PROJECT"],
            max_per_minute=_int_env("KADI4MAT_MAX_PER_MINUTE", 10),
            max_per_session=max_per_session_raw
            if max_per_session_raw > 0
            else float("inf"),
        )

    def wrapper(func: Callable) -> Callable:
        if getattr(func, "_no_autolog", False):
            return func

        @functools.wraps(func)
        def inner(*args: Any, **kwargs: Any) -> Any:
            start_time = datetime.now(timezone.utc)
            result = func(*args, **kwargs)
            end_time = datetime.now(timezone.utc)
            duration_ms = int((end_time - start_time).total_seconds() * 1000)

            try:
                sig = inspect.signature(func)
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()
                call_args = dict(bound.arguments)
            except Exception:
                call_args = {"args": str(args), "kwargs": str(kwargs)}

            exp_id = f"exp_{start_time:%Y%m%d_%H%M%S_%f}"

            out_dir = _output_dir
            if out_dir is None:
                return result

            # Snapshot the active batch under the lock so it can't be swapped
            # out from under us mid-record; the write itself happens unlocked.
            with _state_lock:
                batch = _current_batch

            try:
                _record_call(
                    exp_id=exp_id,
                    tool_name=func.__name__,
                    timestamp=start_time.isoformat(),
                    duration_ms=duration_ms,
                    call_args=call_args,
                    result=result,
                    batch=batch,
                    output_dir=out_dir,
                )
            except Exception:
                logger.warning(
                    "auto-log: failed to record call to '%s'",
                    func.__name__,
                    exc_info=True,
                )

            return result

        return inner

    return wrapper


# ---------------------------------------------------------------------------
# Session summary
# ---------------------------------------------------------------------------

_FIGURE_SUFFIXES = {".png", ".jpg", ".jpeg", ".svg", ".gif", ".pdf", ".webp"}


def write_session_summary(output_dir: Path) -> Path | None:
    """Read all entry JSON files and write ``session_summary.json`` + ``.zip``.

    Scans *output_dir* for ``exp_*.json``, ``batch_*.json``, and
    ``analysis_*.json`` files, merges them into a single summary sorted by
    timestamp, then creates a ZIP archive of everything in the directory.

    Returns the path to ``session_summary.json``, or ``None`` if no entries
    exist yet.
    """
    json_files = sorted(
        [
            *output_dir.glob("exp_*.json"),
            *output_dir.glob("batch_*.json"),
            *output_dir.glob("analysis_*.json"),
        ]
    )
    if not json_files:
        logger.info("auto-log: no entries in %s — skipping session summary", output_dir)
        return None

    entries = []
    for path in json_files:
        try:
            entries.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            logger.warning("auto-log: could not parse %s", path.name, exc_info=True)

    def _sort_key(e: dict) -> str:
        return e.get("timestamp") or e.get("started_at") or ""

    entries.sort(key=_sort_key)

    # File manifest
    all_files = sorted(f for f in output_dir.iterdir() if f.is_file())
    manifest = {
        "json": sorted(
            f.name
            for f in all_files
            if f.suffix == ".json" and f.name != "session_summary.json"
        ),
        "hdf5": sorted(f.name for f in all_files if f.suffix == ".h5"),
        "figures": sorted(
            f.name for f in all_files if f.suffix.lower() in _FIGURE_SUFFIXES
        ),
        "npy": sorted(f.name for f in all_files if f.suffix == ".npy"),
    }

    summary = {
        "type": "session_summary",
        "id": f"session_summary_{datetime.now(timezone.utc):%Y%m%d_%H%M%S_%f}",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "entry_count": len(entries),
        "entries": entries,
        "files": manifest,
    }

    summary_path = output_dir / "session_summary.json"
    try:
        summary_path.write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8"
        )
    except Exception:
        logger.warning("auto-log: failed to write session summary", exc_info=True)
        return None

    # Interoperable archive: a standard ``.eln`` RO-Crate (importable by
    # mainstream ELNs) instead of a plain ZIP of the folder.
    try:
        from safe_lab_agents.export import build_eln

        build_eln(output_dir, output_dir / f"{output_dir.parent.name or 'session'}.eln")
    except Exception:
        logger.warning("auto-log: failed to write session .eln", exc_info=True)

    logger.info(
        "auto-log: session summary written → %s (%d entries)",
        summary_path,
        len(entries),
    )
    return summary_path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------



def _make_result_entry(result: Any) -> Any:
    """Apply json_safe to a result that has already had arrays extracted."""
    if isinstance(result, dict):
        return {
            k: json_safe(v)
            if not (isinstance(v, dict) and v.get("_type") == "ndarray")
            else v
            for k, v in result.items()
        }
    return {"result": json_safe(result)}


def _serialize_param_value(value: Any) -> Any:
    """Serialize a single parameter value, preserving ndarray reference dicts."""
    if isinstance(value, dict) and value.get("_type") == "ndarray":
        return value
    if isinstance(value, dict):
        return {
            k: v
            if (isinstance(v, dict) and v.get("_type") == "ndarray")
            else json_safe(v)
            for k, v in value.items()
        }
    return json_safe(value)


def _record_call(
    *,
    exp_id: str,
    tool_name: str,
    timestamp: str,
    duration_ms: int,
    call_args: dict,
    result: Any,
    batch: _Batch | None,
    output_dir: Path,
) -> None:
    """Build an experiment entry and either append it to the batch or write to disk."""
    if batch is not None:
        h5_path = batch.h5_path
        group = f"/{exp_id}"
        # Open the shared batch file once and reuse the handle across the whole
        # sweep, so thousands of fast calls don't each pay an open/close.  Only
        # open it when this call actually has arrays to write — a scalar-only
        # sweep never touches HDF5.
        if has_arrays(result) or any(has_arrays(v) for v in call_args.values()):
            with _state_lock:
                if batch.h5_file is None:
                    batch.h5_file = h5py.File(str(h5_path), "a")
                h5_file = batch.h5_file
        else:
            h5_file = None
    else:
        file_stem = f"{exp_id}-{tool_name}"
        h5_path = output_dir / f"{file_stem}.h5"
        group = ""
        h5_file = None

    modified_result = extract_arrays(result, h5_path, group, h5_file=h5_file)

    # Extract arrays from call_args (numpy params saved to HDF5)
    params_base = f"{group}/params" if group else "/params"
    modified_call_args: dict[str, Any] = {}
    for k, v in call_args.items():
        modified_call_args[k] = extract_arrays(
            v, h5_path, f"{params_base}/{k}", h5_file=h5_file
        )

    entry: dict[str, Any] = {
        "id": exp_id,
        "title": tool_name,
        "timestamp": timestamp,
        "duration_ms": duration_ms,
        "parameters": {
            f"param_{k}": _serialize_param_value(v)
            for k, v in modified_call_args.items()
        },
        "result": _make_result_entry(modified_result),
    }
    if h5_path.exists() and batch is None:
        entry["h5_file"] = h5_path.name

    if batch is not None:
        with _state_lock:
            batch.experiments.append(entry)
    else:
        file_stem = f"{exp_id}-{tool_name}"
        json_path = output_dir / f"{file_stem}.json"
        record = {"type": "individual", **entry}
        json_path.write_text(
            json.dumps(record, indent=2, default=str), encoding="utf-8"
        )
        _push_to_kadi(record, output_dir)
