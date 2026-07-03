"""Export an auto-log folder as a ``.eln`` file (RO-Crate / The ELN Consortium).

A ``.eln`` file is a ZIP whose single root folder is an RO-Crate 1.1: a
``ro-crate-metadata.json`` (flattened JSON-LD using schema.org) plus the data
files.  This lets a logged session be imported into mainstream electronic lab
notebooks (eLabFTW, Kadi4Mat, PASTA, SampleDB, …).

Mapping from our per-call records to the crate:

* each ``exp_*`` / ``analysis_*`` / ``batch_*`` record → a ``Dataset`` (a
  per-record folder inside the crate) holding that record's files;
* every parameter / result / data value → a ``PropertyValue`` node referenced
  from the Dataset's ``variableMeasured`` — units travel as ``unitText`` (and
  ``unitCode`` when a QUDT IRI is known/supplied);
* every file → a ``File`` node with ``encodingFormat`` / ``contentSize`` /
  ``sha256``;
* ``references`` (analysis) → ``mentions`` relations between Datasets.

Provenance/authorship is attributed honestly to the *software*
(``safe-lab-agents``); an optional human ``author`` may be supplied.
"""

from __future__ import annotations

import hashlib
import json
import logging
import mimetypes
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from safe_lab_agents.mcp.predefined.records import (
    is_quantity,
    json_safe,
    split_quantity,
)
from safe_lab_agents.report.builder import _load_entries

logger = logging.getLogger(__name__)

_CRATE_CONTEXT = "https://w3id.org/ro/crate/1.1/context"
_CRATE_CONFORMS = "https://w3id.org/ro/crate/1.1"
_SOFTWARE_ID = "#safe-lab-agents"
_SOFTWARE_URL = "https://pypi.org/project/safe-lab-agents/"

# A small, extensible lookup of common units → QUDT IRIs for schema.org
# ``unitCode``.  Unknown units still get a human-readable ``unitText``.
_QUDT_UNITS: dict[str, str] = {
    "V": "http://qudt.org/vocab/unit/V",
    "mV": "http://qudt.org/vocab/unit/MilliV",
    "A": "http://qudt.org/vocab/unit/A",
    "mA": "http://qudt.org/vocab/unit/MilliA",
    "W": "http://qudt.org/vocab/unit/W",
    "mW": "http://qudt.org/vocab/unit/MilliW",
    "s": "http://qudt.org/vocab/unit/SEC",
    "ms": "http://qudt.org/vocab/unit/MilliSEC",
    "m": "http://qudt.org/vocab/unit/M",
    "mm": "http://qudt.org/vocab/unit/MilliM",
    "Hz": "http://qudt.org/vocab/unit/HZ",
    "kHz": "http://qudt.org/vocab/unit/KiloHZ",
    "MHz": "http://qudt.org/vocab/unit/MegaHZ",
    "K": "http://qudt.org/vocab/unit/K",
    "Pa": "http://qudt.org/vocab/unit/PA",
    "ohm": "http://qudt.org/vocab/unit/OHM",
    "Ω": "http://qudt.org/vocab/unit/OHM",
    "°C": "http://qudt.org/vocab/unit/DEG_C",
}

# Exclude previously-generated artifacts from the file inventory.
_EXCLUDE_SUFFIXES = {".eln", ".zip"}
_EXCLUDE_NAMES = {"report.html", "report_safe_lab_agents.html"}


def _package_version() -> str:
    try:
        from importlib.metadata import version

        return version("safe_lab_agents")
    except Exception:
        return "0"


def _encoding_format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in (".h5", ".hdf5"):
        return "application/x-hdf5"
    if suffix in (".npz", ".npy"):
        return "application/octet-stream"
    return mimetypes.guess_type(path.name)[0] or "application/octet-stream"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _file_node(arc_id: str, path: Path) -> dict[str, Any]:
    """Build a schema.org ``File`` node for a data file."""
    stat = path.stat()
    return {
        "@id": arc_id,
        "@type": "File",
        "name": path.name,
        "encodingFormat": _encoding_format(path),
        "contentSize": str(stat.st_size),
        "sha256": _sha256(path),
        "dateModified": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
    }


def _property_value(node_id: str, name: str, value: Any) -> dict[str, Any]:
    """Build a ``PropertyValue`` node, lifting a quantity's unit into the graph."""
    node: dict[str, Any] = {"@id": node_id, "@type": "PropertyValue", "name": name}
    unit: str | None = None
    term: str | None = None
    if isinstance(value, dict) and value.get("_type") == "ndarray":
        shape = "×".join(str(s) for s in value.get("shape", []))
        node["value"] = f"ndarray[{shape}] {value.get('dtype', '')}".strip()
        unit = value.get("unit")
    elif is_quantity(value):
        num, unit, term = split_quantity(value)
        node["value"] = json_safe(num)
    else:
        node["value"] = json_safe(value)
    if unit:
        node["unitText"] = unit
        code = term or _QUDT_UNITS.get(unit)
        if code:
            node["unitCode"] = code
    return node


def _measurements(
    graph: list[dict], dataset_id: str, prefix: str, mapping: dict | None
) -> list[dict[str, str]]:
    """Append PropertyValue nodes for *mapping* to *graph*; return their refs."""
    refs: list[dict[str, str]] = []
    for key, value in (mapping or {}).items():
        name = key[6:] if key.startswith("param_") else key
        node_id = f"#{dataset_id.rstrip('/')}-{prefix}-{name}"
        graph.append(_property_value(node_id, name, value))
        refs.append({"@id": node_id})
    return refs


def _entry_files(log_dir: Path, entry: dict) -> list[Path]:
    """Disk files belonging to *entry*: id-prefixed records + referenced figures."""
    eid = entry.get("id", "")
    files = [
        p for p in log_dir.iterdir() if p.is_file() and eid and p.name.startswith(eid)
    ]
    seen = {p.name for p in files}
    for fig in entry.get("figures", []):
        name = fig.get("file") if isinstance(fig, dict) else fig
        if not name or name in seen:
            continue
        p = log_dir / name
        if p.is_file():
            files.append(p)
            seen.add(name)
    return files


def _entry_dataset(
    graph: list[dict], log_dir: Path, entry: dict, author_ref: dict[str, str]
) -> tuple[str, list[tuple[str, Path]]]:
    """Add one record's Dataset (+ PropertyValue/File nodes) to *graph*.

    Returns ``(dataset_id, [(arcname, disk_path), …])`` for the files to pack.
    """
    eid = entry.get("id", "unknown")
    etype = entry.get("type")
    dataset_id = f"{eid}/"

    node: dict[str, Any] = {
        "@id": dataset_id,
        "@type": "Dataset",
        "name": entry.get("title") or entry.get("label") or eid,
        "identifier": eid,
        "dateCreated": entry.get("timestamp") or entry.get("started_at") or "",
        "author": author_ref,
    }
    if entry.get("description"):
        node["description"] = entry["description"]
    if entry.get("text"):
        node["text"] = entry["text"]
    if entry.get("kind"):
        node["keywords"] = entry["kind"]

    var_refs: list[dict[str, str]] = []
    if etype == "batch":
        node["keywords"] = "batch"
        for scalar in ("experiment_count", "started_at", "completed_at"):
            if entry.get(scalar) is not None:
                var_refs += _measurements(
                    graph, dataset_id, "batch", {scalar: entry[scalar]}
                )
        # Flatten each run's params + results as PropertyValues (full per-run
        # detail also lives in the attached batch JSON file).
        for i, exp in enumerate(entry.get("experiments", []), 1):
            var_refs += _measurements(
                graph, dataset_id, f"run{i}", exp.get("parameters")
            )
            result = exp.get("result")
            if isinstance(result, dict):
                var_refs += _measurements(graph, dataset_id, f"run{i}", result)
    else:
        var_refs += _measurements(graph, dataset_id, "param", entry.get("parameters"))
        result = entry.get("result")
        if isinstance(result, dict):
            var_refs += _measurements(graph, dataset_id, "result", result)
        var_refs += _measurements(graph, dataset_id, "data", entry.get("data"))

    if var_refs:
        node["variableMeasured"] = var_refs

    # Files belonging to this record, packed under its folder.
    packed: list[tuple[str, Path]] = []
    has_part: list[dict[str, str]] = []
    for path in _entry_files(log_dir, entry):
        arc = f"{eid}/{path.name}"
        graph.append(_file_node(arc, path))
        has_part.append({"@id": arc})
        packed.append((arc, path))
    if has_part:
        node["hasPart"] = has_part

    # References → provenance relations between Datasets.
    refs = [r for r in entry.get("references", []) if isinstance(r, str)]
    if refs:
        node["mentions"] = [{"@id": f"{r}/"} for r in refs]

    graph.append(node)
    return dataset_id, packed


def build_eln(
    log_dir: Path,
    out_path: Path,
    *,
    name: str | None = None,
    author: str | None = None,
    affiliation: str | None = None,
) -> Path:
    """Package *log_dir* as a ``.eln`` RO-Crate at *out_path*.

    Args:
        log_dir: An ``auto_log/`` folder (JSON + HDF5 + figures).
        out_path: Destination ``.eln`` file.
        name: Human name for the session (root Dataset); defaults to the folder.
        author: Optional human author name; when given, a ``Person`` is emitted
            and used as ``author`` instead of the software entity.
        affiliation: Optional organisation name for the human author.

    Returns the written ``out_path``.
    """
    log_dir = Path(log_dir)
    out_path = Path(out_path)
    entries = _load_entries(log_dir)

    graph: list[dict[str, Any]] = []

    # Authorship entities — software always; human only when supplied.
    software = {
        "@id": _SOFTWARE_ID,
        "@type": "SoftwareApplication",
        "name": "safe-lab-agents",
        "version": _package_version(),
        "url": _SOFTWARE_URL,
    }
    graph.append(software)
    author_ref: dict[str, str] = {"@id": _SOFTWARE_ID}
    if author:
        person: dict[str, Any] = {"@id": "#author", "@type": "Person", "name": author}
        if affiliation:
            graph.append(
                {"@id": "#affiliation", "@type": "Organization", "name": affiliation}
            )
            person["affiliation"] = {"@id": "#affiliation"}
        graph.append(person)
        author_ref = {"@id": "#author"}

    # Per-record Datasets.
    root_parts: list[dict[str, str]] = []
    packed: list[tuple[str, Path]] = []
    claimed: set[str] = set()
    for entry in entries:
        dataset_id, files = _entry_dataset(graph, log_dir, entry, author_ref)
        root_parts.append({"@id": dataset_id})
        for arc, path in files:
            packed.append((arc, path))
            claimed.add(path.name)

    # Remaining files (summary JSON, orphan figures, npy) → flat File entities.
    for path in sorted(log_dir.iterdir()):
        if not path.is_file() or path.name in claimed:
            continue
        if path.suffix.lower() in _EXCLUDE_SUFFIXES or path.name in _EXCLUDE_NAMES:
            continue
        if path.resolve() == out_path.resolve():
            continue
        graph.append(_file_node(path.name, path))
        root_parts.append({"@id": path.name})
        packed.append((path.name, path))

    # Root data entity + metadata descriptor.
    root = {
        "@id": "./",
        "@type": "Dataset",
        "name": name or log_dir.parent.name or "safe-lab-agents session",
        "datePublished": datetime.now(timezone.utc).isoformat(),
        "description": "Experiment session logged by safe-lab-agents.",
        "hasPart": root_parts,
    }
    descriptor = {
        "@id": "ro-crate-metadata.json",
        "@type": "CreativeWork",
        "conformsTo": {"@id": _CRATE_CONFORMS},
        "about": {"@id": "./"},
        "sdPublisher": {"@id": _SOFTWARE_ID},
    }
    crate = {"@context": _CRATE_CONTEXT, "@graph": [descriptor, root, *graph]}

    # Write the ZIP: single root folder == archive stem.
    root_folder = out_path.stem
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(str(out_path), "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            f"{root_folder}/ro-crate-metadata.json",
            json.dumps(crate, indent=2, default=str),
        )
        for arc, path in packed:
            zf.write(str(path), f"{root_folder}/{arc}")

    logger.info("eln: wrote %s (%d entries)", out_path, len(entries))
    return out_path
