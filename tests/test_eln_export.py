"""Tests for the .eln (RO-Crate) exporter."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from safe_lab_agents.export import build_eln


def _write(folder: Path, name: str, record: dict) -> None:
    (folder / name).write_text(json.dumps(record), encoding="utf-8")


def _crate(eln_path: Path) -> dict:
    with zipfile.ZipFile(str(eln_path), "r") as zf:
        names = zf.namelist()
        meta = next(n for n in names if n.endswith("ro-crate-metadata.json"))
        return json.loads(zf.read(meta))


def _graph_by_type(crate: dict, typ: str) -> list[dict]:
    return [n for n in crate["@graph"] if n.get("@type") == typ]


def test_build_eln_produces_valid_crate(tmp_path: Path):
    log_dir = tmp_path / "auto_log"
    log_dir.mkdir()
    _write(
        log_dir,
        "exp_20260101_000000_000001-measure.json",
        {
            "type": "individual",
            "id": "exp_20260101_000000_000001",
            "title": "measure",
            "timestamp": "2026-01-01T00:00:00+00:00",
            "parameters": {"param_channel": 1},
            "result": {"power": {"value": 2.5, "unit": "W"}, "status": "ok"},
        },
    )

    out = tmp_path / "session.eln"
    build_eln(log_dir, out)
    assert out.exists()

    crate = _crate(out)
    # @context is a list: the RO-Crate 1.1 context plus a local term declaring
    # ``sha256`` (not defined upstream) so the key resolves in compacted JSON-LD.
    assert crate["@context"][0] == "https://w3id.org/ro/crate/1.1/context"
    assert "sha256" in crate["@context"][1]

    # RO-Crate 1.1 REQUIRES a license on the root data entity.
    root = next(n for n in crate["@graph"] if n["@id"] == "./")
    assert root["license"]

    # Descriptor + root Dataset MUSTs.
    descriptor = next(
        n for n in crate["@graph"] if n["@id"] == "ro-crate-metadata.json"
    )
    assert descriptor["@type"] == "CreativeWork"
    assert descriptor["conformsTo"]["@id"] == "https://w3id.org/ro/crate/1.1"
    assert root["@type"] == "Dataset"
    assert root["hasPart"]

    # Software publisher (honest authorship).
    software = _graph_by_type(crate, "SoftwareApplication")
    assert software and software[0]["name"] == "safe-lab-agents"
    assert descriptor["sdPublisher"]["@id"] == software[0]["@id"]

    # The measurement is a PropertyValue carrying the unit.
    pvs = _graph_by_type(crate, "PropertyValue")
    power = next(p for p in pvs if p["name"] == "power")
    assert power["value"] == 2.5
    assert power["unitText"] == "W"
    assert power["unitCode"] == "http://qudt.org/vocab/unit/W"


def test_build_eln_includes_files_with_checksums(tmp_path: Path):
    import h5py
    import numpy as np

    log_dir = tmp_path / "auto_log"
    log_dir.mkdir()
    with h5py.File(log_dir / "exp_a-measure.h5", "w") as f:
        f.create_dataset("trace", data=np.arange(4))
    _write(
        log_dir,
        "exp_a-measure.json",
        {
            "type": "individual",
            "id": "exp_a",
            "title": "measure",
            "timestamp": "2026-01-01T00:00:00+00:00",
            "parameters": {},
            "result": {
                "trace": {
                    "_type": "ndarray",
                    "file": "exp_a-measure.h5",
                    "dataset": "/trace",
                    "shape": [4],
                    "dtype": "int64",
                    "unit": "V",
                }
            },
            "h5_file": "exp_a-measure.h5",
        },
    )

    out = tmp_path / "session.eln"
    build_eln(log_dir, out)
    crate = _crate(out)

    files = _graph_by_type(crate, "File")
    h5 = next(f for f in files if f["name"] == "exp_a-measure.h5")
    assert h5["encodingFormat"] == "application/x-hdf5"
    assert "sha256" in h5 and len(h5["sha256"]) == 64
    assert int(h5["contentSize"]) > 0

    # The array PropertyValue keeps the unit as unitText.
    pvs = _graph_by_type(crate, "PropertyValue")
    trace = next(p for p in pvs if p["name"] == "trace")
    assert trace["unitText"] == "V"

    # Files are packed under a single root folder named after the archive.
    with zipfile.ZipFile(str(out), "r") as zf:
        assert all(n.startswith("session/") for n in zf.namelist())


def test_entry_files_requires_separator_after_id(tmp_path: Path):
    """A file for entry 'exp_1' must not grab 'exp_10-…' (prefix without separator)."""
    from safe_lab_agents.export.eln import _entry_files

    (tmp_path / "exp_1-measure.json").write_text("{}", encoding="utf-8")
    (tmp_path / "exp_1.h5").write_text("x", encoding="utf-8")
    (tmp_path / "exp_10-measure.json").write_text("{}", encoding="utf-8")

    names = {p.name for p in _entry_files(tmp_path, {"id": "exp_1"})}
    assert "exp_1-measure.json" in names
    assert "exp_1.h5" in names
    assert "exp_10-measure.json" not in names


def test_build_eln_batch_run_param_and_result_same_name_have_distinct_ids(tmp_path: Path):
    """A batch run with a param and a result of the same name must produce two
    PropertyValues with distinct @ids (duplicate @ids are invalid JSON-LD)."""
    log_dir = tmp_path / "auto_log"
    log_dir.mkdir()
    _write(
        log_dir,
        "batch_a.json",
        {
            "type": "batch",
            "id": "batch_a",
            "label": "sweep",
            "started_at": "2026-01-01T00:00:00+00:00",
            "experiment_count": 1,
            "experiments": [
                {
                    "id": "exp_1",
                    "title": "measure",
                    "parameters": {"param_x": 1.0},
                    "result": {"x": 2.0},
                }
            ],
        },
    )

    out = tmp_path / "session.eln"
    build_eln(log_dir, out)
    crate = _crate(out)

    ids = [n["@id"] for n in crate["@graph"] if "@id" in n]
    assert len(ids) == len(set(ids)), "duplicate @id in RO-Crate graph"

    # Both the param and the result surface as separate PropertyValues.
    pv_ids = {n["@id"] for n in _graph_by_type(crate, "PropertyValue")}
    assert any("run1-param-x" in i for i in pv_ids)
    assert any("run1-result-x" in i for i in pv_ids)


def test_build_eln_flattens_nested_array_into_own_measurement(tmp_path: Path):
    """An array nested inside a dict value becomes its own PropertyValue with a
    dotted name and the ndarray summary — not an object-valued PropertyValue."""
    log_dir = tmp_path / "auto_log"
    log_dir.mkdir()
    _write(
        log_dir,
        "exp_a-measure.json",
        {
            "type": "individual",
            "id": "exp_a",
            "title": "measure",
            "timestamp": "2026-01-01T00:00:00+00:00",
            "parameters": {},
            "result": {
                "scan": {
                    "x": {
                        "_type": "ndarray",
                        "file": "exp_a-measure.h5",
                        "dataset": "/scan/x",
                        "shape": [5],
                        "dtype": "float64",
                        "unit": "V",
                    },
                    "n": 5,
                }
            },
        },
    )

    out = tmp_path / "session.eln"
    build_eln(log_dir, out)
    crate = _crate(out)
    pvs = {p["name"]: p for p in _graph_by_type(crate, "PropertyValue")}

    assert "scan.x" in pvs and "scan.n" in pvs
    assert pvs["scan.x"]["value"].startswith("ndarray[5]")
    assert pvs["scan.x"]["unitText"] == "V"
    assert pvs["scan.n"]["value"] == 5
    # No PropertyValue carries a raw object value (the pre-fix failure mode).
    assert all(not isinstance(p.get("value"), (dict, list)) for p in pvs.values())


def test_build_eln_with_human_author(tmp_path: Path):
    log_dir = tmp_path / "auto_log"
    log_dir.mkdir()
    _write(
        log_dir,
        "exp_a-m.json",
        {
            "type": "individual",
            "id": "exp_a",
            "title": "m",
            "timestamp": "2026-01-01T00:00:00+00:00",
            "parameters": {},
            "result": {},
        },
    )

    out = tmp_path / "session.eln"
    build_eln(log_dir, out, author="Ada Lovelace", affiliation="Analytical Engine Lab")
    crate = _crate(out)

    person = next(n for n in crate["@graph"] if n.get("@type") == "Person")
    assert person["name"] == "Ada Lovelace"
    org = next(n for n in crate["@graph"] if n.get("@type") == "Organization")
    assert org["name"] == "Analytical Engine Lab"
