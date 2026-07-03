"""Tests for autolog.py."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _setup_autolog(tmp_path: Path):
    """Set AUTO_LOG_DIR and initialise module-level state via make_autolog_wrapper."""
    os.environ["AUTO_LOG_DIR"] = str(tmp_path)
    from safe_lab_agents.mcp.predefined.autolog import make_autolog_wrapper

    return make_autolog_wrapper()


def _reset_autolog():
    os.environ.pop("AUTO_LOG_DIR", None)
    import safe_lab_agents.mcp.predefined.autolog as _m

    _m._current_batch = None
    _m._output_dir = None


@pytest.fixture()
def wrapper(tmp_path: Path):
    w = _setup_autolog(tmp_path)
    yield w
    _reset_autolog()


@pytest.fixture()
def tools(tmp_path: Path):
    _setup_autolog(tmp_path)
    from safe_lab_agents.mcp.predefined.autolog import start_batch, stop_batch

    yield {"start_batch": start_batch, "stop_batch": stop_batch}
    _reset_autolog()


@pytest.fixture()
def wrapper_and_tools(tmp_path: Path):
    w = _setup_autolog(tmp_path)
    from safe_lab_agents.mcp.predefined.autolog import start_batch, stop_batch

    yield w, {"start_batch": start_batch, "stop_batch": stop_batch}
    _reset_autolog()


# ---------------------------------------------------------------------------
# Module-level functions exported
# ---------------------------------------------------------------------------


def test_batch_functions_importable():
    from safe_lab_agents.mcp.predefined.autolog import (
        start_batch,
        stop_batch,
        make_autolog_wrapper,
    )

    assert callable(start_batch)
    assert callable(stop_batch)
    assert callable(make_autolog_wrapper)


def test_auto_log_dir_is_agent_writable(tmp_path: Path):
    """The auto-log dir is created 0777 so the in-container (non-root) agent,
    which saves figures/data into it via auto_log_client, can write there too.

    It is created host-side by the MCP server; without widening, the container
    'agent' user (a mismatched UID) falls in the 'other' class and is blocked.
    """
    target = tmp_path / "auto_log"
    os.environ["AUTO_LOG_DIR"] = str(target)
    try:
        from safe_lab_agents.mcp.predefined.autolog import make_autolog_wrapper

        make_autolog_wrapper()
        assert (target.stat().st_mode & 0o777) == 0o777
    finally:
        _reset_autolog()


# ---------------------------------------------------------------------------
# start_batch / stop_batch: happy path
# ---------------------------------------------------------------------------


def test_start_stop_batch_writes_json(wrapper_and_tools, tmp_path: Path):
    wrapper, tools = wrapper_and_tools
    tools["start_batch"]("Voltage sweep", description="0 to 5 V")

    def measure(v: float) -> dict:
        return {"voltage": v, "current": v * 0.1}

    wrapped = wrapper(measure)
    wrapped(1.0)
    wrapped(2.0)

    result = tools["stop_batch"]()
    assert "Voltage sweep" in result

    json_files = list(tmp_path.glob("batch_*.json"))
    assert len(json_files) == 1

    record = json.loads(json_files[0].read_text())
    assert record["type"] == "batch"
    assert record["label"] == "Voltage sweep"
    assert record["description"] == "0 to 5 V"
    assert record["experiment_count"] == 2
    assert len(record["experiments"]) == 2
    assert "started_at" in record
    assert "completed_at" in record


def test_stop_batch_without_start_returns_error(tools):
    msg = tools["stop_batch"]()
    assert "No batch is active" in msg


def test_start_batch_when_already_active_returns_error(tools):
    tools["start_batch"]("First")
    msg = tools["start_batch"]("Second")
    assert "already active" in msg
    tools["stop_batch"]()


# ---------------------------------------------------------------------------
# Wrapper: individual records (no batch)
# ---------------------------------------------------------------------------


def test_wrapper_writes_individual_json(wrapper, tmp_path: Path):
    def measure(sample: str) -> dict:
        return {"temperature": 25.0, "sample": sample}

    wrapped = wrapper(measure)
    result = wrapped("A1")

    assert result == {"temperature": 25.0, "sample": "A1"}

    json_files = list(tmp_path.glob("exp_*-measure.json"))
    assert len(json_files) == 1

    record = json.loads(json_files[0].read_text())
    assert record["type"] == "individual"
    assert record["title"] == "measure"
    assert "timestamp" in record
    assert "duration_ms" in record
    assert record["result"]["temperature"] == 25.0
    assert record["parameters"]["param_sample"] == "A1"


def test_wrapper_appends_to_batch_no_individual_file(wrapper_and_tools, tmp_path: Path):
    wrapper, tools = wrapper_and_tools
    tools["start_batch"]("Sweep")

    def measure(x: int) -> dict:
        return {"value": x}

    wrapped = wrapper(measure)
    wrapped(1)
    wrapped(2)

    assert list(tmp_path.glob("exp_*.json")) == []

    tools["stop_batch"]()
    assert len(list(tmp_path.glob("batch_*.json"))) == 1


def test_experiment_entries_have_timestamps(wrapper_and_tools, tmp_path: Path):
    wrapper, tools = wrapper_and_tools
    tools["start_batch"]("Test")

    def probe() -> dict:
        return {"x": 1}

    wrapper(probe)()
    tools["stop_batch"]()

    record = json.loads(next(tmp_path.glob("batch_*.json")).read_text())
    exp = record["experiments"][0]
    assert "timestamp" in exp
    assert "duration_ms" in exp
    assert isinstance(exp["duration_ms"], int)


# ---------------------------------------------------------------------------
# Numpy array handling
# ---------------------------------------------------------------------------


def test_numpy_array_in_result_saved_to_hdf5(wrapper, tmp_path: Path):
    arr = np.linspace(0, 1, 10)

    def measure() -> dict:
        return {"spectrum": arr, "peak": 0.5}

    original_result = wrapper(measure)()

    assert isinstance(original_result["spectrum"], np.ndarray)
    np.testing.assert_array_equal(original_result["spectrum"], arr)

    h5_files = list(tmp_path.glob("exp_*-measure.h5"))
    assert len(h5_files) == 1
    with h5py.File(h5_files[0], "r") as f:
        np.testing.assert_array_equal(f["/spectrum"][:], arr)

    record = json.loads(next(tmp_path.glob("exp_*-measure.json")).read_text())
    ref = record["result"]["spectrum"]
    assert ref["_type"] == "ndarray"
    assert ref["file"] == h5_files[0].name
    assert ref["dataset"] == "/spectrum"
    assert ref["shape"] == [10]
    assert ref["dtype"] == "float64"


def test_scalar_quantity_recorded_with_unit(wrapper, tmp_path: Path):
    from safe_lab_agents import quantity

    def measure() -> dict:
        return {"power": quantity(2.5, "W"), "n": 3}

    wrapper(measure)()
    record = json.loads(next(tmp_path.glob("exp_*-measure.json")).read_text())
    assert record["result"]["power"] == {"value": 2.5, "unit": "W"}
    assert record["result"]["n"] == 3


def test_array_quantity_writes_hdf5_units_attr(wrapper, tmp_path: Path):
    from safe_lab_agents import quantity

    arr = np.linspace(0, 1, 8)

    def measure() -> dict:
        return {"trace": quantity(arr, "V")}

    wrapper(measure)()

    h5_files = list(tmp_path.glob("exp_*-measure.h5"))
    assert len(h5_files) == 1
    with h5py.File(h5_files[0], "r") as f:
        assert f["/trace"].attrs["units"] == "V"

    record = json.loads(next(tmp_path.glob("exp_*-measure.json")).read_text())
    ref = record["result"]["trace"]
    assert ref["_type"] == "ndarray"
    assert ref["unit"] == "V"


def test_bare_ndarray_result_saved(wrapper, tmp_path: Path):
    arr = np.array([1, 2, 3])

    def get_data() -> Any:
        return arr

    original_result = wrapper(get_data)()
    np.testing.assert_array_equal(original_result, arr)

    h5_files = list(tmp_path.glob("exp_*-get_data.h5"))
    assert len(h5_files) == 1
    with h5py.File(h5_files[0], "r") as f:
        np.testing.assert_array_equal(f["/data"][:], arr)


def test_batch_arrays_in_single_hdf5_file(wrapper_and_tools, tmp_path: Path):
    wrapper, tools = wrapper_and_tools
    tools["start_batch"]("Array sweep")

    def measure(i: int) -> dict:
        return {"data": np.ones(5) * i}

    wrapped = wrapper(measure)
    wrapped(1)
    wrapped(2)
    tools["stop_batch"]()

    h5_files = list(tmp_path.glob("batch_*.h5"))
    assert len(h5_files) == 1
    with h5py.File(h5_files[0], "r") as f:
        groups = list(f.keys())
        assert len(groups) == 2
        for g in groups:
            assert "data" in f[g]


def test_no_h5_file_when_no_arrays(wrapper, tmp_path: Path):
    def measure() -> dict:
        return {"value": 42.0}

    wrapper(measure)()
    assert list(tmp_path.glob("*.h5")) == []


# ---------------------------------------------------------------------------
# Serialisation of arbitrary Python objects
# ---------------------------------------------------------------------------


def test_arbitrary_object_stringified(wrapper, tmp_path: Path):
    class MyObj:
        def __str__(self):
            return "MyObj()"

    def tool(obj: Any) -> dict:
        return {"result": obj}

    wrapper(tool)(MyObj())
    record = json.loads(next(tmp_path.glob("exp_*.json")).read_text())
    assert record["result"]["result"] == "MyObj()"


def test_non_serialisable_param_stringified(wrapper, tmp_path: Path):
    class Config:
        def __str__(self):
            return "Config(x=1)"

    def tool(cfg: Any) -> str:
        return "ok"

    wrapper(tool)(Config())
    record = json.loads(next(tmp_path.glob("exp_*.json")).read_text())
    assert record["parameters"]["param_cfg"] == "Config(x=1)"


def test_numpy_array_param_saved_to_hdf5(wrapper, tmp_path: Path):
    arr = np.linspace(0, 1, 5)

    def measure(spectrum: Any) -> dict:
        return {"peak": 0.5}

    wrapper(measure)(arr)

    h5_files = list(tmp_path.glob("exp_*-measure.h5"))
    assert len(h5_files) == 1
    with h5py.File(h5_files[0], "r") as f:
        np.testing.assert_array_equal(f["/params/spectrum/data"][:], arr)

    record = json.loads(next(tmp_path.glob("exp_*-measure.json")).read_text())
    ref = record["parameters"]["param_spectrum"]
    assert ref["_type"] == "ndarray"
    assert ref["shape"] == [5]
    assert ref["dtype"] == "float64"


# ---------------------------------------------------------------------------
# Exception propagation and no_autolog
# ---------------------------------------------------------------------------


def test_wrapper_propagates_exceptions(wrapper):
    def explode() -> None:
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        wrapper(explode)()


def test_no_autolog_decorator_skips_logging(wrapper, tmp_path: Path):
    from safe_lab_agents.mcp.predefined.autolog import no_autolog

    @no_autolog
    def silent_tool() -> dict:
        return {"x": 1}

    wrapped = wrapper(silent_tool)
    assert wrapped is silent_tool
    wrapped()
    assert list(tmp_path.glob("*.json")) == []


# ---------------------------------------------------------------------------
# write_session_summary
# ---------------------------------------------------------------------------


def test_session_summary_empty_dir(tmp_path: Path):
    from safe_lab_agents.mcp.predefined.autolog import write_session_summary

    result = write_session_summary(tmp_path)
    assert result is None
    assert not (tmp_path / "session_summary.json").exists()


def test_session_summary_collects_all_entry_types(wrapper, tmp_path: Path):
    from safe_lab_agents.mcp.predefined.autolog import (
        write_session_summary,
        start_batch,
        stop_batch,
    )

    # Individual entry
    def measure() -> dict:
        return {"value": 1.0}

    wrapper(measure)()

    # Batch entry
    start_batch("Test batch")
    wrapper(measure)()
    stop_batch()

    # Fake analysis entry
    analysis = {
        "type": "analysis",
        "id": "analysis_20260522_120000_000000",
        "title": "My analysis",
        "timestamp": "2026-05-22T12:00:00+00:00",
        "text": "Some text",
        "data": {},
        "references": [],
        "script": "",
        "figures": [],
    }
    (tmp_path / "analysis_20260522_120000_000000.json").write_text(json.dumps(analysis))

    summary_path = write_session_summary(tmp_path)
    assert summary_path is not None
    assert summary_path.exists()

    summary = json.loads(summary_path.read_text())
    assert summary["type"] == "session_summary"
    assert summary["entry_count"] == 3
    types = {e["type"] for e in summary["entries"]}
    assert types == {"individual", "batch", "analysis"}


def test_session_summary_sorted_by_timestamp(wrapper, tmp_path: Path):
    from safe_lab_agents.mcp.predefined.autolog import write_session_summary

    def tool() -> dict:
        return {"x": 1}

    wrapper(tool)()
    wrapper(tool)()

    summary_path = write_session_summary(tmp_path)
    summary = json.loads(summary_path.read_text())
    timestamps = [e["timestamp"] for e in summary["entries"]]
    assert timestamps == sorted(timestamps)


def test_session_summary_creates_eln(wrapper, tmp_path: Path):
    from safe_lab_agents.mcp.predefined.autolog import write_session_summary
    import zipfile

    def tool() -> dict:
        return {"x": 1}

    wrapper(tool)()

    write_session_summary(tmp_path)
    # The plain ZIP is replaced by a standard .eln RO-Crate archive.
    eln_files = list(tmp_path.glob("*.eln"))
    assert len(eln_files) == 1
    assert (tmp_path / "session_summary.json").exists()

    with zipfile.ZipFile(str(eln_files[0]), "r") as zf:
        names = zf.namelist()
    assert any(n.endswith("ro-crate-metadata.json") for n in names)


def test_session_summary_includes_file_manifest(wrapper, tmp_path: Path):
    from safe_lab_agents.mcp.predefined.autolog import write_session_summary
    import numpy as np

    arr = np.linspace(0, 1, 5)

    def measure() -> dict:
        return {"data": arr}

    wrapper(measure)()

    # Add a fake figure
    (tmp_path / "my_plot.png").write_bytes(b"PNG")

    summary_path = write_session_summary(tmp_path)
    summary = json.loads(summary_path.read_text())
    assert len(summary["files"]["hdf5"]) >= 1
    assert "my_plot.png" in summary["files"]["figures"]


# ---------------------------------------------------------------------------
# Generated auto_log_client targets the right host
# ---------------------------------------------------------------------------


def _exec_auto_log_client(
    tmp_path: Path, monkeypatch, *, port: str, host: str | None
) -> dict:
    """Generate auto_log_client.py, exec it with the given env, and return its namespace."""
    from safe_lab_agents.cli import _write_auto_log_client

    _write_auto_log_client(tmp_path, "/agent/auto_log")
    source = (tmp_path / "auto_log_client.py").read_text(encoding="utf-8")

    monkeypatch.setenv("MCP_PORT", port)
    if host is None:
        monkeypatch.delenv("MCP_HOST", raising=False)
    else:
        monkeypatch.setenv("MCP_HOST", host)
    namespace: dict = {}
    exec(compile(source, "<auto-log-client>", "exec"), namespace)
    return namespace


def test_auto_log_client_url_defaults_to_host_docker_internal(
    tmp_path: Path, monkeypatch
):
    """Without MCP_HOST, the auto-log client targets host.docker.internal (Docker default)."""
    ns = _exec_auto_log_client(tmp_path, monkeypatch, port="5000", host=None)
    assert ns["_URL"] == "http://host.docker.internal:5000/invoke"


def test_auto_log_client_url_honours_mcp_host_override(tmp_path: Path, monkeypatch):
    """When MCP_HOST is set (Podman/Windows), the auto-log client targets that address."""
    ns = _exec_auto_log_client(tmp_path, monkeypatch, port="5000", host="172.26.80.1")
    assert ns["_URL"] == "http://172.26.80.1:5000/invoke"


# ---------------------------------------------------------------------------
# log_analysis kind field
# ---------------------------------------------------------------------------


def test_log_analysis_defaults_kind_to_analysis(tmp_path: Path):
    """Omitting kind yields kind='analysis' so pre-kind records read consistently."""
    _setup_autolog(tmp_path)
    from safe_lab_agents.mcp.predefined.autolog import log_analysis

    try:
        log_analysis(title="Some result", text="It worked.")
        record = json.loads(next(tmp_path.glob("analysis_*.json")).read_text())
        assert record["kind"] == "analysis"
    finally:
        _reset_autolog()


def test_log_analysis_records_failed_kind(tmp_path: Path):
    """A failed attempt can be logged and is tagged kind='failed'."""
    _setup_autolog(tmp_path)
    from safe_lab_agents.mcp.predefined.autolog import log_analysis

    try:
        log_analysis(
            title="Gaussian fit did not converge",
            text="curve_fit raised RuntimeError.",
            script="raise RuntimeError()",
            kind="failed",
        )
        record = json.loads(next(tmp_path.glob("analysis_*.json")).read_text())
        assert record["kind"] == "failed"
        assert record["script"] == "raise RuntimeError()"
    finally:
        _reset_autolog()
