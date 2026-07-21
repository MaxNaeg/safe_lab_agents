"""Auto-generated client for experiment auto-log tools.

Import inside Docker to group tool calls into batches or record analysis results.

Example::

    import sys; sys.path.insert(0, "/agent/workspace")
    from auto_log_client import start_batch, stop_batch, log_analysis, AUTO_LOG_DIR
"""
import base64 as _base64
import io as _io
import json as _json
import os as _os
import pickle as _pickle
import shutil as _shutil
import urllib.request as _urllib_request
from datetime import datetime as _datetime, timezone as _timezone
from pathlib import Path as _Path

try:
    import h5py as _h5py
    _H5PY = True
except ImportError:
    _H5PY = False

try:
    import numpy as _np
    _NUMPY = True
except ImportError:
    _NUMPY = False

AUTO_LOG_DIR = '__CONTAINER_LOG_DIR__'

_HOST = _os.environ.get("MCP_HOST", "host.docker.internal")
_PORT = _os.environ["MCP_PORT"]
_URL = f"http://{_HOST}:{_PORT}/invoke"
_HEADERS = {"Content-Type": "application/json"}
_TOKEN = _os.environ.get("MCP_AUTH_TOKEN", "")
if _TOKEN:
    _HEADERS["Authorization"] = f"Bearer {_TOKEN}"


def _encode_arg(obj):
    if _NUMPY and isinstance(obj, _np.ndarray):
        buf = _io.BytesIO()
        _np.save(buf, obj)
        return {"__type__": "ndarray", "data": _base64.b64encode(buf.getvalue()).decode()}
    if _NUMPY and isinstance(obj, _np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _encode_arg(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_encode_arg(x) for x in obj]
    return obj


def _invoke(tool_name: str, **kwargs) -> str:
    body = _json.dumps({"tool": tool_name, "args": {k: _encode_arg(v) for k, v in kwargs.items()}}).encode()
    req = _urllib_request.Request(_URL, data=body, headers=_HEADERS)
    with _urllib_request.urlopen(req) as r:
        return _pickle.loads(r.read())


def start_batch(label: str, description: str = "") -> str:
    """Start collecting experiment results into a single ELN batch record.

    All tool calls until stop_batch() are grouped into one merged record
    instead of creating individual records per call.

    Use this when running parameter sweeps, optimisation loops, multi-step
    protocols, or repeated measurements — any set of calls that logically
    form one experiment.

    Args:
        label: Short label, e.g. "Voltage sweep 0-5 V".
        description: Optional longer description of the batch.
    """
    return _invoke("start_batch", label=label, description=description)


def stop_batch() -> str:
    """Finalise the active batch and write a merged ELN record to disk.

    Returns a summary string with the output file path and experiment count.
    """
    return _invoke("stop_batch")


def log_analysis(
    title: str,
    text: str = "",
    data: dict = None,
    references: list = None,
    script: str = "",
    figures: list = None,
    kind: str = "analysis",
) -> str:
    """Record analysis results as an ELN entry and push to Kadi4Mat if configured.

    Args:
        title: Short title, e.g. "Linear fit of voltage sweep".
        text: Free-text narrative, observations, or conclusions (markdown OK).
        data: Dict of analysis results. numpy arrays are saved to HDF5
              automatically. Scalars and strings are stored as JSON metadata.
              **Note:** values must be JSON-serializable or numpy arrays —
              other types (DataFrames, arbitrary objects) are not supported.
        references: List of exp_*/batch_*/analysis_* IDs this analysis is
                    based on.
        script: Python source code used to produce this analysis.
        figures: Filenames of figures already saved to AUTO_LOG_DIR.
                 **Note:** figures must be saved to AUTO_LOG_DIR before calling
                 log_analysis — files outside AUTO_LOG_DIR are not accessible.
                 Pass only the filename, not the full path, e.g. ``["fit.png"]``.
        kind: What sort of record this is. One of "analysis" (default, a
              successful result), "hypothesis", "decision", "debug" (a
              debugging step or failed-then-fixed iteration), "failed" (an
              attempt that did not succeed), or "observation". Record failures
              and debug steps too — a failed attempt is data, not noise.

    Returns:
        Confirmation string with the output file name.

    Example::

        from auto_log_client import log_analysis, AUTO_LOG_DIR
        import numpy as np, matplotlib.pyplot as plt

        slope, intercept = np.polyfit(voltages, powers, 1)
        residuals = powers - (slope * voltages + intercept)

        fig, ax = plt.subplots()
        ax.scatter(voltages, powers)
        ax.plot(voltages, slope * voltages + intercept, "r-")
        fig.savefig(f"{AUTO_LOG_DIR}/fit.png")
        plt.close(fig)

        log_analysis(
            title="Linear fit",
            text="Power is linear with voltage (R²=0.998).",
            data={"slope": slope, "residuals": residuals},
            references=["exp_20260522_111149_616781"],
            script=open(__file__).read(),
            figures=["fit.png"],
        )
    """
    return _invoke("log_analysis",
        title=title, text=text, data=data or {},
        references=references or [], script=script,
        figures=figures or [], kind=kind,
    )
