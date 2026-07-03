"""Auto-generated Python client for experiment tools.

Import this to call experiment tools from Python scripts inside Docker.
Inputs are sent as JSON (numpy arrays encoded as .npy byte streams).
Outputs are received as pickle from the trusted host.
"""
from __future__ import annotations
import base64
import io
import json
import os
import pickle
import urllib.error
import urllib.request

try:
    import numpy as np
    _NUMPY = True
except ImportError:
    _NUMPY = False

_HOST = os.environ.get("MCP_HOST", "host.docker.internal")
_PORT = os.environ["MCP_PORT"]
_URL  = f"http://{_HOST}:{_PORT}/invoke"
_HEADERS = {"Content-Type": "application/json"}
_TOKEN = os.environ.get("MCP_AUTH_TOKEN", "")
if _TOKEN:
    _HEADERS["Authorization"] = f"Bearer {_TOKEN}"


def _encode_arg(obj):
    """Encode one argument for JSON transport. Numpy arrays use the .npy byte stream."""
    if _NUMPY and isinstance(obj, np.ndarray):
        buf = io.BytesIO()
        np.save(buf, obj)
        return {"__type__": "ndarray", "data": base64.b64encode(buf.getvalue()).decode()}
    if isinstance(obj, (list, tuple)):
        return [_encode_arg(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _encode_arg(v) for k, v in obj.items()}
    return obj  # scalars, str, bool, None — natively JSON-safe


def _invoke(tool_name: str, **kwargs):
    body = json.dumps(
        {"tool": tool_name, "args": {k: _encode_arg(v) for k, v in kwargs.items()}}
    ).encode()
    req = urllib.request.Request(_URL, data=body, headers=_HEADERS)
    try:
        with urllib.request.urlopen(req) as r:
            return pickle.loads(r.read())
    except urllib.error.HTTPError as e:
        if e.code == 422:
            msg = json.loads(e.read().decode()).get("error", "type validation failed")
            raise TypeError(f"Tool '{tool_name}': {msg}") from None
        if e.code == 500:
            try:
                payload = json.loads(e.read().decode())
            except Exception:
                raise
            detail = payload.get("traceback") or payload.get("error", "tool raised an exception")
            raise RuntimeError(
                f"Tool '{tool_name}' raised an error on the host:\n{detail}"
            ) from None
        raise


def set_angle(angle: float, component: str) -> str:
    """Set the angle of the optical component.
        Angles are not absolute but only up to a fixed, unknown offset.

        Args:
            angle: Desired angle in degrees (0 to 360).
            component: The optical component whose angle to set ('polarizer', 'lambda_quarter', or 'lambda_half').
        Returns:
            A string indicating success.
        Raises:
            ValueError: If the angle is out of the valid range or if the component name is invalid.
    """
    return _invoke('set_angle', angle=angle, component=component)

def measure_power() -> dict:
    """Measure the optical power at the detector.

        Returns:
            A dictionary containing the measured optical power in Watts.
    """
    return _invoke('measure_power')

