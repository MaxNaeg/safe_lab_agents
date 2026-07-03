"""Static HTML report generation for auto-log folders.

``build_report`` reads an ``auto_log/`` directory (the JSON / HDF5 / figure
files written by :mod:`safe_lab_agents.mcp.predefined.autolog`) and emits a
single self-contained ``report.html`` with client-side filtering, search, and
collapsible scripts — no server required.
"""

from safe_lab_agents.report.builder import build_report

__all__ = ["build_report"]
