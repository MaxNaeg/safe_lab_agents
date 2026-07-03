"""Interoperable export of auto-log folders.

``build_eln`` packages an ``auto_log/`` directory as a standard ``.eln`` file —
a ZIP wrapping an RO-Crate (``ro-crate-metadata.json``, JSON-LD / schema.org) as
defined by The ELN Consortium — so a session can be imported into mainstream
electronic lab notebooks (eLabFTW, Kadi4Mat, PASTA, …).
"""

from safe_lab_agents.export.eln import build_eln

__all__ = ["build_eln"]
