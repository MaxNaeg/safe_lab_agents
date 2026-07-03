"""Resume-command behaviour guards.

Resume is intentionally interactive-only: a previously autonomous (``--task``)
session is continued interactively, never re-run. The ``--task`` option was
removed so it cannot be requested.
"""

from __future__ import annotations

import inspect

from safe_lab_agents.cli import resume


def test_resume_has_no_task_option() -> None:
    """`resume` must not expose a --task option (resume is interactive-only)."""
    assert "task" not in inspect.signature(resume).parameters
