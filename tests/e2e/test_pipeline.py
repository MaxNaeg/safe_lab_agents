"""Full-pipeline matrix test.

Each cell runs a complete lifecycle against real containers and a real agent:

    start (autonomous | interactive)  →  commit  →  resume

and asserts on the durable artifacts (metadata status, history.json tool calls,
committed image). Cells are parametrized and self-skipping (see ``conftest.py``);
a plain ``pytest`` run collects none of this without ``SLA_E2E=1``.
"""

from __future__ import annotations

import os

import pytest

from . import _driver
from .conftest import agent_args_for

# Autonomous runs build the image (first time) + run a model turn; interactive
# adds TUI boot. Generous ceilings — the drivers fail fast on clean container exit.
AUTONOMOUS_TIMEOUT = float(os.environ.get("SLA_E2E_AUTONOMOUS_TIMEOUT", "900"))
BOOT_TIMEOUT = float(os.environ.get("SLA_E2E_BOOT_TIMEOUT", "300"))

RESUME_CONVERSE = os.environ.get("SLA_E2E_RESUME_CONVERSE") == "1"
STRICT_TUI = os.environ.get("SLA_E2E_STRICT_TUI", "1") == "1"
KEEP_ON_FAIL = os.environ.get("SLA_E2E_KEEP_ON_FAIL") == "1"

TASK = "Call the `ping` MCP tool exactly once, then stop."


def test_full_pipeline(runtime, agent, mode):
    """start (mode) → commit → resume, asserting each stage on real artifacts."""
    name = f"e2e-{agent}-{runtime}-{mode}-{os.getpid()}"
    args = agent_args_for(agent)
    assert args is not None  # guaranteed by skip_reason gating

    _driver.cleanup_session(name, runtime)
    passed = False
    try:
        # ---- Stage 1: start ----------------------------------------------
        if mode == "autonomous":
            proc = _driver.run_autonomous(
                agent, runtime, name, TASK, args, timeout=AUTONOMOUS_TIMEOUT
            )
            assert proc.returncode == 0, (
                f"autonomous start failed (rc={proc.returncode}).\n"
                f"--- stdout ---\n{proc.stdout[-3000:]}\n"
                f"--- stderr ---\n{proc.stderr[-3000:]}"
            )
        else:
            _driver.drive_interactive_start(
                agent, runtime, name, args, boot_timeout=BOOT_TIMEOUT
            )

        # ---- Stage 2: assert the session committed with the tool call ----
        _driver.assert_status(name, "committed")
        _driver.assert_committed_image(runtime, name)
        _assert_ping(name, minimum=1, stage=f"{mode} start")

        # ---- Stage 2b: end-to-end output chain (auto-log → reports/export) --
        # The run was --auto-log, so the ping call is recorded under
        # <shared>/auto_log/. Assert the records, then build the two HTML reports
        # (auto-log report + conversation history) and the .eln export from them.
        _driver.assert_autolog_records(name)
        _driver.build_and_check_report(name)  # agent report  (auto-log HTML)
        _driver.build_and_check_eln(name)  # agent export-eln (.eln RO-Crate)
        _driver.build_and_check_history_html(name)  # agent history --html

        # ---- Stage 3: resume (plumbing, + optional conversational turn) --
        # Re-pass credentials: secrets are scrubbed from the committed image, so
        # resume must re-inject them or it prompts (e.g. OpenClaw's API key).
        _driver.drive_resume(
            name,
            runtime,
            boot_timeout=BOOT_TIMEOUT,
            agent_args=args,
            converse=RESUME_CONVERSE,
        )

        # ---- Stage 4: resume re-committed and history is still intact ----
        _driver.assert_status(name, "committed")
        _driver.assert_committed_image(runtime, name)
        _driver.load_history(name)  # must still parse
        if RESUME_CONVERSE:
            _assert_ping(name, minimum=2, stage="resume converse")
        passed = True
    finally:
        # Preserve artifacts (session dir, container, image) on failure when
        # SLA_E2E_KEEP_ON_FAIL=1, so history.json and container logs can be
        # inspected. On success (or without the flag) always clean up.
        if passed or not KEEP_ON_FAIL:
            _driver.cleanup_session(name, runtime)
        else:
            print(
                f"\n[SLA_E2E_KEEP_ON_FAIL] left artifacts for inspection:\n"
                f"  session dir : {_driver.get_sessions_dir() / name}\n"
                f"  container   : {_driver.container_name(name)}\n"
                f"  image       : {_driver.session_image_tag(name)}\n"
                f"  shared dir  : {_driver._shared_dir(name)}"
            )


def _assert_ping(name: str, minimum: int, stage: str) -> None:
    """Assert >= ``minimum`` sentinel ping calls; soft-xfail for fragile TUI turns."""
    calls = _driver.count_ping_calls(name)
    if calls >= minimum:
        return
    msg = (
        f"{stage}: expected >= {minimum} ping call(s), found {calls}. "
        f"{_driver.history_debug(name)}"
    )
    # PTY-driven turns depend on the agent's TUI accepting typed input; when not
    # strict we treat a miss as an expected failure rather than a hard error.
    if not STRICT_TUI and stage != "autonomous start":
        pytest.xfail(msg)
    raise AssertionError(msg)
