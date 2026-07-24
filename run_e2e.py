#!/usr/bin/env python3
"""Cross-platform runner for the e2e pipeline tests (works on Windows too).

Equivalent to ``run_e2e.sh`` but bash-free, so it runs on native Windows where
``run_e2e.sh`` cannot. Sets ``SLA_E2E=1``, prints which matrix cells are
eligible (given installed runtimes + configured credentials), then runs pytest.
Extra arguments pass straight through, e.g.::

    python run_e2e.py -k "docker and claude-code and autonomous"

See docs/E2E_TESTING.md for credentials and platform notes. On Windows the
interactive/resume cells need ``pip install pywinpty``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> int:
    os.environ["SLA_E2E"] = "1"
    os.environ["PYTHONUNBUFFERED"] = "1"
    # Import the tests' own gating so the preview never drifts from real behaviour.
    sys.path.insert(0, str(ROOT / "tests"))
    from e2e.conftest import ALL_AGENTS, ALL_MODES, ALL_RUNTIMES, skip_reason

    print("Eligible cells (credential/runtime gating only — pytest -k / -m below")
    print("narrows this further; see the 'selected/deselected' line):")
    for runtime in ALL_RUNTIMES:
        for agent in ALL_AGENTS:
            reason = skip_reason(runtime, agent)
            for mode in ALL_MODES:
                cid = f"{runtime}-{agent}-{mode}"
                print(f"  {cid:<40} {'SKIP: ' + reason if reason else 'RUN'}")
    print(flush=True)  # flush so the preview prints before pytest's output

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-m",
        "e2e",
        str(ROOT / "tests" / "e2e"),
        "-v",
        *sys.argv[1:],
    ]
    return subprocess.run(cmd, cwd=ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
