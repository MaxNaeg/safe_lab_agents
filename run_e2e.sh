#!/usr/bin/env bash
# Convenience runner for the full-pipeline (e2e) tests.
#
# These boot REAL containers and launch REAL agents across the
# {docker,podman} × {claude-code,openclaw} × {autonomous,interactive} matrix,
# then exercise resume. They cost tokens and take minutes. See docs/E2E_TESTING.md.
#
# Configure credentials via environment before running, e.g.:
#   export SLA_E2E_CLAUDE_OAUTH_TOKEN=...      # from `claude setup-token`
#   export SLA_E2E_OPENCLAW_API_KEY=...        # provider key
#   export SLA_E2E_OPENCLAW_MODEL=claude-sonnet-4-6
#   export SLA_E2E_OPENCLAW_PROVIDER=anthropic # default: anthropic
#
# Narrowing (optional): SLA_E2E_RUNTIMES=docker  SLA_E2E_AGENTS=claude-code
# Extra pytest args pass straight through, e.g.:
#   ./run_e2e.sh -k "docker and claude-code and autonomous"
set -euo pipefail

export SLA_E2E=1
export PYTHONUNBUFFERED=1   # stream pytest output line-by-line, not at the end

# If we're already inside the agents env, plain pytest streams to the TTY. Only
# wrap in `conda run` otherwise — and then with --no-capture-output, since conda
# run buffers the child's stdout by default and pytest's -v lines wouldn't stream.
# (Detection avoids `conda env list | grep -q`: grep -q closes the pipe on first
# match, which makes conda dump a BrokenPipeError report.)
RUNNER=(pytest)
if [ "${CONDA_DEFAULT_ENV:-}" != "agents" ] && command -v conda >/dev/null 2>&1; then
    _envs=$(conda env list 2>/dev/null || true)
    if [[ "$_envs" == *"/envs/agents"* ]]; then
        RUNNER=(conda run --no-capture-output -n agents pytest)
    fi
fi

# Preview exactly what THIS invocation will do. We let pytest itself resolve the
# selection (so any -k / -m / node-id args are honoured), then annotate each
# selected cell as RUN or SKIP using the same gating logic the tests use — a
# single source of truth, no drift.
echo "This invocation will run the following cells (pytest -k/-m applied):"
# Let pytest resolve the selection so any -k / -m / node-id args are honoured,
# then annotate each cell as RUN or SKIP with the tests' own gating logic.
# `|| true` keeps a narrow -k that matches nothing from aborting under pipefail.
# The annotator runs as a temp file (not `python - <<HEREDOC`): `conda run` does
# not forward heredoc stdin to the child, and cells arrive via an env var.
PREVIEW_CELLS=$(
    "${RUNNER[@]}" --collect-only -q -m e2e tests/e2e "$@" 2>/dev/null \
        | grep -oE 'test_full_pipeline\[[^]]+\]' \
        | sed -E 's/test_full_pipeline\[(.*)\]/\1/' \
        | sort -u
) || true
PREVIEW_PY=$(mktemp -t sla_e2e_preview.XXXXXX.py)
trap 'rm -f "$PREVIEW_PY"' EXIT
cat >"$PREVIEW_PY" <<'PY'
import os, sys
sys.path.insert(0, "tests")
from e2e.conftest import skip_reason  # reuse the tests' own gating

cells = [c for c in os.environ.get("PREVIEW_CELLS", "").split() if c]
if not cells:
    print("  (none selected — check your -k / -m expression)")
for cid in cells:
    parts = cid.split("-")            # e.g. docker-claude-code-autonomous
    runtime, mode = parts[0], parts[-1]
    agent = "-".join(parts[1:-1])     # agent may contain a hyphen (claude-code)
    reason = skip_reason(runtime, agent)
    print(f"  {cid:<40} {'SKIP: ' + reason if reason else 'RUN'}")
PY
PREVIEW_CELLS="$PREVIEW_CELLS" "${RUNNER[@]/%pytest/python}" "$PREVIEW_PY" || true
echo

exec "${RUNNER[@]}" -m e2e tests/e2e -v "$@"
