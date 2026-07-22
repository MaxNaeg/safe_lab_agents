#!/usr/bin/env bash
# =============================================================================
# Entrypoint for the Claude Code agent container.
#
# Environment variables (set by the DockerManager):
#   MCP_PORT      – TCP port of the MCP server on the host.
#   MCP_HOST      – Hostname/IP of the MCP server (defaults to host.docker.internal;
#                   set to the WSL gateway IP for Podman on Windows).
#   MODE          – "interactive" or "autonomous".
#   TASK_PROMPT   – The task description (only used in autonomous mode).
#   CONTEXT_DIR   – Path to the read-only context directory (optional).
#   NO_WEB        – If "true", the WebSearch/WebFetch tools are hard-blocked in
#                   every mode via --disallowedTools (deny rules apply even under
#                   --dangerously-skip-permissions).  This is a TOOL-level block,
#                   not a network one: other egress vectors (Bash curl/wget,
#                   Python) are covered only by a system-prompt restriction, since
#                   the container keeps full network egress (required for the
#                   agent's own model API).
#   CLAUDE_MODEL  – Optional model alias or full name passed to --model.
#   CLAUDE_EFFORT – Optional effort level (low/medium/high/xhigh/max) passed to --effort.
#   EGRESS_LOCKDOWN – If "true", apply the in-container egress firewall
#                   (/firewall.sh) before dropping privileges: the host is then
#                   reachable ONLY on the MCP port and private/LAN ranges are
#                   blocked, while the public internet (model API) stays open.
# =============================================================================
set -euo pipefail

# ---- Egress lockdown + privilege drop (root phase) ----
# The container starts as root (no USER directive) solely so this block can
# install the egress firewall, which needs CAP_NET_ADMIN. It must run before
# anything else — especially before any file is written — and must not create
# files itself (root-owned files in the bind mounts would be unwritable for
# the host user). It then permanently drops to the 'agent' user via setpriv
# and re-execs this script; the exec keeps PID 1 and the controlling TTY, so
# interactive/resume/login flows behave exactly as before. After the drop no
# capabilities remain and no-new-privileges prevents reacquisition, so the
# firewall rules are immutable from inside the container.
if [ "$(id -u)" = "0" ]; then
    if [ "${EGRESS_LOCKDOWN:-}" = "true" ]; then
        if ! /firewall.sh; then
            echo "ERROR: could not apply the egress firewall (host/LAN lockdown)." >&2
            echo "If your container runtime cannot support in-container iptables," >&2
            echo "rerun with --no-egress-lockdown to start without it." >&2
            exit 1
        fi
    fi
    # The runtime chowns the console device to the image's configured user
    # (root here); hand it to 'agent' so TUIs can reopen /dev/tty.
    chown agent "$(tty)" 2>/dev/null || true
    exec setpriv --reuid=agent --regid=agent --init-groups --inh-caps=-all \
        env HOME=/home/agent USER=agent LOGNAME=agent /entrypoint.sh "$@"
fi

# ---- Make everything the agent creates writable from the host ----
# The agent runs as the non-root 'agent' user, whose UID never matches the host
# user (and maps to a subuid under rootless Podman). Files it writes into the
# bind mounts would default to 0644/0755, leaving the host user unable to edit
# or clean them up — and the host can't chown/chmod them back (it isn't the
# owner and lacks CAP_FOWNER for a subuid). umask 000 makes new files 0666 and
# dirs 0777, so the host (the 'other' class) can read, write, and delete them.
# Explicit chmods below (e.g. the 600 on credentials) still take precedence.
umask 000

# ---- Seed OAuth credentials from host ----
# The host credential JSON ({"claudeAiOauth": {...}}) is passed via env var.
# Write it to the file Claude Code reads on Linux when no secret service is available.
if [ -n "${CLAUDE_CREDENTIALS_JSON:-}" ]; then
    mkdir -p /home/agent/.claude
    printf '%s' "$CLAUDE_CREDENTIALS_JSON" > /home/agent/.claude/.credentials.json
    chmod 600 /home/agent/.claude/.credentials.json
fi


# ---- Login-only bootstrap mode (passed as first argument) ----
# Mint a long-lived OAuth token via `claude setup-token`, then exit.  setup-token
# suppresses its sign-in UI when stdout is not a terminal, so we run it under
# `script`, which gives it a PTY while recording the session to a file.  The host
# extracts the printed token from that recording and seeds it into the real run.
# Wide columns keep the ~108-char token on a single unwrapped line.
# Skips MCP setup and the agent session entirely.
if [ "${1:-}" = "--login" ]; then
    mkdir -p /home/agent/.claude
    stty cols 400 2>/dev/null || true
    script -q -c "claude setup-token" /home/agent/.setup-token.log || true
    exit 0
fi


# ---- Network resilience: abort & retry stalled streams ----
# On the direct Anthropic API route (OAuth credentials), Claude Code has no
# default idle-stream timeout — that abort-on-stall default ships only for the
# Vertex/Foundry routes.  Podman's userspace network proxy (gvproxy) — used on
# macOS (applehv/vfkit) and on the Windows/WSL2 VM alike, as well as Docker
# Desktop's NAT — silently reaps an idle TCP flow without an RST.  During a
# cold-cache, max-effort first turn the time-to-first-byte (or a mid-stream
# gap) is long enough that the socket is dropped, and the CLI stalls mid-turn
# ("thinking", tokens frozen) until a watchdog aborts and retries.
#
# The window measures wire silence (no bytes/SSE events), NOT thinking time:
# a long think still streams tokens and the API sends periodic ping events, so
# it never trips.  We enable both watchdogs and set the BYTE-level idle timeout
# via CLAUDE_BYTE_STREAM_IDLE_TIMEOUT_MS — this is the effective recovery timer
# and it has no floor (clamp 10s..30min).  Do NOT rely on
# CLAUDE_STREAM_IDLE_TIMEOUT_MS: the CLI applies Math.max(it, 300000), so it
# can never lower recovery, and merely setting it *suppresses* the built-in
# 180000 ms first-party byte-watchdog default (pinning recovery up to 5 min).
# 90000 ms is comfortably above any healthy cold first-byte latency (the CLI's
# slow-first-byte warning fires at 30s) while cutting the observed ~5 min stall
# to ~90s.  All values are overridable via env passed into the container.
export API_FORCE_IDLE_TIMEOUT="${API_FORCE_IDLE_TIMEOUT:-1}"
export CLAUDE_ENABLE_STREAM_WATCHDOG="${CLAUDE_ENABLE_STREAM_WATCHDOG:-1}"
export CLAUDE_ENABLE_BYTE_WATCHDOG="${CLAUDE_ENABLE_BYTE_WATCHDOG:-1}"
export CLAUDE_BYTE_STREAM_IDLE_TIMEOUT_MS="${CLAUDE_BYTE_STREAM_IDLE_TIMEOUT_MS:-90000}"
export API_TIMEOUT_MS="${API_TIMEOUT_MS:-600000}"

# ---- Configure MCP server connection ----
# Remove any stale entry from a previous session before re-adding
# (committed images already contain the previous MCP config).
# stdout is silenced ("Added HTTP MCP server … / Headers … / File modified …"
# would otherwise clutter the host terminal right after the start banner);
# stderr stays visible so real failures still surface.
claude mcp remove experiment-tools >/dev/null 2>&1 || true
if [ -n "${MCP_AUTH_TOKEN:-}" ]; then
    claude mcp add --transport http experiment-tools \
        "http://${MCP_HOST:-host.docker.internal}:${MCP_PORT}/mcp" \
        --header "Authorization: Bearer ${MCP_AUTH_TOKEN}" >/dev/null
else
    claude mcp add --transport http experiment-tools \
        "http://${MCP_HOST:-host.docker.internal}:${MCP_PORT}/mcp" >/dev/null
fi

# ---- Mark onboarding complete (only when credentials were injected) ----
# Claude Code shows the interactive login/onboarding flow when
# hasCompletedOnboarding is absent from ~/.claude.json.  Set it only when
# host credentials were injected so that the default (no-copy) path still
# triggers the normal login dialog.  Also set it when an OAuth token was
# injected (the
# autonomous login-bootstrap path) so `claude -p` is not gated by onboarding.
if [ -n "${CLAUDE_CREDENTIALS_JSON:-}" ] || [ -n "${CLAUDE_CODE_OAUTH_TOKEN:-}" ]; then
python3 - <<'PYEOF'
import json, pathlib
p = pathlib.Path.home() / ".claude.json"
try:
    d = json.loads(p.read_text())
except Exception:
    d = {}
d["hasCompletedOnboarding"] = True
d.setdefault("lastOnboardingVersion", "2.1.117")
p.write_text(json.dumps(d))
PYEOF
fi

# ---- Build system prompt ----
# The base environment prose is generated once on the host (BaseAgent.
# get_system_prompt) and written to system_prompt.txt, so it is not duplicated
# here. CONTEXT_DIR / NO_WEB are already reflected in that file.
SYSTEM_PROMPT=""
if [ -f "/agent/workspace/system_prompt.txt" ]; then
    SYSTEM_PROMPT="$(cat /agent/workspace/system_prompt.txt)"
fi

# ---- Python tools info (injected when PYTHON_TOOLS was declared) ----
if [ -f "/agent/workspace/python_tools_info.txt" ]; then
    SYSTEM_PROMPT="${SYSTEM_PROMPT}
$(cat /agent/workspace/python_tools_info.txt)"
fi

# ---- Auto-log info (injected when --auto-log is active) ----
if [ -f "/agent/workspace/auto_log_info.txt" ]; then
    SYSTEM_PROMPT="${SYSTEM_PROMPT}
$(cat /agent/workspace/auto_log_info.txt)"
fi

# ---- Tools-reload info (injected when --update-tools is active) ----
if [ -f "/agent/workspace/reload_info.txt" ]; then
    SYSTEM_PROMPT="${SYSTEM_PROMPT}
$(cat /agent/workspace/reload_info.txt)"
fi

# ---- Optional model / effort flags (apply to all modes) ----
CLAUDE_OPTS=()
[ -n "${CLAUDE_MODEL:-}" ] && CLAUDE_OPTS+=(--model "$CLAUDE_MODEL")
[ -n "${CLAUDE_EFFORT:-}" ] && CLAUDE_OPTS+=(--effort "$CLAUDE_EFFORT")

# ---- Resume mode (passed as first argument) ----
# Resume is always interactive (the host never resumes in autonomous mode), so
# this branch exits at the end and never falls through to the autonomous block.
if [ "${1:-}" = "--resume" ]; then
    SKIP_PERMS_FLAG=()
    if [ "${SKIP_PERMISSIONS:-}" = "true" ]; then
        SKIP_PERMS_FLAG=(--dangerously-skip-permissions)
    fi
    NO_WEB_FLAG=()
    if [ "${NO_WEB:-}" = "true" ]; then
        # Hard block on resume too (same as the interactive non-resume path).
        NO_WEB_FLAG=(--disallowedTools "WebSearch,WebFetch")
    fi

    # Resume by explicit session id, not `claude --continue`. `--continue` uses
    # Claude Code's session picker, which intentionally EXCLUDES headless
    # (`claude -p`) sessions — i.e. every autonomous run. So `--continue` reports
    # "No conversation found to continue" for those even though the transcript
    # exists on disk. Resuming by id works for both headless and interactive
    # sessions. The transcript filename is the session id; pick the most recently
    # modified transcript for this working directory's project, falling back to
    # the newest transcript across all projects if the cwd→slug mapping differs.
    PROJECTS_DIR="/home/agent/.claude/projects"
    CWD_SLUG="$(pwd | sed 's#/#-#g')"
    LATEST_TRANSCRIPT="$(ls -1t "$PROJECTS_DIR/$CWD_SLUG"/*.jsonl 2>/dev/null | head -n1 || true)"
    if [ -z "$LATEST_TRANSCRIPT" ]; then
        LATEST_TRANSCRIPT="$(ls -1t "$PROJECTS_DIR"/*/*.jsonl 2>/dev/null | head -n1 || true)"
    fi
    RESUME_FLAG=(--continue)
    if [ -n "$LATEST_TRANSCRIPT" ]; then
        SESSION_ID="$(basename "$LATEST_TRANSCRIPT" .jsonl)"
        echo "Resuming Claude Code session ${SESSION_ID} …"
        RESUME_FLAG=(--resume "$SESSION_ID")
    fi

    claude "${RESUME_FLAG[@]}" \
        --append-system-prompt "$SYSTEM_PROMPT" \
        "${NO_WEB_FLAG[@]}" \
        "${SKIP_PERMS_FLAG[@]}" \
        "${CLAUDE_OPTS[@]}"
    bash
    exit 0
fi

# ---- Autonomous-mode system-prompt instruction ----
# Appended after the shared prompt is fully built so it only appears in
# autonomous runs and does not affect interactive or resume sessions.
if [ "$MODE" = "autonomous" ]; then
    SYSTEM_PROMPT="${SYSTEM_PROMPT}
You are running in fully autonomous mode. Work independently to complete the task.
Do NOT ask the user questions, pause for confirmation, or wait for input — make reasonable decisions on your own and proceed until the task is complete."
fi

# ---- Autonomous mode ----
if [ "$MODE" = "autonomous" ]; then
    # No allowlist: --dangerously-skip-permissions bypasses allow rules
    # entirely (permission order: deny → allow → mode), so an allowlist would
    # be dead weight. Deny rules ARE evaluated before the bypass, so
    # --disallowedTools is a real hard block for the web tools.
    NO_WEB_FLAG=()
    if [ "${NO_WEB:-}" = "true" ]; then
        NO_WEB_FLAG=(--disallowedTools "WebSearch,WebFetch")
    fi
    exec claude -p "$TASK_PROMPT" \
        --verbose \
        --append-system-prompt "$SYSTEM_PROMPT" \
        "${NO_WEB_FLAG[@]}" \
        --output-format stream-json \
        --dangerously-skip-permissions \
        "${CLAUDE_OPTS[@]}"
fi

# ---- Interactive mode but no resume ----
if [ "${1:-}" != "--resume" ] && [ "$MODE" != "autonomous" ]; then
    SKIP_PERMS_FLAG=()
    if [ "${SKIP_PERMISSIONS:-}" = "true" ]; then
        SKIP_PERMS_FLAG=(--dangerously-skip-permissions)
    fi
    if [ "${NO_WEB:-}" = "true" ]; then
        # Hard block: deny web tools explicitly (strict, not just a permission prompt).
        claude --append-system-prompt "$SYSTEM_PROMPT" \
            --disallowedTools "WebSearch,WebFetch" \
            "${SKIP_PERMS_FLAG[@]}" \
            "${CLAUDE_OPTS[@]}"
    else
        claude --append-system-prompt "$SYSTEM_PROMPT" \
            "${SKIP_PERMS_FLAG[@]}" \
            "${CLAUDE_OPTS[@]}"
    fi
    bash
fi