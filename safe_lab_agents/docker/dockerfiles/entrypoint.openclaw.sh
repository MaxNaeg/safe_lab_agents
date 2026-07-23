#!/usr/bin/env bash
# =============================================================================
# Entrypoint for the OpenClaw agent container.
#
# Environment variables (set by the DockerManager):
#   MCP_PORT      – TCP port of the MCP server on the host.
#   MCP_HOST      – Hostname/IP of the MCP server (defaults to host.docker.internal;
#                   set to the WSL gateway IP for Podman on Windows).
#   MODE          – "interactive" or "autonomous".
#   TASK_PROMPT   – The task description (only used in autonomous mode).
#   CONTEXT_DIR   – Path to the read-only context directory (optional).
#   (no NO_WEB here: this entrypoint reads no such variable. The --no-web
#    restriction reaches OpenClaw purely through system_prompt.txt, written on
#    the host. It is a soft restriction only — OpenClaw has no CLI flag to
#    hard-block tool access the way Claude Code does.)
#   LLM_API_KEY   – Generic API key (always set when any key is provided).
#   ANTHROPIC_API_KEY / OPENAI_API_KEY / GOOGLE_API_KEY / OPENROUTER_API_KEY
#                 – Provider-specific key (set alongside LLM_API_KEY).
#   LLM_PROVIDER  – Provider identifier (anthropic, openai, google, openrouter).
#   LLM_MODEL     – Full model ID in provider/model format (e.g.
#                   "anthropic/claude-sonnet-4-6").  Also set as the config
#                   default so the interactive TUI uses it without re-prompting.
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
# interactive/resume flows behave exactly as before. After the drop no
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
umask 000

# ---- Un-world-write OpenClaw's plugin tree (umask 000 side effect) ----
# umask 000 (above) is for the /agent/* bind mounts: it lets the host edit and
# delete files the container's 'agent' user writes there despite the UID
# mismatch. As a side effect it also makes everything OpenClaw materializes
# under ~/.openclaw/npm world-writable (0777) — and OpenClaw's own plugin loader
# refuses to load any plugin from a world-writable path (e.g. the codex provider
# plugin), emitting repeated "blocked plugin candidate" warnings.
#
# That npm tree is container-internal: it is never bind-mounted and never copied
# out (copy_agent_logs deliberately extracts only ~/.openclaw/agents, not npm),
# and the 'agent' user owns it, so stripping *group/other* write costs nothing —
# the owner keeps full access. We scope the chmod to ~/.openclaw/npm on purpose:
# ~/.openclaw/agents must stay world-writable so the host can clean up the logs
# that get docker-cp'd out of it on Linux (a foreign-UID directory).
#
# This first pass fixes a codex dir already present at 0777 when *resuming* a
# committed image, before the setup commands below would otherwise re-warn.
chmod -R go-w /home/agent/.openclaw/npm 2>/dev/null || true

# ---- Register MCP server ----
if [ -n "${MCP_AUTH_TOKEN:-}" ]; then
    openclaw mcp set experiment-tools \
        "{\"url\": \"http://${MCP_HOST:-host.docker.internal}:${MCP_PORT}/mcp\", \"transport\": \"streamable-http\", \"headers\": {\"Authorization\": \"Bearer ${MCP_AUTH_TOKEN}\"}}"
else
    openclaw mcp set experiment-tools \
        "{\"url\": \"http://${MCP_HOST:-host.docker.internal}:${MCP_PORT}/mcp\", \"transport\": \"streamable-http\"}"
fi

# ---- Register provider API key via non-interactive onboarding ----
# models.providers.<provider> is for custom/proxy providers (needs baseUrl).
# models auth paste-token opens /dev/tty even when stdin is piped.
# The correct path is `openclaw onboard --non-interactive` with a
# provider-specific key flag. --skip-health writes config only, no daemon.
if [ -n "${LLM_PROVIDER:-}" ] && [ -n "${LLM_API_KEY:-}" ]; then
    case "$LLM_PROVIDER" in
        openai)     KEY_FLAG="--openai-api-key" ;;
        anthropic)  KEY_FLAG="--anthropic-api-key" ;;
        google)     KEY_FLAG="--google-api-key" ;;
        openrouter) KEY_FLAG="--openrouter-api-key" ;;
        *)          KEY_FLAG="" ;;
    esac
    if [ -n "$KEY_FLAG" ]; then
        # Run onboarding under a normal umask, NOT umask 000. onboard writes only
        # to ~/.openclaw — config plus any plugins it installs/refreshes with a
        # valid key (notably the codex provider plugin) — and never to the
        # /agent/* bind mounts, so it does not need the world-writable umask.
        # Under umask 000 the plugin dirs it materializes are born 0777, and
        # OpenClaw's own loader then refuses to load them, spamming
        # "blocked plugin candidate: world-writable path" warnings on every
        # subsequent command. A normal umask makes them 0755 and silent.
        umask 022
        openclaw onboard \
            --non-interactive \
            --accept-risk \
            --auth-choice "${LLM_PROVIDER}-api-key" \
            "$KEY_FLAG" "$LLM_API_KEY" \
            --skip-health \
        || { echo "ERROR: API key registration failed for provider '${LLM_PROVIDER}'. Check that the key is valid." >&2; exit 1; }
        # Restore the world-writable umask for the agent run (bind-mount writes).
        umask 000
        # onboard may have (re)created the session-log tree under umask 022, i.e.
        # 0755. Those logs are docker-cp'd out at shutdown and the host must be
        # able to delete them (a foreign UID on Linux), so restore world-write on
        # that subtree only. The plugin tree under ~/.openclaw/npm is intentionally
        # left non-world-writable (it is never copied out) so plugins keep loading.
        chmod -R go+w /home/agent/.openclaw/agents 2>/dev/null || true
    fi
fi

# ---- Set default model (applies to both interactive and autonomous runs) ----
if [ -n "${LLM_MODEL:-}" ]; then
    openclaw config set agents.defaults.model.primary "${LLM_MODEL}"
fi

# ---- Point openclaw at the workspace where SOUL.md lives ----
openclaw config set agents.defaults.workspace /home/agent/.openclaw/workspace

# ---- Build and write SOUL.md (system prompt injected by openclaw) ----
SOUL="/home/agent/.openclaw/workspace/SOUL.md"
mkdir -p "$(dirname "$SOUL")"

# The base environment prose is generated once on the host (OpenClawAgent.
# get_system_prompt) and written to system_prompt.txt, so it is not duplicated
# here. The context dir and any --no-web restriction are already reflected in
# that file.
if [ -f "/agent/workspace/system_prompt.txt" ]; then
    cat /agent/workspace/system_prompt.txt > "$SOUL"
else
    : > "$SOUL"
fi

if [ "$MODE" = "autonomous" ]; then
    cat >> "$SOUL" <<'SOULDOC'
You are running in fully autonomous mode. Work independently to complete the
task. Do NOT ask the user questions, pause for confirmation, or wait for
input — make reasonable decisions on your own and proceed until the task is
complete.
SOULDOC
fi

if [ -f "/agent/workspace/python_tools_info.txt" ]; then
    cat /agent/workspace/python_tools_info.txt >> "$SOUL"
fi

if [ -f "/agent/workspace/auto_log_info.txt" ]; then
    cat /agent/workspace/auto_log_info.txt >> "$SOUL"
fi

if [ -f "/agent/workspace/reload_info.txt" ]; then
    cat /agent/workspace/reload_info.txt >> "$SOUL"
fi

# ---- Safety net: un-world-write the plugin tree once more, pre-launch ----
# onboard already installs under a normal umask (0755) above, so this is belt-
# and-suspenders: should any later step under umask 000 re-materialize a plugin
# dir at 0777 (e.g. a config reload triggering a refresh), strip group/other
# write before the agent run. A no-op when everything is already 0755. Verified
# to persist across plugin reloads and the run. See the note by the first pass.
chmod -R go-w /home/agent/.openclaw/npm 2>/dev/null || true

# ---- Resume mode ----
if [ "${1:-}" = "--resume" ]; then
    openclaw tui --local || true
    echo "Shutting down — this may take a moment …"
    stty sane 2>/dev/null || true
    bash
    exit 0
fi

# ---- Autonomous mode ----
if [ "$MODE" = "autonomous" ]; then
    SESSION_ID=$(openssl rand -hex 12)

    MODEL_FLAG=()
    [ -n "${LLM_MODEL:-}" ] && MODEL_FLAG=(--model "$LLM_MODEL")

    # Run the agent in the background and suppress its plain-text stdout.
    # The session JSONL file is written incrementally as records arrive, so
    # tailing it gives structured real-time output that the Python formatter
    # can render as ⚙ tool calls and result lines.
    openclaw agent \
        --session-id "$SESSION_ID" \
        --local \
        "${MODEL_FLAG[@]}" \
        --message "$TASK_PROMPT" >/dev/null &
    AGENT_PID=$!

    # Poll until the session JSONL file appears or the agent exits.
    SESSION_FILE=""
    while kill -0 "$AGENT_PID" 2>/dev/null; do
        F=$(find /home/agent/.openclaw/agents -name "${SESSION_ID}.jsonl" \
            2>/dev/null | head -1)
        if [ -n "$F" ]; then
            SESSION_FILE="$F"
            break
        fi
        sleep 0.5
    done

    # One final check after the agent process exits.
    if [ -z "$SESSION_FILE" ]; then
        SESSION_FILE=$(find /home/agent/.openclaw/agents -name "${SESSION_ID}.jsonl" \
            2>/dev/null | head -1)
    fi

    if [ -n "$SESSION_FILE" ]; then
        echo "OPENCLAW_JSONL: $SESSION_FILE"
        tail -n +1 -f "$SESSION_FILE" --pid="$AGENT_PID" 2>/dev/null
    else
        echo "OPENCLAW_JSONL: not found"
        echo "OPENCLAW_DEBUG: $(find /home/agent/.openclaw -type f 2>/dev/null \
            | head -10 | tr '\n' ' ' || echo 'dir not found')"
    fi

    wait "$AGENT_PID"
    exit $?
fi

# ---- Interactive mode ----
openclaw tui --local || true
echo "Shutting down — this may take a moment …"
stty sane 2>/dev/null || true
bash
