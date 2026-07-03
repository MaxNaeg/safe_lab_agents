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
#   NO_WEB        – If "true", injects a system-prompt restriction into
#                   SOUL.md. NOTE: soft restriction only — OpenClaw has no
#                   CLI flag to hard-block tool access the way Claude Code does.
#   LLM_API_KEY   – Generic API key (always set when any key is provided).
#   ANTHROPIC_API_KEY / OPENAI_API_KEY / GOOGLE_API_KEY / OPENROUTER_API_KEY
#                 – Provider-specific key (set alongside LLM_API_KEY).
#   LLM_PROVIDER  – Provider identifier (anthropic, openai, google, openrouter).
#   LLM_MODEL     – Full model ID in provider/model format (e.g.
#                   "anthropic/claude-sonnet-4-6").  Also set as the config
#                   default so the interactive TUI uses it without re-prompting.
# =============================================================================
set -euo pipefail

# ---- Make everything the agent creates writable from the host ----
# The agent runs as the non-root 'agent' user, whose UID never matches the host
# user (and maps to a subuid under rootless Podman). Files it writes into the
# bind mounts would default to 0644/0755, leaving the host user unable to edit
# or clean them up — and the host can't chown/chmod them back (it isn't the
# owner and lacks CAP_FOWNER for a subuid). umask 000 makes new files 0666 and
# dirs 0777, so the host (the 'other' class) can read, write, and delete them.
umask 000

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
        openclaw onboard \
            --non-interactive \
            --accept-risk \
            --auth-choice "${LLM_PROVIDER}-api-key" \
            "$KEY_FLAG" "$LLM_API_KEY" \
            --skip-health \
        || { echo "ERROR: API key registration failed for provider '${LLM_PROVIDER}'. Check that the key is valid." >&2; exit 1; }
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
# here. CONTEXT_DIR / NO_WEB are already reflected in that file.
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
