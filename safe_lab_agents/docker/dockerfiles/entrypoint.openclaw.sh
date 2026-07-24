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

# ---- Scrub provider credentials before the container is committed ----
# The provider API key reaches OpenClaw via env vars (which the host blanks at
# commit time). But the bundled `codex` provider plugin also writes the raw key
# to ~/.openclaw/agents/**/codex-home/auth.json, and `openclaw onboard` can
# record an auth profile in ~/.openclaw/state/openclaw.sqlite. Both live inside
# the container filesystem, so they would persist in the committed session image
# — and, because copy_agent_logs pulls ~/.openclaw/agents out to the host, in
# the host session logs too — where anyone with container-runtime access could
# read them. This entrypoint is PID 1 in every OpenClaw mode (none `exec` away),
# and the host commits (and copies logs from) the *exited* container, so an EXIT
# trap runs before the snapshot is taken. Resume re-onboards with a freshly
# supplied key, so clearing is loss-free. Best-effort — never block shutdown.
_scrub_credentials() {
    find /home/agent/.openclaw/agents -name auth.json -delete 2>/dev/null || true
    python3 - <<'PYEOF' 2>/dev/null || true
import sqlite3
try:
    con = sqlite3.connect("/home/agent/.openclaw/state/openclaw.sqlite")
    tables = [r[0] for r in con.execute(
        "select name from sqlite_master where type='table' and name like 'auth_profile%'"
    )]
    for t in tables:
        con.execute(f"delete from {t}")
    con.commit()
    con.execute("vacuum")
    con.close()
except Exception:
    pass
PYEOF
}
trap _scrub_credentials EXIT


# ---- Prefer npm's offline cache for container-start plugin resolution ----
# Launching the interactive TUI (and `openclaw onboard`) makes OpenClaw resolve
# its provider plugins — notably @openclaw/codex — via `npm install`. Even though
# that plugin is pre-baked into the image at the matching version (see
# Dockerfile.openclaw), npm still performs a network *revalidation* of the
# packument against registry.npmjs.org before trusting the cached copy. The npm
# registry is public (the egress firewall leaves it reachable), but a container
# whose resolver is the host's LAN DNS with corporate search domains — the norm
# under rootless Podman, which copies the host's /etc/resolv.conf verbatim,
# unlike Docker's clean embedded resolver — makes npm's getaddrinfo path flake
# with EAI_AGAIN. npm then dies after 3 retries and `openclaw onboard` hangs, so
# the TUI never renders (Docker/macOS avoid this only by having reliable DNS).
#
# Everything the agent needs at container start is already baked, so prefer the
# offline cache: with the packument+tarball cached, npm serves the plugin from
# ~/.npm without any registry round-trip, making startup deterministic across
# runtimes. `prefer-offline` (not hard `offline`) still lets a genuinely-missing
# plugin fetch over the network when DNS is healthy, so nothing regresses.
export NPM_CONFIG_PREFER_OFFLINE=true

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
# Reopen the DEFAULT "main" session explicitly so the TUI continues the SAME
# conversation the autonomous run recorded (it stores its transcript under key
# agent:main:main — see the autonomous block below). `openclaw tui --local`
# already defaults to --session main, but we pass it explicitly so resume never
# silently depends on that default staying in sync with the agent side. Without
# targeting this session the TUI opens a brand-new empty one with no memory of
# the previous autonomous run.
if [ "${1:-}" = "--resume" ]; then
    openclaw tui --local --session main || true
    echo "Shutting down — this may take a moment …"
    stty sane 2>/dev/null || true
    bash
    exit 0
fi

# ---- Autonomous mode ----
if [ "$MODE" = "autonomous" ]; then
    MODEL_FLAG=()
    [ -n "${LLM_MODEL:-}" ] && MODEL_FLAG=(--model "$LLM_MODEL")

    # Record the run under the DEFAULT "main" session (key agent:main:main)
    # instead of a random --session-id. This is what makes `agent resume` work:
    # the resume path above runs `openclaw tui --local --session main`, which
    # reopens exactly this session and replays its history. A random --session-id
    # would be stored under agent:main:explicit:<id> — a key the TUI's main
    # session never reopens — so a resumed run would start with no memory of the
    # autonomous work (the bug this fixes).
    #
    # openclaw assigns the session's JSONL filename itself (a UUID we don't know
    # up front), so we discover it below rather than predicting it as before.
    openclaw agent \
        --session-key main \
        --local \
        "${MODEL_FLAG[@]}" \
        --message "$TASK_PROMPT" >/dev/null &
    AGENT_PID=$!

    # Locate the session's record stream, which openclaw writes (incrementally,
    # so tailing it gives live output) to
    #   /home/agent/.openclaw/agents/<agent>/sessions/<uuid>.jsonl
    # The search is deliberately narrow:
    #   * -maxdepth 1 into the per-agent "sessions/" dir only — NOT a recursive
    #     walk of the whole agents tree. The codex runtime writes its own rollout
    #     "*.jsonl" files under agents/<agent>/agent/codex-home/… (the same trees
    #     parse_conversation_history prunes via _PRUNE_DIRS); a recursive match
    #     could return one of those and tail would follow the wrong file, so
    #     nothing would stream.
    #   * ! -name '*.trajectory.jsonl' — skip the sibling human-readable progress
    #     file; we want the record stream the Python formatter parses.
    # On a fresh autonomous start this yields exactly one file (a freshly built
    # image and the config/onboard steps above create no sessions).
    find_session_file() {
        find /home/agent/.openclaw/agents/*/sessions -maxdepth 1 \
            -name '*.jsonl' ! -name '*.trajectory.jsonl' 2>/dev/null | head -1
    }

    # Poll until the session JSONL file appears or the agent exits.
    SESSION_FILE=""
    while kill -0 "$AGENT_PID" 2>/dev/null; do
        F=$(find_session_file)
        if [ -n "$F" ]; then
            SESSION_FILE="$F"
            break
        fi
        sleep 0.5
    done

    # One final check after the agent process exits.
    if [ -z "$SESSION_FILE" ]; then
        SESSION_FILE=$(find_session_file)
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
