# End-to-end pipeline testing

The unit tests under `tests/` mock Docker/Podman — fast, but they never prove the
real pipeline works. The **e2e** suite (`tests/e2e/`) does: it boots real
containers, launches real agents, and drives the full lifecycle

```
start (autonomous | interactive)  →  commit  →  resume
```

across the matrix

```
{docker, podman} × {claude-code, openclaw} × {autonomous, interactive}
```

= **8 cells**, each also exercising `resume`. Every cell asserts on the durable
artifacts the pipeline leaves behind: the `committed` status in `metadata.json`,
the sentinel `ping` tool call in `history.json`, and the committed session image.

Each cell runs with `--auto-log`, so it additionally exercises the **output
chain** end-to-end (Stage 2b):

- **auto-log records** — the ping call is recorded under `<shared>/auto_log/`
  (JSON, plus HDF5 for arrays), with the sentinel present;
- **auto-log HTML report** — `build_report()` (`agent report`) renders it;
- **`.eln` export** — `build_eln()` (`agent export-eln`) produces a valid
  RO-Crate zip mentioning the ping dataset;
- **history HTML report** — `build_conversation_html()` (`agent history --html`)
  renders the conversation.

> These cost tokens and take minutes (~1–4 min/cell; full matrix ~8 min with
> warm image caches, up to ~30 min on a cold first run that builds both images).
> They are **inert by default** — a plain `pytest` never collects them.

## Platform support

| Platform | Supported | Notes |
|----------|-----------|-------|
| **Linux** (incl. Ubuntu) | ✅ Yes | The ideal target: native Docker & Podman (no VM). The daemon must be running / your user in the `docker` group (`runtime.py` will `systemctl start` Docker). |
| **macOS** | ✅ Yes | Docker Desktop / `podman machine` VMs are auto-started by `runtime.py`. Verified: full 8-cell matrix green. |
| **Windows** | ✅ Yes (native) | Needs `pip install pywinpty` (the ConPTY backend for the interactive/resume console) and Docker Desktop. Run with `python run_e2e.py …` (not the bash `run_e2e.sh`). Native Windows is the *only* way to exercise the product's Windows-specific branches (interactive-start subprocess fallback, Windows Docker/Podman detection, the Podman firewall notice) — WSL2 would take the Linux branches instead. |

The pseudo-terminal that drives interactive/resume cells is abstracted in
[`tests/e2e/_console.py`](../tests/e2e/_console.py): POSIX uses the stdlib `pty`;
Windows uses ConPTY via `pywinpty`. Everything else (autonomous cells, runtime
CLI calls, report/`.eln`/history builders) is already cross-platform.

**Host does not need the agent CLIs installed.** Agents run inside the container
image, so the Windows host needs only Docker/Podman, the Python env, and the
credential env vars. Generate the Claude token anywhere (`claude setup-token`,
e.g. on a Mac) and set `SLA_E2E_CLAUDE_OAUTH_TOKEN` — it is account-scoped, not
machine-scoped.

> **Windows caveat (unverified):** the ConPTY backend is best-effort and has not
> yet been run on a Windows machine. ConPTY differs from a POSIX pty in ANSI
> handling and Ctrl-C delivery (the graceful-teardown interrupt goes in via
> `pywinpty`'s `sendintr()`), so expect a round or two of on-machine iteration —
> `SLA_E2E_KEEP_ON_FAIL=1` preserves the PTY capture (`sla-e2e-pty-<name>.log`)
> for exactly that debugging.

## Enabling

Everything is gated behind `SLA_E2E=1`. Without it the whole `tests/e2e/` package
is skipped at collection time. Each cell additionally self-skips when its runtime
is unavailable or its agent's credentials are unset, so a partial environment
(e.g. Docker only, one agent) still produces a meaningful green subset.

## Credentials

Credentials are injected non-interactively via `--agent-args` and are never
persisted (popped before metadata is written, scrubbed from the committed image).

| Agent | Environment variables |
|-------|-----------------------|
| `claude-code` | `SLA_E2E_CLAUDE_OAUTH_TOKEN` — a token from `claude setup-token` |
| `openclaw` | `SLA_E2E_OPENCLAW_API_KEY`, `SLA_E2E_OPENCLAW_MODEL`, and `SLA_E2E_OPENCLAW_PROVIDER` (default `anthropic`) |

## Running

```bash
export SLA_E2E_CLAUDE_OAUTH_TOKEN="$(claude setup-token)"   # example
./run_e2e.sh                       # prints the plan, then runs all available cells
```

On **Windows**, use the bash-free runner instead (same behaviour):

```powershell
python run_e2e.py -k "docker and claude-code and autonomous"
```

`run_e2e.sh` sets `SLA_E2E=1`, prints which cells will run vs skip, and runs the
suite in the `agents` conda env if present. Extra args pass straight to pytest:

```bash
./run_e2e.sh -k "docker and claude-code and autonomous"   # one cell
./run_e2e.sh -k "interactive"                             # only interactive-start cells
```

Or invoke pytest directly:

```bash
SLA_E2E=1 conda run -n agents pytest -m e2e tests/e2e -v
```

## Knobs

| Variable | Effect |
|----------|--------|
| `SLA_E2E_RUNTIMES` | Comma list to restrict runtimes (default: all installed). |
| `SLA_E2E_AGENTS` | Comma list to restrict agents (default: all credentialed). |
| `SLA_E2E_RESUME_CONVERSE=1` | Drive a real conversational turn on resume and assert a *second* ping call (fragile; off by default). |
| `SLA_E2E_KEEP_ON_FAIL=1` | On failure, keep the session dir, container, committed image, shared/`auto_log` dir, and the PTY capture (`sla-e2e-pty-<name>.log`) for inspection instead of cleaning up. |
| `SLA_E2E_STRICT_TUI` | `1` (default): a PTY-driven turn with no tool call is a hard failure. `0`: soft-`xfail` for the interactive/converse turns (the autonomous path always hard-fails). |
| `SLA_E2E_AUTONOMOUS_TIMEOUT` | Seconds for an autonomous cell incl. first-time image build (default 900). |
| `SLA_E2E_BOOT_TIMEOUT` | Seconds to wait for an interactive container to reach `running` (default 300). |

## How it works

- **Autonomous** cells run `agent start --task …` (non-interactive, self-exits,
  auto-commits) and capture output as a subprocess.
- **Interactive** cells and **resume** run the CLI under a real pseudo-terminal
  (stdlib `pty`, no extra dependency): wait for the container to reach `running`,
  optionally type one prompt and let output settle, then send `Ctrl-C` to trigger
  the CLI's graceful teardown → (re-)commit.
- The sentinel tool lives in [`tests/e2e/tools_ping.py`](../tests/e2e/tools_ping.py);
  its unique `SENTINEL` string is what assertions look for in `history.json`.

## Non-regression

`pytest` and `pytest -m "not e2e"` collect and run exactly as before — the e2e
package is excluded unless `SLA_E2E=1`.
