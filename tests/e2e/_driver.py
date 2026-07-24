"""Drivers and assertion helpers for the end-to-end pipeline tests.

Deliberately free of pytest imports so the helpers can also be exercised from a
plain script. The system-under-test is the real ``agent`` CLI, invoked as a
subprocess (``python -m safe_lab_agents.cli``) so nothing here re-implements the
internal start/resume logic — it drives the same entry points a user would.

Two launch shapes:

* :func:`run_autonomous` — non-interactive ``start --task …``. The container runs
  the task and self-exits; the call blocks until then and captures output.
* :func:`drive_pty_session` — the single PTY driver for every interactive launch
  (interactive ``start``, ``resume``, and the opt-in converse-on-resume turn). It
  attaches a real pseudo-terminal, optionally types one prompt, waits for the
  output to settle, then sends Ctrl-C to trigger the CLI's graceful teardown.

Assertions read the durable artifacts the pipeline leaves on disk
(``metadata.json`` status, ``history.json`` tool calls) plus the committed image.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from safe_lab_agents.config import get_sessions_dir

from . import _console
from .tools_ping import SENTINEL

# Strips CSI escape sequences (colours, cursor moves). Claude Code's TUI positions
# each word of a prompt with a cursor-move escape, so dialog text like "trust this
# folder" is NOT a contiguous substring of the raw bytes — match on stripped text.
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")

# The sentinel tools file, alongside this module.
TOOLS_FILE = Path(__file__).with_name("tools_ping.py")


def _shared_dir(name: str) -> Path:
    """Stable per-session shared directory (keyed by name so cleanup can find it)."""
    return Path(tempfile.gettempdir()) / f"sla-e2e-shared-{name}"


def pty_log_path(name: str) -> Path:
    """Stable path where drive_pty_session dumps the raw PTY capture for debugging."""
    return Path(tempfile.gettempdir()) / f"sla-e2e-pty-{name}.log"


def auto_log_dir(name: str) -> Path:
    """Host path of the session's auto_log/ folder (``<shared>/auto_log``).

    Mirrors cli.start: with --auto-log and --shared, AUTO_LOG_DIR is
    ``<shared>/auto_log`` on the host, so records are readable here.
    """
    return _shared_dir(name) / "auto_log"


def _artifact_dir(name: str) -> Path:
    """Scratch dir for report/.eln/history HTML we build during assertions."""
    return Path(tempfile.gettempdir()) / f"sla-e2e-out-{name}"


# Image naming mirrors DockerManager._session_image_tag: the committed image for a
# session is "safe-lab-agents-session-<name>:latest".
_SESSION_IMAGE_PREFIX = "safe-lab-agents-session-"

# Container naming mirrors DockerManager.create_container: the running container
# for a session is "safe-lab-agents-<name>" (manager.py: name=f"safe-lab-agents-…").
_CONTAINER_PREFIX = "safe-lab-agents-"


def container_name(name: str) -> str:
    """Return the runtime container name for a session (matches manager.py)."""
    return f"{_CONTAINER_PREFIX}{name}"


# ----------------------------------------------------------------------
# CLI invocation
# ----------------------------------------------------------------------
def cli_argv(*args: str) -> list[str]:
    """Return the argv to invoke the ``agent`` CLI as a module (PATH-independent)."""
    return [sys.executable, "-m", "safe_lab_agents.cli", *args]


def _common_start_args(
    agent: str,
    runtime: str,
    name: str,
    agent_args: list[str],
) -> list[str]:
    """Flags shared by autonomous and interactive ``start`` invocations.

    Every option the interactive wizard would otherwise prompt for must be
    supplied here — ``--task`` selects autonomous *mode* but does not suppress the
    wizard, and the subprocess has no TTY, so any prompt aborts the run. The
    wizard prompts (in ``cli.start``) for container, agent, tools, **shared**,
    name, and **auto-log**; all are provided below (agent-specific required args
    come via ``agent_args``).

    ``--auto-log`` is on so each cell also exercises the logging/report/export
    chain: tool calls are recorded to ``<shared>/auto_log/`` (see
    :func:`auto_log_dir`), which the report/.eln assertions then build from.

    ``--no-config`` is essential: the repo ships an example config that
    auto-discovers in the CWD, and we must not let its defaults leak in.
    """
    shared = _shared_dir(name)
    shared.mkdir(parents=True, exist_ok=True)
    argv = [
        "start",
        "--no-config",
        "--agent",
        agent,
        "--container",
        runtime,
        "--name",
        name,
        "--tools",
        str(TOOLS_FILE),
        "--shared",
        str(shared),
        "--auto-log",
    ]
    for a in agent_args:
        argv += ["--agent-args", a]
    return argv


# ----------------------------------------------------------------------
# Autonomous launch (non-interactive)
# ----------------------------------------------------------------------
def run_autonomous(
    agent: str,
    runtime: str,
    name: str,
    task: str,
    agent_args: list[str],
    timeout: float,
) -> subprocess.CompletedProcess:
    """Run an autonomous ``start --task`` session to completion.

    Blocks until the container self-exits and the CLI returns, or raises on
    timeout after killing the process group.
    """
    argv = cli_argv(
        *_common_start_args(agent, runtime, name, agent_args), "--task", task
    )
    # Force the child's stdio to UTF-8. We capture output via a pipe, and on
    # Windows a piped stdout defaults to the legacy cp1252 codec — the CLI prints
    # Unicode (the ▶ session marker, box-drawing banners, …), so the first such
    # write would raise UnicodeEncodeError and, in autonomous mode, kill the run
    # before the agent turn completes. PYTHONIOENCODING makes the child encode
    # UTF-8; the matching encoding= below decodes it. A real terminal (a user
    # running `agent start` directly) is never cp1252, so this only matters when
    # output is piped, as here — hence it lives in the harness, not the CLI.
    child_env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            env=child_env,
            # Decode as UTF-8 explicitly to match PYTHONIOENCODING above;
            # subprocess's text mode would otherwise decode with the locale codec
            # (cp1252 on Windows) and raise UnicodeDecodeError in the reader
            # thread, truncating the captured output we rely on for diagnostics.
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
        # Always persist the full autonomous output for post-mortem. pytest
        # discards it whenever the run exits 0, which hides an "empty turn"
        # (the agent produced no transcript / tool call) — the one failure the
        # durable artifacts cannot explain on their own.
        try:
            pty_log_path(name).with_suffix(".autonomous.log").write_text(
                f"rc={proc.returncode}\n--- stdout ---\n{proc.stdout}\n"
                f"--- stderr ---\n{proc.stderr}\n",
                encoding="utf-8",
                errors="replace",
            )
        except OSError:
            pass
        return proc
    except subprocess.TimeoutExpired as exc:  # pragma: no cover - failure path
        out = _as_text(exc.stdout)
        err = _as_text(exc.stderr)
        raise AssertionError(
            f"Autonomous session '{name}' ({agent}/{runtime}) did not finish "
            f"within {timeout}s.\n--- stdout ---\n{out}\n--- stderr ---\n{err}"
        ) from exc


def _as_text(value: object) -> str:
    """Coerce captured subprocess output (str/bytes/None) to text for messages."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value)


# ----------------------------------------------------------------------
# PTY-driven launch (interactive start / resume / converse)
# ----------------------------------------------------------------------
def drive_pty_session(
    argv: list[str],
    *,
    runtime: str,
    session_name: str,
    boot_timeout: float,
    prompt: str | None = None,
    settle_secs: float = 12.0,
    max_turn_secs: float = 240.0,
) -> str:
    """Drive an interactive CLI invocation over a real pseudo-terminal.

    Forks ``argv`` with a controlling PTY, waits until the session container is
    ``running``, optionally types ``prompt`` and lets output settle, then sends
    Ctrl-C so the CLI's SIGINT handler tears down and (re-)commits gracefully.

    Returns the captured terminal output. Raises ``AssertionError`` on boot
    timeout or a non-clean child exit. The pseudo-terminal is provided by
    :mod:`_console` (POSIX ``pty`` or Windows ConPTY), so this logic is
    platform-agnostic.
    """
    console = _console.spawn(argv)
    captured: list[str] = []

    def _drain(deadline: float) -> None:
        """Read available output until *deadline* or the first idle gap."""
        while time.monotonic() < deadline:
            chunk = console.read(0.2)
            if chunk:
                captured.append(chunk)
            else:
                return

    def _read_for(seconds: float) -> None:
        """Read output for a fixed *seconds*, tolerating idle gaps."""
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            chunk = console.read(0.2)
            if chunk:
                captured.append(chunk)

    try:
        # 1. Wait for the container to come up.
        if not _wait_container_running(runtime, session_name, boot_timeout, _drain):
            console.kill()
            raise AssertionError(
                f"Container for session '{session_name}' did not reach 'running' "
                f"within {boot_timeout}s.\n{''.join(captured)[-2000:]}"
            )

        def _visible(tail: int = 0) -> str:
            """ANSI-stripped, whitespace-collapsed view of the capture (or its tail).

            Claude's TUI positions text with cursor-move escapes, so words are
            only contiguous after stripping ANSI and whitespace. A small *tail*
            approximates the current screen, so a dialog that has scrolled off no
            longer matches.
            """
            raw = "".join(captured)
            if tail:
                raw = raw[-tail:]
            return re.sub(r"\s+", "", _ANSI_RE.sub("", raw)).lower()

        # 2. Optionally drive one conversational turn.
        if prompt is not None:
            # Wait for the TUI to be READY for input before typing. Input sent
            # mid-startup is unreliable: Claude swallows it behind a "trust this
            # folder?" dialog; OpenClaw accepts the text into its box but drops the
            # submit Enter. A plain "quiet" heuristic fires too early because
            # OpenClaw's startup prints bursty banners with multi-second gaps, so
            # wait for an explicit readiness signal instead — Claude's trust dialog
            # (accept it), or OpenClaw's status line flipping to "idle". Fall back
            # to a time cap so we still proceed if neither marker appears.
            ready_deadline = time.monotonic() + 45.0
            while time.monotonic() < ready_deadline:
                _read_for(0.5)
                vis = _visible(2500)
                if "trustthisfolder" in vis:  # Claude trust dialog
                    console.write("\r")
                    _read_for(2.0)
                    break
                if "|idle" in vis:  # OpenClaw "local ready | idle" status line
                    _read_for(1.5)
                    break

            # Type the query, then submit with a SEPARATE Enter after a short pause.
            # A combined "text\r" can lose the newline (OpenClaw's box keeps the
            # text but never submits); typing then pausing then Enter is reliable
            # for both agents.
            console.write(prompt)
            _read_for(1.5)
            console.write("\r")

            # Drive the turn. Claude gates each MCP tool call behind a
            # "❯ 1. Yes / 2. Yes, and don't ask again" menu; approve it with Enter
            # as it appears, until the output goes idle (turn complete) or the cap
            # is hit. Detect the menu on a recent tail so a dismissed one does not
            # re-trigger. (OpenClaw shows no such menu — the check is a no-op.)
            deadline = time.monotonic() + max_turn_secs
            last_out = time.monotonic()
            last_answer = 0.0
            while time.monotonic() < deadline:
                before = len(captured)
                _read_for(1.0)
                now = time.monotonic()
                if len(captured) > before:
                    last_out = now
                if "1.yes" in _visible(1500) and now - last_answer > 5.0:
                    console.write("\r")  # approve the highlighted default
                    last_answer = now
                    last_out = now
                    continue
                if now - last_out >= settle_secs:
                    break
        else:
            # Plumbing-only: let the TUI finish booting before we tear down.
            _read_for(min(settle_secs, 8.0))

        # 3. Graceful teardown by stopping the container. A single Ctrl-C into
        # the PTY does NOT end an interactive claude-code session: the entrypoint
        # runs the TUI and then drops to a `bash` shell, so the TUI swallows the
        # ^C and the container keeps running (the real quit sequence is ^C ^C to
        # exit the TUI, then `exit` to leave the shell). Puppeting that through
        # the PTY is brittle, so we instead stop the container directly: that
        # makes the CLI's `docker start -ai` return and unwind into its normal
        # _cleanup → commit path — exactly what a real user's quit-and-exit does,
        # independent of whatever the TUI is showing. Ctrl-C is still sent first
        # as a best-effort graceful signal. Give the commit ample time —
        # committing a multi-GB image is not instant.
        console.interrupt()
        _drain(time.monotonic() + 2.0)
        _runtime_call(runtime, "stop", "-t", "10", container_name(session_name))
        rc = _wait_exit(console, captured, _drain, timeout=180.0)
    finally:
        console.kill()
        # Dump the raw PTY capture for post-mortem (kept by SLA_E2E_KEEP_ON_FAIL;
        # cleanup_session removes it otherwise). Never let this break the run.
        try:
            pty_log_path(session_name).write_text(
                "".join(captured), encoding="utf-8", errors="replace"
            )
        except OSError:
            pass

    if rc not in (0, None):
        raise AssertionError(
            f"Interactive session '{session_name}' exited with {rc}.\n"
            f"{''.join(captured)[-2000:]}"
        )
    return "".join(captured)


def _wait_exit(console, captured, drain, timeout: float):
    """Wait up to ``timeout`` for the child to exit, draining output meanwhile."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        drain(time.monotonic() + 0.5)
        rc = console.poll()
        if rc is not None:
            return rc
    return None  # still running; caller kills it


def _wait_container_running(runtime, session_name, timeout, drain) -> bool:
    """Poll the runtime until the session's container is running.

    The runtime names the container ``safe-lab-agents-<session>`` (see
    :func:`container_name`), so we match on that full name, not the bare session.
    """
    cname = container_name(session_name)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        out = _runtime_output(
            runtime, "ps", "--filter", f"name={cname}", "--format", "{{.Names}}"
        )
        # Podman may qualify names (e.g. "localhost/…"); match by suffix to be safe.
        if any(line == cname or line.endswith("/" + cname) for line in out.split()):
            return True
        drain(time.monotonic() + 1.0)  # keep the PTY buffer clear while we wait
    return False


# ----------------------------------------------------------------------
# High-level drivers
# ----------------------------------------------------------------------
def drive_interactive_start(
    agent: str,
    runtime: str,
    name: str,
    agent_args: list[str],
    boot_timeout: float,
) -> str:
    """Launch an interactive ``start`` (no ``--task``) and drive one ping turn."""
    argv = cli_argv(*_common_start_args(agent, runtime, name, agent_args))
    return drive_pty_session(
        argv,
        runtime=runtime,
        session_name=name,
        boot_timeout=boot_timeout,
        prompt="Call the `ping` MCP tool exactly once, then say DONE.",
    )


def drive_resume(
    name: str,
    runtime: str,
    boot_timeout: float,
    agent_args: list[str],
    converse: bool = False,
) -> str:
    """Resume a committed session. Boots + graceful Ctrl-C; optionally converses.

    Secrets are scrubbed from the committed image, so a resumed agent must
    re-obtain its credentials — OpenClaw otherwise prompts "Re-enter the LLM
    provider API key … to resume" and blocks. ``agent resume`` merges
    ``--agent-args`` into the stored config before ``resume_credential_env`` runs,
    so re-passing the same credential args keeps resume non-interactive.
    """
    argv = ["resume", "--name", name]
    for a in agent_args:
        argv += ["--agent-args", a]
    prompt = "Call the `ping` MCP tool once more, then say DONE." if converse else None
    return drive_pty_session(
        cli_argv(*argv),
        runtime=runtime,
        session_name=name,
        boot_timeout=boot_timeout,
        prompt=prompt,
    )


# ----------------------------------------------------------------------
# Assertions on durable artifacts
# ----------------------------------------------------------------------
def load_status(name: str) -> str:
    """Return the ``status`` recorded in the session's metadata.json."""
    meta = json.loads((get_sessions_dir() / name / "metadata.json").read_text())
    return meta.get("status", "")


def assert_status(name: str, expected: str) -> None:
    actual = load_status(name)
    assert actual == expected, f"session '{name}': status {actual!r} != {expected!r}"


def load_history(name: str) -> list[dict]:
    """Load and parse the session's history.json (empty list if absent)."""
    path = get_sessions_dir() / name / "history.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    return data if isinstance(data, list) else data.get("entries", [])


def count_ping_calls(name: str, sentinel: str = SENTINEL) -> int:
    """Count history entries that carry a ``ping`` tool result with the sentinel."""
    n = 0
    for entry in load_history(name):
        blob = json.dumps(entry, default=str)
        if entry.get("tool_name") == "ping" and sentinel in blob:
            n += 1
        elif sentinel in blob:  # some parsers fold output into content
            n += 1
    return n


def history_debug(name: str, sentinel: str = SENTINEL) -> str:
    """Summarise history.json for failure messages (roles, tool names, sentinel).

    Read before cleanup so a failing assertion can explain *why* no ping was
    found — e.g. the transcript is empty, the tool was never invoked, or the
    sentinel is present but under an unexpected field.
    """
    path = get_sessions_dir() / name / "history.json"
    if not path.exists():
        return f"history.json missing at {path}"
    entries = load_history(name)
    if not entries:
        return f"history.json present but 0 entries ({path.stat().st_size} bytes)"
    roles: dict[str, int] = {}
    tools: dict[str, int] = {}
    sentinel_here = False
    for e in entries:
        roles[e.get("role", "?")] = roles.get(e.get("role", "?"), 0) + 1
        tn = e.get("tool_name")
        if tn:
            tools[tn] = tools.get(tn, 0) + 1
        if sentinel in json.dumps(e, default=str):
            sentinel_here = True
    return (
        f"history.json: {len(entries)} entries; roles={roles}; "
        f"tool_names={tools or '{}'}; sentinel_present={sentinel_here}"
    )


def assert_tool_called(name: str, minimum: int = 1, sentinel: str = SENTINEL) -> None:
    calls = count_ping_calls(name, sentinel)
    assert calls >= minimum, (
        f"session '{name}': expected >= {minimum} ping/sentinel occurrence(s) in "
        f"history.json, found {calls}"
    )


def session_image_tag(name: str) -> str:
    return f"{_SESSION_IMAGE_PREFIX}{name}:latest"


def assert_committed_image(runtime: str, name: str) -> None:
    tag = session_image_tag(name)
    rc = _runtime_call(runtime, "image", "inspect", tag)
    assert rc == 0, f"no committed {runtime} image '{tag}' for session '{name}'"


# ----------------------------------------------------------------------
# Auto-log / report / .eln / history-HTML (end-to-end output chain)
# ----------------------------------------------------------------------
def assert_autolog_records(name: str, sentinel: str = SENTINEL) -> None:
    """Assert the auto_log/ folder holds a record carrying the ping sentinel.

    Exercises the host-side auto-logging chain: the MCP wrapper records each tool
    call as a JSON record under ``<shared>/auto_log/`` (arrays additionally to
    HDF5). Our ping returns a plain dict, so the sentinel lands in the JSON.
    """
    d = auto_log_dir(name)
    assert d.is_dir(), f"auto_log dir missing at {d}"
    records = [p for p in d.glob("*.json") if p.name != "session_summary.json"]
    assert records, f"no auto_log records in {d}"
    blob = "".join(p.read_text(errors="replace") for p in records)
    assert (
        sentinel in blob
    ), f"sentinel {sentinel!r} not in {len(records)} auto_log record(s) at {d}"


def build_and_check_report(name: str, sentinel: str = SENTINEL) -> Path:
    """Build the auto-log HTML report (``agent report``) and check it renders."""
    from safe_lab_agents.report import build_report

    _artifact_dir(name).mkdir(parents=True, exist_ok=True)
    out = _artifact_dir(name) / "report.html"
    build_report(auto_log_dir(name), out)
    assert out.is_file() and out.stat().st_size > 0, f"report not written at {out}"
    html = out.read_text(errors="replace")
    assert "ping" in html or sentinel in html, "auto-log report HTML missing ping"
    return out


def build_and_check_eln(name: str, sentinel: str = SENTINEL) -> Path:
    """Build the ``.eln`` export (``agent export-eln``) and validate the RO-Crate."""
    import zipfile

    from safe_lab_agents.export import build_eln

    _artifact_dir(name).mkdir(parents=True, exist_ok=True)
    out = _artifact_dir(name) / "export.eln"
    build_eln(auto_log_dir(name), out)
    assert zipfile.is_zipfile(out), f".eln is not a valid zip: {out}"
    with zipfile.ZipFile(out) as z:
        names = z.namelist()
        meta = next((n for n in names if n.endswith("ro-crate-metadata.json")), None)
        assert meta, f".eln has no ro-crate-metadata.json (members: {names[:5]})"
        crate = z.read(meta).decode("utf-8", "replace")
    assert "ping" in crate or sentinel in crate, ".eln RO-Crate missing ping/sentinel"
    return out


def build_and_check_history_html(name: str) -> Path:
    """Build the conversation HTML (``agent history --html``) and check it renders."""
    from safe_lab_agents.config import SessionMetadata
    from safe_lab_agents.history.html import build_conversation_html
    from safe_lab_agents.history.store import HistoryStore

    _artifact_dir(name).mkdir(parents=True, exist_ok=True)
    out = _artifact_dir(name) / "history.html"
    try:
        metadata = SessionMetadata.load(name)
    except FileNotFoundError:
        metadata = None
    entries = HistoryStore(name).load_history()
    build_conversation_html(entries, metadata, out)
    assert (
        out.is_file() and out.stat().st_size > 0
    ), f"history HTML not written at {out}"
    assert "ping" in out.read_text(errors="replace"), "history HTML missing ping"
    return out


# ----------------------------------------------------------------------
# Cleanup + runtime helpers
# ----------------------------------------------------------------------
def cleanup_session(name: str, runtime: str) -> None:
    """Best-effort removal of the session dir, container, and committed image."""
    import shutil

    _runtime_call(runtime, "rm", "-f", container_name(name))
    _runtime_call(runtime, "rmi", "-f", session_image_tag(name))
    shutil.rmtree(get_sessions_dir() / name, ignore_errors=True)
    shutil.rmtree(_shared_dir(name), ignore_errors=True)
    shutil.rmtree(_artifact_dir(name), ignore_errors=True)
    pty_log_path(name).unlink(missing_ok=True)


def runtime_available(runtime: str) -> bool:
    """True if the runtime binary exists and its daemon answers ``info``."""
    import shutil

    if shutil.which(runtime) is None:
        return False
    return _runtime_call(runtime, "info") == 0


def _runtime_call(runtime: str, *args: str) -> int:
    try:
        return subprocess.run(
            [runtime, *args],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=60,
        ).returncode
    except (OSError, subprocess.TimeoutExpired):
        return 1


def _runtime_output(runtime: str, *args: str) -> str:
    try:
        return subprocess.run(
            [runtime, *args],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""
