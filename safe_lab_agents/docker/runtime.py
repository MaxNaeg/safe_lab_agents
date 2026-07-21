"""Container runtime detection and setup for safe_lab_agents.

Provides helpers for configuring alternate container runtimes (e.g. Podman)
by setting the DOCKER_HOST environment variable before Docker SDK or CLI
operations are performed.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import re
import shutil
import socket as _socket
import subprocess
import time
from pathlib import Path

from rich.console import Console

logger = logging.getLogger(__name__)


# All user-facing status/warning output in this module goes through this one
# stderr Console (never bare print()), matching cli.py. markup/highlight are
# off so an interpolated path or exception string can't be misparsed.
console = Console(stderr=True)


def _status(message: str) -> None:
    """Print a plain status/warning line to stderr via the shared Console."""
    console.print(message, markup=False, highlight=False)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Env var recording which CLI binary to shell out to for container operations.
# The Docker SDK talks to whatever DOCKER_HOST points at, but our direct
# subprocess calls (build, cp, start) need the matching binary on PATH — and a
# Podman-only host (notably Windows) has no ``docker`` binary at all. This is
# set process-wide by setup_podman_host(), mirroring how DOCKER_HOST is set.
_CONTAINER_CLI_ENV = "SAFE_LAB_AGENTS_CONTAINER_CLI"


def container_cli() -> str:
    """Return the container CLI binary to invoke ('docker' or 'podman').

    Defaults to ``docker``; :func:`setup_podman_host` switches it to
    ``podman`` for the lifetime of the process when Podman is selected.
    """
    return os.environ.get(_CONTAINER_CLI_ENV, "docker")


def is_runtime_installed(runtime: str) -> bool:
    """Best-effort check whether the given container runtime is installed.

    Looks for the CLI binary on PATH; on Windows, where the installer does not
    always add the binary to PATH, also searches the common install locations
    (the same ones the runtime-setup helpers fall back to). This reports
    *installation*, not whether the daemon/machine is currently running.
    """
    binary = "podman" if runtime == "podman" else "docker"
    if shutil.which(binary):
        return True
    if platform.system() == "Windows":
        if runtime == "podman":
            return _find_podman_windows() is not None
        return find_docker_desktop_windows() is not None
    return False


def setup_podman_host() -> str:
    """Ensure Podman is initialized and running, then point DOCKER_HOST at it.

    On macOS and Windows, manages a rootful Podman machine (initializing and
    starting it automatically if needed) and sets the VM memory to half the
    host's physical RAM when it is below that target.  On Linux, activates
    the Podman socket via systemd if it is not already present (no VM tuning
    needed — Podman runs natively).

    Returns:
        The ``DOCKER_HOST`` URL that was set (e.g. ``unix:///path/to/podman.sock``).

    Raises:
        RuntimeError: If Podman is not installed, cannot be started, or the
            socket cannot be found.
    """
    # Route subsequent CLI subprocess calls (build/cp/start) to podman too,
    # since a Podman-only host may not have a ``docker`` binary on PATH.
    os.environ[_CONTAINER_CLI_ENV] = "podman"

    system = platform.system()
    if system == "Darwin":
        return _setup_podman_macos()
    elif system == "Linux":
        return _setup_podman_linux()
    elif system == "Windows":
        return _setup_podman_windows()
    else:
        raise RuntimeError(f"Podman support is not implemented for platform: {system}")


def configure_docker_desktop_memory() -> None:
    """Try to ensure Docker Desktop's VM has at least half the host RAM.

    Reads Docker Desktop's ``settings.json`` and updates ``memoryMiB`` when
    the current value is below the target.  Prints a restart reminder if a
    change was made.  Silently no-ops on Linux (native daemon, no VM) or if
    the Docker Desktop settings file cannot be found or parsed.
    """
    system = platform.system()
    if system == "Darwin":
        settings_path = (
            Path.home()
            / "Library"
            / "Group Containers"
            / "group.com.docker"
            / "settings.json"
        )
    elif system == "Windows":
        appdata = os.environ.get("APPDATA", "")
        if not appdata:
            return
        settings_path = Path(appdata) / "Docker" / "settings.json"
    else:
        return  # Linux: native daemon, no VM memory to configure.

    if not settings_path.exists():
        return

    try:
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return

    target_mb = _get_half_host_ram_mb()
    current_mb = settings.get("memoryMiB", 0)

    if current_mb >= target_mb:
        return

    settings["memoryMiB"] = target_mb
    try:
        settings_path.write_text(json.dumps(settings, indent=2), encoding="utf-8")
        _status(
            f"Docker Desktop memory updated: {current_mb} MB → {target_mb} MB "
            f"(half of host RAM). Restart Docker Desktop to apply.",
        )
    except OSError as exc:
        _status(f"Warning: could not update Docker Desktop settings: {exc}")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _get_half_host_ram_mb() -> int:
    """Return half the host's physical RAM in MB, rounded to the nearest 512 MB."""
    system = platform.system()
    try:
        if system == "Darwin":
            out = subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"], stderr=subprocess.DEVNULL, text=True
            )
            total_mb = int(out.strip()) // (1024 * 1024)
        elif system == "Linux":
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemTotal:"):
                    total_mb = int(line.split()[1]) // 1024  # kB → MB
                    break
            else:
                return 4096
        elif system == "Windows":
            win_mb = _windows_total_ram_mb()
            if win_mb is None:
                return 4096
            total_mb = win_mb
        else:
            return 4096
    except Exception:
        # Detection failing shouldn't abort the session — fall back to a
        # conservative 4 GB — but log so a broken probe is diagnosable rather
        # than silently capping every container.
        logger.debug(
            "Could not determine host RAM; using 4096 MB fallback", exc_info=True
        )
        return 4096

    half = total_mb // 2
    return max((half // 512) * 512, 1024)  # round to 512 MB boundary, minimum 1 GB


def _windows_total_ram_mb() -> int | None:
    """Total physical RAM in MB on Windows, or ``None`` if undeterminable.

    Prefers PowerShell's CIM query: ``wmic`` was removed in Windows 11 24H2, so
    the old ``wmic computersystem get TotalPhysicalMemory`` raises
    ``FileNotFoundError`` there and silently fell back to a 4 GB guess (which
    then halves the container to a needlessly tiny 2 GB). ``wmic`` is kept only
    as a fallback for older Windows that lack the CIM cmdlets.
    """
    try:
        out = subprocess.check_output(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        if out:
            return int(out) // (1024 * 1024)
    except Exception:
        logger.debug("PowerShell CIM RAM query failed; trying wmic", exc_info=True)
    try:
        out = subprocess.check_output(
            ["wmic", "computersystem", "get", "TotalPhysicalMemory", "/value"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        for line in out.splitlines():
            if line.startswith("TotalPhysicalMemory="):
                return int(line.split("=")[1].strip()) // (1024 * 1024)
    except Exception:
        logger.debug("wmic RAM query failed; RAM undeterminable", exc_info=True)
    return None


def _get_podman_machine_memory_mb() -> int:
    """Return the current memory allocation of the default Podman machine in MB."""
    try:
        out = subprocess.check_output(
            ["podman", "machine", "inspect"], stderr=subprocess.DEVNULL
        )
        data = json.loads(out)
        return data[0].get("Resources", {}).get("Memory", 0)
    except Exception:
        # 0 signals "unknown" to callers, which then skip memory tuning; log so
        # an inspect/parse failure is visible rather than silently skipped.
        logger.debug("Could not read Podman machine memory", exc_info=True)
        return 0


def _apply_podman_machine_memory(target_mb: int) -> None:
    """Set the Podman machine memory to *target_mb* MB (non-fatal on failure)."""
    try:
        subprocess.run(
            ["podman", "machine", "set", "--memory", str(target_mb)], check=True
        )
    except subprocess.CalledProcessError as exc:
        _status(f"Warning: could not update Podman machine memory: {exc}")


_WINDOWS_PODMAN_SEARCH_ROOTS = [
    Path(os.environ.get("LOCALAPPDATA", "")) / "Programs",
    Path(os.environ.get("PROGRAMFILES", r"C:\Program Files")),
    Path(os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)")),
]


def _find_podman_windows() -> Path | None:
    """Search common Windows install trees for the podman.exe binary."""
    for root in _WINDOWS_PODMAN_SEARCH_ROOTS:
        if not root.exists():
            continue
        for candidate in root.rglob("podman.exe"):
            return candidate
    return None


def find_docker_desktop_windows() -> Path | None:
    """Locate the ``Docker Desktop.exe`` launcher on Windows.

    Checks the system-wide install locations and the per-user install under
    ``%LOCALAPPDATA%`` (Docker Desktop 4.x installs there by default), then
    falls back to a search of the common program trees. Returns ``None`` if it
    cannot be found.
    """
    localappdata = os.environ.get("LOCALAPPDATA", "")
    program_files = os.environ.get("PROGRAMFILES", r"C:\Program Files")
    program_files_x86 = os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)")
    candidates = [
        Path(program_files) / "Docker" / "Docker" / "Docker Desktop.exe",
        Path(program_files_x86) / "Docker" / "Docker" / "Docker Desktop.exe",
        Path(localappdata) / "Programs" / "DockerDesktop" / "Docker Desktop.exe",
        Path(localappdata) / "Docker" / "Docker Desktop.exe",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    search_roots = [Path(localappdata) / "Programs", Path(program_files)]
    for root in search_roots:
        if not root.exists():
            continue
        for found in root.rglob("Docker Desktop.exe"):
            return found
    return None


def _require_podman() -> None:
    """Raise RuntimeError if the ``podman`` binary is not available.

    On Windows, also searches common install locations and patches PATH so
    that subsequent subprocess calls find podman even if the installer did
    not add it to the user's PATH.  If the automatic search fails, the user
    is prompted to enter the path manually.
    """
    try:
        subprocess.run(["podman", "--version"], check=True, capture_output=True)
        return
    except FileNotFoundError:
        pass
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Podman is not installed or not on PATH. "
            "Install it from https://podman.io/getting-started/installation"
        ) from exc

    if platform.system() == "Windows":
        podman_exe = _find_podman_windows()
        if podman_exe is None:
            _status(
                "\nPodman was not found on PATH and could not be located automatically.\n"
                "Please enter the full path to podman.exe.\n"
                "Tip: open File Explorer and search for 'podman.exe', or look inside\n"
                "  %LOCALAPPDATA%\\Programs\\  (e.g. ...\\RedHat\\Podman\\podman.exe)\n"
                "  C:\\Program Files\\RedHat\\Podman\\podman.exe\n"
            )
            user_input = input("Path to podman.exe: ").strip().strip('"')
            podman_exe = Path(user_input)
            if not podman_exe.exists():
                raise RuntimeError(f"podman.exe not found at: {podman_exe}")
        os.environ["PATH"] = str(podman_exe.parent) + os.pathsep + os.environ.get("PATH", "")
        return

    raise RuntimeError(
        "Podman is not installed or not on PATH. "
        "Install it from https://podman.io/getting-started/installation"
    )


def _configure_podman_machine_memory(target_mb: int, running: bool) -> None:
    """Check and (if needed) adjust Podman machine memory.

    When the machine is not running, applies the change immediately.
    When it is already running, only prints a warning to avoid disrupting
    containers that may be active.
    """
    current_mb = _get_podman_machine_memory_mb()
    if current_mb <= 0 or current_mb >= target_mb:
        return

    if running:
        _status(
            f"Warning: Podman machine has only {current_mb} MB RAM "
            f"(recommended: {target_mb} MB — half of host RAM). "
            "Stop the machine and re-run to apply the update.",
        )
    else:
        _status(
            f"Podman machine has {current_mb} MB RAM — "
            f"adjusting to {target_mb} MB (half of host RAM) …",
        )
        _apply_podman_machine_memory(target_mb)


# ---------------------------------------------------------------------------
# macOS
# ---------------------------------------------------------------------------

# Known-noise patterns in `podman machine start` output. None of them applies
# here: we set DOCKER_HOST ourselves (helper-service / API-forwarding notes),
# the rootless hint is irrelevant to our container use, and the progress lines
# duplicate our own "Starting Podman machine …" message.
_PODMAN_START_NOISE = [
    # Advisory block: install the system helper so the default Docker socket
    # routes to Podman, through the suggested `export DOCKER_HOST=` command.
    re.compile(
        r"The system helper service is not installed;.*?"
        r"export DOCKER_HOST=.*?\n\s*?\n",
        re.DOTALL,
    ),
    # Advisory block: machine is rootless, switch via `podman machine set --rootful`.
    re.compile(
        r"This machine is currently configured in rootless mode\..*?"
        r"podman machine set --rootful\s*\n",
        re.DOTALL,
    ),
    re.compile(r'^Starting machine ".*"[ \t]*\n?', re.MULTILINE),
    re.compile(r"^API forwarding listening on: .*\n?", re.MULTILINE),
    re.compile(r"^Docker API clients default to this address\. ?.*\n?", re.MULTILINE),
    re.compile(r"^You do not need to set DOCKER_HOST\..*\n?", re.MULTILINE),
]


def _start_podman_machine() -> None:
    """Run ``podman machine start``, hiding known-noise output.

    ``podman machine start`` prints several advisories and progress lines that
    don't apply here (see ``_PODMAN_START_NOISE``) — strip them while still
    passing through everything else (and any real failure).
    """
    result = subprocess.run(
        ["podman", "machine", "start"],
        check=False,
        capture_output=True,
        text=True,
    )
    output = (result.stdout or "") + (result.stderr or "")
    for pattern in _PODMAN_START_NOISE:
        output = pattern.sub("", output)
    cleaned = output.strip()
    if cleaned:
        _status(cleaned)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, ["podman", "machine", "start"]
        )


def _setup_podman_macos() -> str:
    """Initialize/start a rootful Podman machine on macOS and set DOCKER_HOST."""
    _require_podman()

    target_mb = _get_half_host_ram_mb()

    try:
        out = subprocess.check_output(
            ["podman", "machine", "list", "--format", "json"],
            stderr=subprocess.DEVNULL,
        )
        machines = json.loads(out) if out.strip() else []
    except (subprocess.CalledProcessError, json.JSONDecodeError):
        machines = []

    if not machines:
        _status(
            f"No Podman machine found — initializing with {target_mb} MB RAM "
            f"(half of host RAM) …",
        )
        subprocess.run(
            ["podman", "machine", "init", "--rootful", "--memory", str(target_mb)],
            check=True,
        )
        _status("Starting Podman machine …")
        _start_podman_machine()
    else:
        machine = machines[0]
        running = machine.get("Running", False)
        if not running:
            _configure_podman_machine_memory(target_mb, running=False)
            _status("Starting Podman machine …")
            _start_podman_machine()
        else:
            _configure_podman_machine_memory(target_mb, running=True)

    socket_path = _get_podman_macos_socket()
    host_url = f"unix://{socket_path}"
    os.environ["DOCKER_HOST"] = host_url
    return host_url


def _get_podman_macos_socket() -> str:
    """Return the path to the Podman socket on macOS."""
    try:
        out = subprocess.check_output(
            ["podman", "machine", "inspect", "--format", "{{.ConnectionInfo.PodmanSocket.Path}}"],
            stderr=subprocess.PIPE,
        )
        path = out.decode().strip()
        if path:
            return path
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Could not determine Podman socket path: {exc.stderr.decode().strip()}"
        ) from exc
    raise RuntimeError(
        "Podman machine inspect returned an empty socket path. "
        "Try running 'podman machine inspect' manually to diagnose."
    )


# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------

def _setup_podman_windows() -> str:
    """Initialize/start a rootful Podman machine on Windows and set DOCKER_HOST."""
    _require_podman()

    target_mb = _get_half_host_ram_mb()

    try:
        out = subprocess.check_output(
            ["podman", "machine", "list", "--format", "json"],
            stderr=subprocess.DEVNULL,
        )
        machines = json.loads(out) if out.strip() else []
    except (subprocess.CalledProcessError, json.JSONDecodeError):
        machines = []

    if not machines:
        _status(
            f"No Podman machine found — initializing with {target_mb} MB RAM "
            f"(half of host RAM) …",
        )
        subprocess.run(
            ["podman", "machine", "init", "--rootful", "--memory", str(target_mb)],
            check=True,
        )
        _status("Starting Podman machine …")
        _start_podman_machine()
    else:
        machine = machines[0]
        running = machine.get("Running", False)
        if not running:
            _configure_podman_machine_memory(target_mb, running=False)
            _status("Starting Podman machine …")
            _start_podman_machine()
        else:
            _configure_podman_machine_memory(target_mb, running=True)

    # On Windows, Podman uses a named pipe; construct the DOCKER_HOST URL from
    # the machine's connection info.
    host_url = _get_podman_windows_host()
    os.environ["DOCKER_HOST"] = host_url
    return host_url


def _get_podman_windows_host() -> str:
    """Return the DOCKER_HOST URL for the Podman machine on Windows."""
    try:
        out = subprocess.check_output(
            ["podman", "machine", "inspect", "--format", "{{.ConnectionInfo.PodmanPipe.Path}}"],
            stderr=subprocess.PIPE,
        )
        pipe_path = out.decode().strip()
        if pipe_path:
            return _npipe_url(pipe_path)
    except subprocess.CalledProcessError:
        pass

    # Fallback: use the default named pipe path.
    return "npipe:////./pipe/podman-machine-default"


def _npipe_url(pipe_path: str) -> str:
    r"""Convert a Windows named-pipe path into a docker-py ``npipe://`` URL.

    Podman reports the pipe in native Windows form (``\\.\pipe\name``), but
    docker-py's ``parse_host`` runs the URL through ``urlparse`` and keeps only
    the *path* component.  With backslashes there is no ``/`` after the scheme,
    so the whole pipe name is swallowed into the (discarded) netloc — the URL
    collapses to ``npipe:`` and the connection fails.  Converting the
    separators to forward slashes (``//./pipe/name``) makes the value land in
    the URL path, matching the canonical ``npipe:////./pipe/docker_engine``
    form docker-py expects.
    """
    pipe_path = pipe_path.replace("npipe://", "")  # strip a pre-existing scheme
    pipe_path = pipe_path.replace("\\", "/")
    return f"npipe://{pipe_path}"


# ---------------------------------------------------------------------------
# Windows / WSL host reachability (Podman)
# ---------------------------------------------------------------------------

# Display name of the inbound firewall rule that lets the Podman container
# reach the host-side MCP server over the WSL virtual adapter.
MCP_FIREWALL_RULE_NAME = "safe-lab-agents-mcp"


def _parse_default_gateway(ip_route_output: str) -> str | None:
    """Extract the gateway IP from ``ip route show default`` output.

    Example line: ``default via 172.26.80.1 dev eth0 proto kernel`` → ``172.26.80.1``.
    """
    for line in ip_route_output.splitlines():
        tokens = line.split()
        if "via" in tokens:
            idx = tokens.index("via")
            if idx + 1 < len(tokens):
                return tokens[idx + 1]
    return None


def podman_windows_gateway_ip() -> str | None:
    """Return the WSL default-gateway IP that reaches the Windows host.

    Podman on Windows runs inside a WSL2 VM behind NAT, so the host is reached
    via the VM's default gateway (a dynamic ``172.x.x.1`` address), not
    ``host.docker.internal`` (which points at the Podman bridge gateway inside
    the VM). Resolves it by querying the machine's routing table. Returns
    ``None`` if it cannot be determined.
    """
    try:
        out = subprocess.check_output(
            ["podman", "machine", "ssh", "ip route show default"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=30,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return None
    return _parse_default_gateway(out)


def _powershell(command: str, timeout: int = 30) -> str | None:
    """Run a PowerShell command and return stripped stdout, or None on failure."""
    try:
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", command],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=timeout,
        )
        return out.strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return None


def windows_firewall_rule_exists(name: str = MCP_FIREWALL_RULE_NAME) -> bool:
    """Return True if an inbound firewall rule with *name* already exists."""
    result = _powershell(
        f"if (Get-NetFirewallRule -DisplayName '{name}' -ErrorAction SilentlyContinue) "
        "{ 'YES' } else { 'NO' }"
    )
    return result == "YES"


def windows_firewall_rule_status(
    gateway_ip: str | None, name: str = MCP_FIREWALL_RULE_NAME
) -> str:
    """Classify the inbound MCP firewall rule as ``ok``, ``stale``, or ``missing``.

    A rule whose name merely *exists* is not enough: WSL/Hyper-V can tear down
    and recreate the virtual adapter on a reboot or platform update, leaving the
    rule pinned to an interface that no longer matches the live WSL adapter (its
    stored ``InterfaceAlias`` becomes a dead GUID). Such a rule is enabled and
    well-formed but matches no traffic, so inbound packets are silently dropped
    and tool calls time out — the failure this function exists to catch.

    Returns:
        ``'ok'``      – an enabled rule scoped to the adapter that currently
                        carries *gateway_ip* (or scoped to ``Any``).
        ``'stale'``   – a rule exists but is disabled or bound to an interface
                        that no longer matches the live WSL adapter.
        ``'missing'`` – no rule by that name exists (or its state cannot be read).
    """
    enabled = _powershell(
        f"$r = Get-NetFirewallRule -DisplayName '{name}' -ErrorAction SilentlyContinue; "
        "if ($r) { [string]$r.Enabled } else { 'MISSING' }"
    )
    if not enabled or enabled.strip() == "MISSING":
        return "missing"
    if enabled.strip().lower() not in ("true", "1", "enabled"):
        return "stale"

    rule_alias = _powershell(
        f"(Get-NetFirewallRule -DisplayName '{name}' -ErrorAction SilentlyContinue | "
        "Get-NetFirewallInterfaceFilter).InterfaceAlias"
    )
    if rule_alias is None:
        return "stale"
    rule_alias = rule_alias.strip()
    if rule_alias.lower() == "any":
        return "ok"

    current_alias = wsl_interface_alias(gateway_ip)
    if current_alias and rule_alias == current_alias:
        return "ok"
    return "stale"


def wsl_interface_alias(gateway_ip: str | None) -> str | None:
    """Return the Windows adapter alias for the WSL gateway IP (e.g. 'vEthernet (WSL)').

    The WSL default gateway as seen from the VM is the Windows-side WSL adapter
    address, so looking it up with ``Get-NetIPAddress`` yields that adapter's
    alias. The alias name varies across Windows builds, so we discover it rather
    than assume it. Returns ``None`` if it cannot be determined.
    """
    if not gateway_ip:
        return None
    alias = _powershell(
        f"(Get-NetIPAddress -IPAddress {gateway_ip} -ErrorAction SilentlyContinue)"
        ".InterfaceAlias"
    )
    return alias or None


def mcp_firewall_setup_command(
    interface_alias: str | None, recreate: bool = False
) -> str:
    """Build the one-time PowerShell command that opens the MCP path for WSL only.

    Scoping the rule to the WSL virtual adapter means LAN traffic (which arrives
    on the physical NIC) never matches it, so the unauthenticated MCP server is
    not exposed to the network — only the local container can reach it.

    When *recreate* is True, the command first removes any existing rule of the
    same name: a stale rule (pinned to an adapter that no longer exists) must be
    replaced, and ``New-NetFirewallRule`` would otherwise create a duplicate
    sharing the DisplayName rather than re-binding the existing one.
    """
    alias = interface_alias or "vEthernet (WSL)"
    new_rule = (
        f"New-NetFirewallRule -DisplayName '{MCP_FIREWALL_RULE_NAME}' "
        "-Direction Inbound -Action Allow -Protocol TCP "
        f"-InterfaceAlias '{alias}'"
    )
    if recreate:
        return (
            f"Remove-NetFirewallRule -DisplayName '{MCP_FIREWALL_RULE_NAME}' "
            f"-ErrorAction SilentlyContinue; {new_rule}"
        )
    return new_rule


# ---------------------------------------------------------------------------
# Linux
# ---------------------------------------------------------------------------

def _linux_socket_candidates() -> list[Path]:
    candidates = [Path("/run/podman/podman.sock")]
    if hasattr(os, "getuid"):
        candidates.insert(0, Path("/run/user") / str(os.getuid()) / "podman" / "podman.sock")
    return candidates


def _setup_podman_linux() -> str:
    """Activate the Podman socket on Linux and set DOCKER_HOST."""
    _require_podman()

    # Fast path — socket exists and is responsive.
    socket_path = _find_podman_linux_socket()
    if socket_path:
        host_url = f"unix://{socket_path}"
        os.environ["DOCKER_HOST"] = host_url
        return host_url

    # Try activating via systemd (rootless first, then rootful).
    _status(
        "Podman socket not found or not responsive — attempting to activate via systemd …"
    )
    _try_systemd_start(rootful=False)
    socket_path = _find_podman_linux_socket()

    if not socket_path:
        _try_systemd_start(rootful=True)
        socket_path = _find_podman_linux_socket()

    if not socket_path:
        raise RuntimeError(
            "Podman socket not found or not responsive after attempting systemd activation.\n"
            "Try running 'systemctl --user start podman.socket' or "
            "'sudo systemctl start podman.socket' manually."
        )

    host_url = f"unix://{socket_path}"
    os.environ["DOCKER_HOST"] = host_url
    return host_url


def _find_podman_linux_socket() -> str | None:
    """Return the path of a Podman socket that exists and is responsive, or None."""
    for candidate in _linux_socket_candidates():
        try:
            exists = candidate.exists()
        except OSError:
            # e.g. the rootful socket lives in a root-owned dir (/run/podman,
            # mode 0700) that a non-root user can't stat — skip this candidate.
            continue
        if exists and _unix_socket_responsive(str(candidate)):
            return str(candidate)
    return None


def _unix_socket_responsive(path: str) -> bool:
    """Return True if something is actively listening on the Unix domain socket."""
    try:
        sock = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
        sock.settimeout(3)
        sock.connect(path)
        sock.close()
        return True
    except OSError:
        return False


def _try_systemd_start(rootful: bool) -> None:
    """Attempt to start the Podman socket unit via systemctl."""
    cmd = (
        ["sudo", "systemctl", "start", "podman.socket"]
        if rootful
        else ["systemctl", "--user", "start", "podman.socket"]
    )
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=30)
        time.sleep(1)  # Give the socket a moment to appear.
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        pass  # systemd not available or unit not found — try next option.
