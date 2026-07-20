"""Command-line interface for safe_lab_agents.

Provides the ``agent`` command with subcommands for starting,
resuming, listing, and viewing history of agent sessions.

Running ``agent start`` with no arguments launches an interactive
wizard that prompts for each setting.  Power users can pass all flags on the
command line to skip prompts.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import re
import secrets
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional, cast

import platform

import typer
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

from safe_lab_agents.agents import get_agent, list_agents
from safe_lab_agents.agents.base import BaseAgent
from safe_lab_agents.config import SessionConfig, SessionMetadata, get_sessions_dir
from safe_lab_agents.docker.manager import DockerManager
from safe_lab_agents.docker.runtime import (
    configure_docker_desktop_memory,
    is_runtime_installed,
    mcp_firewall_setup_command,
    podman_windows_gateway_ip,
    setup_podman_host,
    windows_firewall_rule_status,
    wsl_interface_alias,
)
from safe_lab_agents.history.display import print_history
from safe_lab_agents.history.store import HistoryStore
from safe_lab_agents.mcp.server import ExperimentMCPServer
from safe_lab_agents.start_config import (
    CONFIG_KEY_MAP,
    DEFAULT_CONFIG_NAME,
    discover_config_path,
    load_start_config,
    resolve_param,
)
from safe_lab_agents.utils import (
    find_free_port,
    generate_session_name,
    wait_for_server,
)

# Reverse of CONFIG_KEY_MAP: ``start`` parameter name -> hyphenated config/flag key.
_PARAM_TO_FLAG: dict[str, str] = {param: key for key, param in CONFIG_KEY_MAP.items()}

logger = logging.getLogger(__name__)
console = Console(stderr=True)


def _flag_passed_on_command_line(ctx: typer.Context, param: str) -> bool:
    """Return True if ``param`` was set by an explicit command-line flag.

    Compares the ``ParameterSource`` by member name rather than identity: newer
    Typer bundles its own vendored copy of Click, so the enum returned by
    ``get_parameter_source`` can be a *different* class than the one imported at
    the top level (``click.core.ParameterSource``). An ``==``/``is`` check across
    those two classes is always False, which silently let a config-file value win
    over an explicit CLI flag (e.g. ``--agent openclaw`` ignored). Matching on
    ``.name`` is independent of which Click copy the enum came from.
    """
    source = ctx.get_parameter_source(param)
    return source is not None and source.name == "COMMANDLINE"


def _maybe_print_podman_windows_firewall_notice(config: SessionConfig) -> None:
    """On Podman/Windows, ensure the one-time inbound firewall rule is in place.

    The container reaches the host MCP server over the WSL virtual adapter; the
    Windows firewall blocks that inbound path by default. We never modify the
    firewall ourselves (that needs admin and would silently expose an
    unauthenticated, hardware-controlling server) — instead we detect the rule
    and, if missing, print the exact one-time command scoped to the WSL adapter
    so the LAN is never exposed.
    """
    if config.container_runtime != "podman" or platform.system() != "Windows":
        return
    try:
        gateway = podman_windows_gateway_ip()
        status = windows_firewall_rule_status(gateway)
        if status == "ok":
            console.print(
                "[dim]✓ WSL firewall rule present (safe-lab-agents-mcp).[/dim]"
            )
            return
        alias = wsl_interface_alias(gateway)
        command = mcp_firewall_setup_command(alias, recreate=(status == "stale"))
    except Exception as exc:  # best-effort guidance; never block the session
        logger.warning("Could not check the Windows firewall rule: %s", exc)
        return

    if status == "stale":
        body = (
            "The [bold]safe-lab-agents-mcp[/bold] firewall rule exists but no longer "
            "matches the live WSL adapter — a reboot or WSL/Hyper-V update recreated the "
            "adapter and left the rule pinned to an interface that no longer exists, so "
            "tool calls from the agent will time out.\n\n"
            "Run this in an [bold]Administrator PowerShell[/bold] to re-bind it:\n\n"
            f"  [cyan]{command}[/cyan]"
        )
    else:
        body = (
            "Podman on Windows needs a [bold]one-time[/bold] firewall rule so the agent's "
            "container can reach the tool server on this host.\n\n"
            "Run this once in an [bold]Administrator PowerShell[/bold]:\n\n"
            f"  [cyan]{command}[/cyan]\n\n"
            "It is scoped to the WSL adapter only, so the tool server is [bold]not[/bold] "
            "exposed to the network. Until it is added, tool calls from the agent will "
            "time out."
        )

    console.print(
        Panel(
            body,
            title="[yellow]Action required: Windows firewall[/yellow]",
            border_style="yellow",
        )
    )


def _prompt_container_runtime() -> str:
    """Prompt the user to choose a container runtime (no default — must choose).

    Shown as a numbered list (1/2), mirroring the agent picker; accepts either
    the number or the name.
    """
    runtimes = ["docker", "podman"]
    installed = {name: is_runtime_installed(name) for name in runtimes}
    console.print("\n[bold]Available container runtimes:[/bold]")
    for i, name in enumerate(runtimes, 1):
        suffix = "" if installed[name] else " [dim red](not detected)[/dim red]"
        console.print(f"  {i}. {name}{suffix}")
    while True:
        choice = Prompt.ask("Which container runtime?")
        if choice.isdigit() and 1 <= int(choice) <= len(runtimes):
            choice = runtimes[int(choice) - 1]
        if choice in runtimes:
            if not installed[choice]:
                console.print(
                    f"[yellow]'{choice}' was not detected on this system — "
                    f"continuing anyway (you may be prompted for its location).[/yellow]"
                )
            return choice
        console.print("[yellow]Enter 1/2 or docker/podman.[/yellow]")


def _activate_container_runtime(runtime: str) -> None:
    """Prepare the selected container runtime before connecting to it.

    For Podman, initialize/start the machine and point DOCKER_HOST at it; for
    Docker, ensure the Docker Desktop VM has enough memory. Exits the CLI if
    Podman setup fails.
    """
    if runtime == "podman":
        try:
            setup_podman_host()
        except RuntimeError as exc:
            console.print(f"[bold red]Podman setup failed:[/bold red] {exc}")
            raise typer.Exit(1)
    else:
        configure_docker_desktop_memory()


def _warn_on_runtime_engine_mismatch(
    selected_runtime: str, docker_mgr: DockerManager
) -> None:
    """Warn when the selected runtime doesn't match the engine actually connected.

    The Docker and Podman endpoints can be cross-wired on a host (e.g. Podman
    Desktop's Docker-compatibility mode serving the ``docker_engine`` pipe), so
    the engine we connect to may not match the selected runtime. We surface the
    mismatch in either direction rather than silently applying the wrong
    handling (WSL networking, firewall guidance, CLI binary).
    """
    try:
        engine_is_podman = docker_mgr.engine_is_podman()
    except Exception:
        return

    if selected_runtime == "docker" and engine_is_podman:
        console.print(
            Panel(
                "Your [bold]docker[/bold] endpoint is served by the [bold]Podman[/bold] engine "
                "(e.g. Podman Desktop's Docker-compatibility mode).\n\n"
                "This run will [bold]not[/bold] apply Podman-specific handling, so on Windows the "
                "agent may be unable to reach the tool server.\n\n"
                "• To run on Podman properly, re-run with [cyan]--container podman[/cyan].\n"
                "• To use real Docker, start Docker Desktop's engine and disable Podman's "
                "Docker-compatibility mode.",
                title="[yellow]Warning: docker is backed by Podman[/yellow]",
                border_style="yellow",
            )
        )
    elif selected_runtime == "podman" and not engine_is_podman:
        console.print(
            Panel(
                "The selected runtime is [cyan]podman[/cyan], but the connected engine is "
                "[bold]Docker[/bold], not Podman.\n\n"
                "Podman-specific handling is active while the runtime is actually Docker — likely "
                "Podman is not set up as expected (e.g. DOCKER_HOST points at a Docker engine).\n\n"
                "• To use Docker, select [cyan]--container docker[/cyan].\n"
                "• To use Podman, check your Podman machine/socket configuration.",
                title="[yellow]Warning: podman selected but engine is Docker[/yellow]",
                border_style="yellow",
            )
        )


app = typer.Typer(
    name="agent",
    help="Safely run AI agents in Docker to control scientific experiments via MCP.",
    add_completion=False,
)

BANNER = r"""
  ╭──────────────────────────────────╮
  │  ▄▖▄▖▄▖▄▖  ▖ ▄▖▄▖  ▄▖▄▖▄▖▖▖▄▖▄▖  │
  │  ▚ ▌▌▙▖▙▖  ▌ ▌▌▙▘  ▌▌▌▖▙▖▛▌▐ ▚   │
  │  ▄▌▛▌▌ ▙▖  ▙▖▛▌▙▘  ▛▌▙▌▙▖▌▌▐ ▄▌  │
  │                                  │
  │ sandboxed AI agents for lab work │
  ╰──────────────────────────────────╯
"""


def _print_banner() -> None:
    """Print the project banner (shown on start/resume)."""
    console.print(f"[cyan]{BANNER}[/cyan]")


def _print_session_start_banner(message: str) -> None:
    """Print a highly visible panel right before the agent is launched.

    Marks the point where all host-side setup is done and the container /
    agent takes over the terminal.
    """
    console.print()
    console.print(
        Panel(
            f"[bold]{message}[/bold]",
            title="[bold green]▶ SESSION STARTING[/bold green]",
            border_style="bold green",
        )
    )
    console.print()


def _print_session_exit_summary(config: SessionConfig, resumable: bool) -> None:
    """Print follow-up commands for a stopped session (resume + HTML exports)."""
    lines = []
    if resumable:
        lines.append(
            f"Resume session:           [cyan]agent resume --name {config.name}[/cyan]"
        )
    lines.append(
        f"Conversation log → HTML:  [cyan]agent history --name {config.name} --open[/cyan]"
    )
    if config.auto_log_dir:
        lines.append(
            f"Auto-log report → HTML:   [cyan]agent report {config.auto_log_dir} --open[/cyan]"
        )
    title = f"Session '{config.name}' " + (
        "committed — you can resume later" if resumable else "stopped"
    )
    console.print(
        Panel(
            "\n".join(lines),
            title=f"[bold green]{title}[/bold green]",
            border_style="green",
        )
    )


# ======================================================================
# start
# ======================================================================


@app.command()
def start(
    ctx: typer.Context,
    agent: Optional[str] = typer.Option(
        None, "--agent", "-a", help="Agent type (e.g. claude-code, openclaw)."
    ),
    tools: Optional[Path] = typer.Option(
        None, "--tools", "-t", help="Path to Python file with MCP tool functions."
    ),
    context: Optional[Path] = typer.Option(
        None, "--context", "-c", help="Directory with experiment context (read-only)."
    ),
    shared: Optional[Path] = typer.Option(
        None, "--shared", "-s", help="Shared directory for data exchange (read-write)."
    ),
    task: Optional[str] = typer.Option(
        None, "--task", help="Initial task for autonomous mode."
    ),
    name: Optional[str] = typer.Option(
        None, "--name", "-n", help="Session name (auto-generated if omitted)."
    ),
    server: Optional[list[str]] = typer.Option(
        None, "--server", help="Predefined MCP servers to enable."
    ),
    requirements: Optional[Path] = typer.Option(
        None,
        "--requirements",
        "-r",
        help="requirements.txt for extra Python packages in Docker.",
    ),
    rebuild: bool = typer.Option(
        False,
        "--rebuild",
        help="Force a full image rebuild (--no-cache --pull), ignoring the build cache. Use to pick up upstream updates to the base image or the agent/Python toolchain, which the cache cannot detect.",
    ),
    agent_args_raw: list[str] = typer.Option(
        [],
        "--agent-args",
        help="Agent-specific argument as KEY=VALUE (or KEY alone for booleans). Can be passed multiple times: --agent-args effort=high --agent-args dangerously-skip-permissions",
    ),
    project: Optional[str] = typer.Option(
        None,
        "--kadi4mat-project",
        help="Kadi4Mat project name. Enables Kadi4Mat ELN push and auto-enables --auto-log.",
    ),
    kadi_max_per_minute: int = typer.Option(
        10,
        "--kadi-max-per-minute",
        help="Kadi4Mat: max records per minute (default 10).",
    ),
    kadi_max_per_session: int = typer.Option(
        500,
        "--kadi-max-per-session",
        help="Kadi4Mat: max records per session (default 500, use 0 for unlimited).",
    ),
    port: int = typer.Option(0, "--port", help="MCP server port (0 = auto)."),
    container: Optional[str] = typer.Option(
        None,
        "--container",
        help="Container runtime: 'docker' or 'podman'. Prompted if omitted.",
    ),
    no_web: bool = typer.Option(
        False,
        "--no-web",
        help=(
            "Disable web tools (SOFT restriction for both agents — does not block network access). "
            "Claude Code: built-in web tools disabled via --disallowedTools, but "
            "Bash is still allowed so curl/wget/python can still reach the network. "
            "OpenClaw: system-prompt instruction only, no CLI enforcement."
        ),
    ),
    egress_lockdown: bool = typer.Option(
        True,
        "--egress-lockdown/--no-egress-lockdown",
        help=(
            "Firewall the container's egress before the agent starts (default: on): "
            "the host is reachable ONLY on the MCP port and private/LAN ranges are "
            "blocked, while the public internet (model API) stays open. If the rules "
            "cannot be applied the container fails closed at start — pass "
            "--no-egress-lockdown only if your runtime cannot support in-container "
            "iptables."
        ),
    ),
    mem_limit: Optional[str] = typer.Option(
        None,
        "--mem-limit",
        help=(
            "Container memory limit, e.g. '8g' or '512m'. Default: half the RAM "
            "visible to the container runtime (min 2g). Swap is always disabled "
            "alongside, so the limit is a hard ceiling (OOM-kill, no host "
            "swap-thrashing)."
        ),
    ),
    cpu_limit: Optional[float] = typer.Option(
        None,
        "--cpu-limit",
        help=(
            "Container CPU limit in cores, e.g. 2 or 2.5. Default: all but one "
            "of the runtime's cores (the spare keeps the host-side MCP tool "
            "server responsive)."
        ),
    ),
    update_tools: bool = typer.Option(
        False,
        "--update-tools",
        help="Watch the tools file and reload the MCP server automatically when it changes.",
    ),
    auto_log: Optional[bool] = typer.Option(
        None,
        "--auto-log/--no-auto-log",
        help=(
            "Automatically log every tool call as a structured ELN record "
            "(JSON + HDF5 for arrays). Records are written to shared_dir/auto_log/ "
            "or workspace/auto_log/ if no shared directory is set. "
            "If omitted, the interactive wizard asks."
        ),
    ),
    task_file: Optional[Path] = typer.Option(
        None,
        "--task-file",
        help="Path to a text/markdown file whose content is used as the initial task (mutually exclusive with --task).",
    ),
    config_path: Optional[Path] = typer.Option(
        None,
        "--config",
        help="Path to a YAML config file supplying defaults for the options below.",
    ),
    no_config: bool = typer.Option(
        False,
        "--no-config",
        help=f"Do not auto-discover {DEFAULT_CONFIG_NAME} in the current directory.",
    ),
) -> None:
    """Start a new agent session.

    If required options are not provided, an interactive wizard prompts for
    them step by step.

    Defaults for the options below can be stored in a YAML config file (keys are
    the flag names without the leading ``--``). A flag passed on the command line
    always overrides the config file.
    """
    _print_banner()
    # ---- Load config file (defaults for the options below) ----
    try:
        cfg_path = discover_config_path(config_path, no_config, Path.cwd())
    except (ValueError, FileNotFoundError) as exc:
        console.print(f"[bold red]{exc}[/bold red]")
        raise typer.Exit(1)
    cfg: dict[str, Any] = {}
    if cfg_path is not None:
        try:
            cfg = load_start_config(cfg_path)
        except (ValueError, yaml.YAMLError) as exc:
            console.print(
                f"[bold red]Failed to load config {cfg_path}: {exc}[/bold red]"
            )
            raise typer.Exit(1)
        console.print(f"[cyan]Using config:[/cyan] {cfg_path}")

    # ``agent_args_raw`` is merged separately (below) so it can be validated against
    # the agent backend; everything else is applied here unless passed on the CLI.
    agent_args_cfg = cfg.pop("agent_args_raw", None)

    def _from_config(param: str, cli_value: Any) -> Any:
        """Return the config value for ``param`` unless the user passed it on the CLI.

        Warns when an explicit command-line flag overrides a config-supplied value.
        """
        explicit = _flag_passed_on_command_line(ctx, param)
        value, override = resolve_param(param, cli_value, explicit, cfg)
        if override is not None:
            cfg_value, cli_val = override
            flag = _PARAM_TO_FLAG.get(param, param)
            console.print(
                f"[yellow]Config '{flag}' ({cfg_value!r}) overridden by command line "
                f"({cli_val!r}).[/yellow]"
            )
        return value

    agent = _from_config("agent", agent)
    tools = _from_config("tools", tools)
    context = _from_config("context", context)
    shared = _from_config("shared", shared)
    task = _from_config("task", task)
    task_file = _from_config("task_file", task_file)
    name = _from_config("name", name)
    server = _from_config("server", server)
    requirements = _from_config("requirements", requirements)
    rebuild = _from_config("rebuild", rebuild)
    project = _from_config("project", project)
    kadi_max_per_minute = _from_config("kadi_max_per_minute", kadi_max_per_minute)
    kadi_max_per_session = _from_config("kadi_max_per_session", kadi_max_per_session)
    port = _from_config("port", port)
    container = _from_config("container", container)
    no_web = _from_config("no_web", no_web)
    egress_lockdown = _from_config("egress_lockdown", egress_lockdown)
    mem_limit = _from_config("mem_limit", mem_limit)
    cpu_limit = _from_config("cpu_limit", cpu_limit)
    update_tools = _from_config("update_tools", update_tools)
    auto_log = _from_config("auto_log", auto_log)

    # ---- Interactive wizard: fill in missing values ----
    container = container or _prompt_container_runtime()
    if container not in ("docker", "podman"):
        console.print("[bold red]--container must be 'docker' or 'podman'.[/bold red]")
        raise typer.Exit(1)
    _activate_container_runtime(container)

    agent = agent or _prompt_agent()
    if tools is None:
        default_tools = Path.cwd() / "tools.py"
        tools = _prompt_path(
            "Path to your tools Python file",
            must_exist=True,
            suffix=".py",
            default=default_tools if default_tools.is_file() else None,
        )
    shared = shared or _prompt_optional_path(
        "Path to SHARED DATA DIRECTORY. The agent can read and write to this. Useful for data exchange between the agent and the host."
    )
    name = name or _prompt_session_name()
    # Kadi4Mat requires auto-log, so it's forced on downstream regardless — only
    # ask when it wasn't given on the CLI and isn't already implied by --kadi4mat-project.
    if auto_log is None and not project:
        auto_log = Confirm.ask(
            "Enable auto-logging of every tool call (structured ELN records)?",
            default=False,
        )

    # ---- Resolve agent backend (needed early for --agent-args parsing) ----
    agent_backend = get_agent(agent)

    # ---- Resolve --task-file into task string ----
    if task_file is not None:
        if task is not None:
            console.print(
                "[bold red]--task and --task-file are mutually exclusive.[/bold red]"
            )
            raise typer.Exit(1)
        if not task_file.exists():
            console.print(f"[bold red]Task file not found: {task_file}[/bold red]")
            raise typer.Exit(1)
        task = task_file.read_text(encoding="utf-8").strip()

    # ---- Parse and prompt for agent-specific args ----
    mode_for_prompt = "autonomous" if task else "interactive"
    agent_args, overridden = _merge_agent_args(
        agent_args_cfg, agent_args_raw, agent_backend
    )
    if overridden:
        console.print(
            f"[yellow]Config 'agent-args' key(s) {', '.join(overridden)} "
            f"overridden by command line.[/yellow]"
        )
    agent_args = _prompt_required_agent_args(agent_backend, agent_args, mode_for_prompt)

    if agent_args.get("dangerously-skip-permissions"):
        console.print(
            "[bold yellow]WARNING: dangerously-skip-permissions is active. Claude Code will not prompt for any permissions.[/bold yellow]"
        )

    # ---- Build session config ----
    workspace_dir = get_sessions_dir() / name / "workspace"
    workspace_dir.mkdir(parents=True, exist_ok=True)

    # ---- Configure auto-logging ----
    predefined_servers = list(server or [])
    if project and not auto_log:
        auto_log = True
        console.print(
            "[cyan]Auto-log enabled automatically (required for Kadi4Mat ELN push).[/cyan]"
        )
    if auto_log:
        shared_dir_resolved = Path(shared).resolve() if shared else None
        auto_log_host_dir = (
            (shared_dir_resolved / "auto_log")
            if shared_dir_resolved
            else (workspace_dir / "auto_log")
        )
        auto_log_container_dir = (
            "/agent/shared/auto_log" if shared else "/agent/workspace/auto_log"
        )
        # AUTO_LOG_DIR (and the KADI4MAT_* vars set below) are read inside the MCP
        # server subprocess — by autolog.make_autolog_wrapper() — not here. They
        # travel via os.environ because the subprocess inherits the environment at
        # spawn time; keep this assignment before ExperimentMCPServer.start().
        os.environ["AUTO_LOG_DIR"] = str(auto_log_host_dir)
        _write_auto_log_info(workspace_dir, auto_log_container_dir)
        _write_auto_log_client(workspace_dir, auto_log_container_dir)
        console.print(f"[cyan]Auto-log enabled:[/cyan] records → {auto_log_host_dir}")

    config = SessionConfig(
        name=name,
        agent_type=agent,
        tools_file=Path(tools).resolve(),
        context_dir=Path(context).resolve() if context else None,
        shared_dir=Path(shared).resolve() if shared else None,
        workspace_dir=workspace_dir,
        requirements_file=Path(requirements).resolve() if requirements else None,
        mcp_port=port,
        task=task if task else None,
        predefined_servers=predefined_servers,
        auto_log_dir=auto_log_host_dir if auto_log else None,
        kadi4mat_project=project,
        kadi4mat_max_per_minute=kadi_max_per_minute,
        kadi4mat_max_per_session=kadi_max_per_session,
        container_runtime=cast(Literal["docker", "podman"], container),
        no_web=no_web,
        egress_lockdown=egress_lockdown,
        mem_limit=mem_limit,
        cpu_limit=cpu_limit,
        update_tools=update_tools,
        agent_args=agent_args,
    )

    # ---- Validate ----
    _validate_config(config)

    # ---- Start MCP server ----
    mcp_port = config.mcp_port if config.mcp_port != 0 else find_free_port()
    config.mcp_port = mcp_port

    # Per-session shared secret: the agent and generated clients must present it
    # as ``Authorization: Bearer <token>``.  Authenticates the otherwise
    # 0.0.0.0-bound MCP server against LAN callers.  Ephemeral by design — a
    # fresh token each start/resume makes any token baked into a committed image
    # stale.
    auth_token = secrets.token_urlsafe(32)

    # Set Kadi4Mat env var so the predefined server can read it in the subprocess.
    if config.kadi4mat_project:
        os.environ["KADI4MAT_PROJECT"] = config.kadi4mat_project
        os.environ["KADI4MAT_MAX_PER_MINUTE"] = str(config.kadi4mat_max_per_minute)
        os.environ["KADI4MAT_MAX_PER_SESSION"] = str(config.kadi4mat_max_per_session)

    console.print(f"\n[bold]Starting MCP server on port {mcp_port} …[/bold]")
    _mcp = [
        ExperimentMCPServer(
            tools_file=config.tools_file,
            port=mcp_port,
            predefined_servers=config.predefined_servers,
            shared_dir=config.shared_dir,
            update_tools=config.update_tools,
            auth_token=auth_token,
        )
    ]
    _mcp[0].start()

    console.print("[bold]Waiting for MCP server …[/bold]")
    if not wait_for_server(mcp_port):
        console.print("[bold red]MCP server did not become ready in time.[/bold red]")
        _mcp[0].stop()
        raise typer.Exit(1)
    console.print("[green]MCP server ready.[/green]")

    _maybe_print_podman_windows_firewall_notice(config)

    from safe_lab_agents.mcp.client_generator import generate_client_files

    generate_client_files(config.tools_file, config.workspace_dir)

    if config.kadi4mat_project:
        session_limit = config.kadi4mat_max_per_session
        session_str = str(session_limit) if session_limit > 0 else "unlimited"
        console.print(
            f"[cyan]Kadi4Mat ELN enabled:[/cyan] project={config.kadi4mat_project}, "
            f"rate limits: {config.kadi4mat_max_per_minute}/min, {session_str}/session"
        )

    # ---- Tools reload monitor (--update-tools) ----
    _watcher_stop = threading.Event()
    if config.update_tools:
        _start_reload_monitor(_mcp, config, mcp_port, auth_token, _watcher_stop)

    # ---- Connect to Docker (starts Docker Desktop automatically if needed) ----
    try:
        docker_mgr = DockerManager()
    except RuntimeError as exc:
        _mcp[0].stop()
        console.print(f"[bold red]{exc}[/bold red]")
        raise typer.Exit(1)

    _warn_on_runtime_engine_mismatch(container, docker_mgr)

    # ---- Build / ensure Docker image ----
    console.print(f"[bold]Preparing Docker image for {config.agent_type} …[/bold]")
    try:
        image_tag = docker_mgr.ensure_image(
            config.agent_type, config.requirements_file, rebuild=rebuild
        )
    except Exception as exc:
        _mcp[0].stop()
        console.print(f"[bold red]Failed to build Docker image:[/bold red] {exc}")
        raise typer.Exit(1)
    console.print(f"[green]Image ready: {image_tag}[/green]")

    # ---- Determine run mode ----
    mode = "autonomous" if config.task else "interactive"
    tty = mode != "autonomous"

    extra_env: dict[str, str] = {"MCP_AUTH_TOKEN": auth_token}
    if config.agent_type == "claude-code":
        # A directly-supplied token short-circuits all credential resolution.
        # Popped (not read) so the secret is never persisted to session metadata.
        supplied_token = config.agent_args.pop("oauth-token", None)
        copy_host = config.agent_args.get("copy-host-credentials", False)
        if supplied_token:
            console.print("[dim]Using the supplied Claude OAuth token …[/dim]")
            extra_env["CLAUDE_CODE_OAUTH_TOKEN"] = supplied_token
        else:
            creds = _get_claude_credentials() if copy_host else None
            if creds:
                try:
                    oauth = json.loads(creds)["claudeAiOauth"]
                    console.print("[dim]Copying Claude credentials from host …[/dim]")
                    extra_env["CLAUDE_CREDENTIALS_JSON"] = json.dumps(
                        {"claudeAiOauth": oauth}
                    )
                except (json.JSONDecodeError, KeyError):
                    console.print(
                        "[yellow]Warning: could not parse host Claude credentials — you may need to log in inside the container.[/yellow]"
                    )
            elif mode == "autonomous":
                # No host credentials (copy is off by default): autonomous runs can't
                # prompt for login, so bootstrap one interactively inside a throwaway
                # container and seed the captured token into the real run.
                token = _bootstrap_login_in_container(
                    docker_mgr, agent_backend, image_tag
                )
                if not token:
                    console.print(
                        "[bold red]Login was not completed — no token was produced.[/bold red]"
                    )
                    _mcp[0].stop()
                    raise typer.Exit(1)
                extra_env["CLAUDE_CODE_OAUTH_TOKEN"] = token
            elif copy_host:
                console.print(
                    "[dim]No Claude credentials found on host — you may need to log in inside the container.[/dim]"
                )

    if tty:
        extra_env["TERM"] = os.environ.get("TERM", "xterm-256color")
        if "COLORTERM" in os.environ:
            extra_env["COLORTERM"] = os.environ["COLORTERM"]

    # Single-source the environment prompt: the host writes it; entrypoints read it.
    _write_system_prompt(config, agent_backend)
    # reload_tools (MCP tool) exists only with --update-tools.
    _sync_reload_info(config, reload_available=config.update_tools)

    # ---- Cleanup on exit ----
    # Registered BEFORE the container is created so an interrupt during creation
    # still runs teardown. _cleanup reads container_obj/metadata late (they stay
    # None, and are guarded, until assigned just below).
    session_dir = get_sessions_dir() / config.name
    _cleaned_up = [False]
    container_obj: Any = None
    metadata: Any = None

    def _cleanup(signum=None, frame=None):
        if _cleaned_up[0]:
            return
        _cleaned_up[0] = True
        console.print("\n[bold]Shutting down …[/bold]")

        # Copy native logs out of the container before it is committed/removed.
        if container_obj is not None:
            try:
                docker_mgr.copy_agent_logs(
                    container_obj.id, session_dir, config.agent_type
                )
            except Exception as exc:
                logger.warning("Could not copy agent logs: %s", exc)

        # Import the freshly-copied logs into history.json.
        try:
            store = HistoryStore(config.name)
            store.import_from_agent(agent_backend, session_dir / "logs")
        except Exception as exc:
            logger.warning("Could not import history: %s", exc)

        committed = False
        if container_obj is not None:
            try:
                docker_mgr.commit_container(container_obj.id, config.name)
                committed = True
            except Exception as exc:
                logger.warning("Could not commit container: %s", exc)
            try:
                docker_mgr.remove_container(container_obj.id, force=True)
            except Exception:
                pass
        _watcher_stop.set()
        _mcp[0].shutdown()
        if config.auto_log_dir:
            try:
                from safe_lab_agents.mcp.predefined.autolog import (
                    write_session_summary,
                )

                write_session_summary(config.auto_log_dir)
            except Exception as exc:
                logger.warning("Could not write auto-log session summary: %s", exc)
        if metadata is not None:
            metadata.status = "committed"
            metadata.stopped_at = datetime.now()
            metadata.save()
        _print_session_exit_summary(config, resumable=committed)
        if signum is not None:
            sys.exit(0)

    signal.signal(signal.SIGINT, _cleanup)
    signal.signal(signal.SIGTERM, _cleanup)
    atexit.register(_cleanup)

    # ---- Create the container (cleanup handlers are already live) ----
    container_obj = docker_mgr.create_container(
        config,
        agent_backend,
        mcp_port,
        image_tag=image_tag,
        extra_env=extra_env,
        tty=tty,
    )

    # ---- Save metadata ----
    metadata = SessionMetadata(
        config=config,
        container_id=container_obj.id,
        image_tag=image_tag,
        status="running",
        started_at=datetime.now(),
    )
    metadata.save()

    # ---- Run ----
    _print_session_start_banner(f"Launching {config.agent_type} in {mode} mode …")

    try:
        if mode == "autonomous":
            stream_log = _prepare_stream_log(config, session_dir)
            docker_mgr.start_autonomous(
                container_obj, agent_backend, stream_log_file=stream_log
            )
        else:
            docker_mgr.start_interactive(container_obj)
    except Exception as exc:
        console.print(f"[bold red]Error: {exc}[/bold red]")
    finally:
        _cleanup()


# ======================================================================
# resume
# ======================================================================


@app.command()
def resume(
    name: Optional[str] = typer.Option(
        None, "--name", "-n", help="Session name to resume."
    ),
    agent_args_raw: list[str] = typer.Option(
        [],
        "--agent-args",
        help="Override agent-specific args for this resume (KEY=VALUE or KEY for booleans).",
    ),
) -> None:
    """Resume a previously stopped session.

    Resume is always interactive — a previously autonomous (``--task``) session
    is continued interactively so you can drive it by hand, never re-run
    autonomously.
    """
    _print_banner()
    name = name or _pick_session("Which session to resume?")
    if not name:
        raise typer.Exit(1)

    metadata = SessionMetadata.load(name)
    config = metadata.config

    agent_backend = get_agent(config.agent_type)

    # Resume always runs interactively. Clear any persisted task so the container
    # env is built for interactive mode (no MODE=autonomous / TASK_PROMPT) and the
    # entrypoint continues the conversation interactively rather than re-running.
    config.task = None

    # Merge --agent-args overrides into stored config.
    if agent_args_raw:
        overrides = _parse_agent_args(agent_args_raw, agent_backend)
        config.agent_args = {**config.agent_args, **overrides}

    # The runtime is autodetected from the session's saved metadata — the engine
    # it was created with is the engine its committed image lives on. Activate it
    # so every downstream runtime-dependent step (image lookup, MCP host
    # resolution) stays consistent with the engine we actually connect to.
    _activate_container_runtime(config.container_runtime)
    try:
        docker_mgr = DockerManager()
    except RuntimeError as exc:
        console.print(f"[bold red]{exc}[/bold red]")
        raise typer.Exit(1)

    _warn_on_runtime_engine_mismatch(config.container_runtime, docker_mgr)

    # Check for committed image on the autodetected engine only.
    session_image = docker_mgr.get_session_image_tag(name)
    if not session_image:
        console.print(
            f"[bold red]No committed {config.container_runtime} image found for "
            f"session '{name}'.[/bold red]"
        )
        raise typer.Exit(1)

    # Restore env vars needed by predefined servers in the MCP subprocess
    # (mirrors `start`; without these the resumed session silently drops
    # auto-logging and the Kadi4Mat ELN push).
    if config.auto_log_dir:
        os.environ["AUTO_LOG_DIR"] = str(config.auto_log_dir)
    if config.kadi4mat_project:
        os.environ["KADI4MAT_PROJECT"] = config.kadi4mat_project
        os.environ["KADI4MAT_MAX_PER_MINUTE"] = str(config.kadi4mat_max_per_minute)
        os.environ["KADI4MAT_MAX_PER_SESSION"] = str(config.kadi4mat_max_per_session)

    # Start MCP server.
    mcp_port = config.mcp_port if config.mcp_port != 0 else find_free_port()
    config.mcp_port = mcp_port

    # Fresh per-session token (see start_session): a new secret each resume makes
    # any token baked into the committed image stale.  The entrypoint re-registers
    # the MCP server on every start, so the new token flows in.
    auth_token = secrets.token_urlsafe(32)

    # Honour --update-tools if the session was originally created with it: the
    # server re-registers the reload_tools MCP tool, and the monitor below
    # restarts the server on reload — exactly as `start` does.
    _mcp = [
        ExperimentMCPServer(
            tools_file=config.tools_file,
            port=mcp_port,
            predefined_servers=config.predefined_servers,
            shared_dir=config.shared_dir,
            update_tools=config.update_tools,
            auth_token=auth_token,
        )
    ]
    _mcp[0].start()
    if not wait_for_server(mcp_port):
        console.print("[bold red]MCP server did not become ready in time.[/bold red]")
        _mcp[0].stop()
        raise typer.Exit(1)

    _watcher_stop = threading.Event()
    if config.update_tools:
        _start_reload_monitor(_mcp, config, mcp_port, auth_token, _watcher_stop)

    _maybe_print_podman_windows_firewall_notice(config)

    # Create container from committed image (always interactive — see above).
    # A directly-supplied OAuth token is re-injected (never persisted): token-based
    # sessions don't carry credentials in the committed image, so resuming them
    # requires re-supplying the token via --agent-args oauth-token=…
    resume_extra_env: dict[str, str] = {"MCP_AUTH_TOKEN": auth_token}
    # Propagate the terminal type so the interactive TUI renders correctly (mirrors start).
    resume_extra_env["TERM"] = os.environ.get("TERM", "xterm-256color")
    if "COLORTERM" in os.environ:
        resume_extra_env["COLORTERM"] = os.environ["COLORTERM"]
    if config.agent_type == "claude-code":
        resume_token = config.agent_args.pop("oauth-token", None)
        if resume_token:
            console.print("[dim]Using the supplied Claude OAuth token …[/dim]")
            resume_extra_env["CLAUDE_CODE_OAUTH_TOKEN"] = resume_token

    # Regenerate the prompt and Python client files from the current tools file so
    # a resume picks up edits made between sessions (mirrors start).
    _write_system_prompt(config, agent_backend)
    # reload_tools is available on resume only if the session uses --update-tools.
    _sync_reload_info(config, reload_available=config.update_tools)
    from safe_lab_agents.mcp.client_generator import generate_client_files

    generate_client_files(config.tools_file, config.workspace_dir)

    # ---- Cleanup on exit ----
    # Registered BEFORE the container is created so an interrupt during creation
    # still runs teardown. _cleanup reads `container` late (None until assigned).
    session_dir = get_sessions_dir() / config.name
    _cleaned_up = [False]
    container: Any = None

    def _cleanup(signum=None, frame=None):
        if _cleaned_up[0]:
            return
        _cleaned_up[0] = True
        console.print("\n[bold]Shutting down …[/bold]")

        # Copy native logs out of the container before it is committed/removed.
        if container is not None:
            try:
                docker_mgr.copy_agent_logs(container.id, session_dir, config.agent_type)
            except Exception as exc:
                logger.warning("Could not copy agent logs: %s", exc)

        # Import the freshly-copied logs into history.json.
        try:
            store = HistoryStore(config.name)
            store.import_from_agent(agent_backend, session_dir / "logs")
        except Exception as exc:
            logger.warning("Could not import history: %s", exc)

        committed = False
        if container is not None:
            try:
                docker_mgr.commit_container(container.id, config.name)
                committed = True
            except Exception:
                pass
            try:
                docker_mgr.remove_container(container.id, force=True)
            except Exception:
                pass
        _watcher_stop.set()
        _mcp[0].shutdown()
        if config.auto_log_dir:
            try:
                from safe_lab_agents.mcp.predefined.autolog import (
                    write_session_summary,
                )

                write_session_summary(config.auto_log_dir)
            except Exception as exc:
                logger.warning("Could not write auto-log session summary: %s", exc)
        metadata.status = "committed"
        metadata.stopped_at = datetime.now()
        metadata.save()
        _print_session_exit_summary(config, resumable=committed)
        if signum is not None:
            sys.exit(0)

    signal.signal(signal.SIGINT, _cleanup)
    signal.signal(signal.SIGTERM, _cleanup)
    atexit.register(_cleanup)

    # ---- Create the container (cleanup handlers are already live) ----
    container = docker_mgr.create_container(
        config,
        agent_backend,
        mcp_port,
        image_tag=session_image,
        resume=True,
        extra_env=resume_extra_env,
        tty=True,
    )

    metadata.container_id = container.id
    metadata.status = "running"
    metadata.started_at = datetime.now()
    metadata.save()

    _print_session_start_banner(
        f"Resuming {config.agent_type} ({name}) interactively …"
    )

    try:
        docker_mgr.start_interactive(container)
    except Exception as exc:
        console.print(f"[bold red]Error: {exc}[/bold red]")
    finally:
        _cleanup()


# ======================================================================
# history
# ======================================================================


@app.command()
def history(
    name: Optional[str] = typer.Option(None, "--name", "-n", help="Session name."),
    last: Optional[int] = typer.Option(
        None, "--last", "-l", help="Show last N entries."
    ),
    html_out: Optional[Path] = typer.Option(
        None,
        "--html",
        "-o",
        help="Write a self-contained HTML viewer instead of printing "
        "(default path: <session>/conversation_safe_lab_agents.html). Implied by --open.",
    ),
    open_browser: bool = typer.Option(
        False,
        "--open",
        help="Open the HTML conversation viewer in the default browser.",
    ),
) -> None:
    """View the conversation history of a session.

    Without flags this prints to the terminal; ``--html``/``--open`` instead
    render a self-contained HTML viewer (works for both claude-code and
    openclaw sessions).
    """
    name = name or _pick_session("Which session's history to view?")
    if not name:
        raise typer.Exit(1)

    # Re-import from on-disk logs (copied at shutdown) so history.json is
    # always up-to-date without requiring Docker to be running.
    sessions = {m.config.name: m for m in SessionMetadata.list_sessions()}
    if name in sessions:
        cfg = sessions[name].config
        logs_dir = get_sessions_dir() / name / "logs"
        if logs_dir.exists():
            try:
                agent = get_agent(cfg.agent_type)
                HistoryStore(name).import_from_agent(agent, logs_dir)
            except Exception as exc:
                logger.warning("Could not refresh history from logs: %s", exc)

    if html_out is not None or open_browser:
        from safe_lab_agents.history.html import build_conversation_html

        try:
            metadata = SessionMetadata.load(name)
        except FileNotFoundError:
            metadata = None

        entries = HistoryStore(name).load_history()
        out_path = html_out or (
            get_sessions_dir() / name / "conversation_safe_lab_agents.html"
        )
        build_conversation_html(entries, metadata, out_path)
        console.print(f"[green]Conversation written →[/green] {out_path}")
        if open_browser:
            import webbrowser

            webbrowser.open(out_path.resolve().as_uri())
        return

    print_history(name, last=last, console=console)


# ======================================================================
# list
# ======================================================================


@app.command("list")
def list_sessions() -> None:
    """List all saved agent sessions."""
    sessions = SessionMetadata.list_sessions()
    if not sessions:
        console.print("[dim]No sessions found.[/dim]")
        return

    table = Table(title="Agent Sessions")
    table.add_column("Name", style="bold")
    table.add_column("Agent")
    table.add_column("Container")
    table.add_column("Status")
    table.add_column("Created")
    table.add_column("Last stopped")
    table.add_column("Task")

    for meta in sessions:
        cfg = meta.config
        stopped = meta.stopped_at.strftime("%Y-%m-%d %H:%M") if meta.stopped_at else "-"
        task_str = (
            (cfg.task[:50] + "…")
            if cfg.task and len(cfg.task) > 50
            else (cfg.task or "-")
        )
        table.add_row(
            cfg.name,
            cfg.agent_type,
            cfg.container_runtime,
            meta.status,
            cfg.created_at.strftime("%Y-%m-%d %H:%M"),
            stopped,
            task_str,
        )

    console.print(table)


# ======================================================================
# report
# ======================================================================


@app.command()
def report(
    log_dir: Path = typer.Argument(
        ..., help="Path to an auto_log/ folder (JSON + HDF5 + figures)."
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output HTML path (default: <log_dir>/report_safe_lab_agents.html).",
    ),
    open_browser: bool = typer.Option(
        False, "--open", help="Open the report in the default browser when done."
    ),
) -> None:
    """Build a self-contained HTML report from an auto-log folder."""
    from safe_lab_agents.report import build_report

    if not log_dir.is_dir():
        console.print(f"[bold red]Log directory not found: {log_dir}[/bold red]")
        raise typer.Exit(1)

    out_path = output or (log_dir / "report_safe_lab_agents.html")
    try:
        build_report(log_dir, out_path)
    except Exception as exc:
        console.print(f"[bold red]Failed to build report:[/bold red] {exc}")
        raise typer.Exit(1)

    console.print(f"[green]Report written →[/green] {out_path}")
    if open_browser:
        import webbrowser

        webbrowser.open(out_path.resolve().as_uri())


# ======================================================================
# export-eln
# ======================================================================


@app.command("export-eln")
def export_eln(
    log_dir: Path = typer.Argument(
        ..., help="Path to an auto_log/ folder (JSON + HDF5 + figures)."
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output .eln path (default: <log_dir>/<session>.eln).",
    ),
    name: Optional[str] = typer.Option(
        None, "--name", help="Human name for the session (root Dataset)."
    ),
    author: Optional[str] = typer.Option(
        None, "--author", help="Optional human author name to attribute the records to."
    ),
    affiliation: Optional[str] = typer.Option(
        None, "--affiliation", help="Optional organisation for the author."
    ),
) -> None:
    """Export an auto-log folder as a standard .eln (RO-Crate) file for ELN import."""
    from safe_lab_agents.export import build_eln

    if not log_dir.is_dir():
        console.print(f"[bold red]Log directory not found: {log_dir}[/bold red]")
        raise typer.Exit(1)

    out_path = output or (log_dir / f"{log_dir.parent.name or 'session'}.eln")
    try:
        build_eln(log_dir, out_path, name=name, author=author, affiliation=affiliation)
    except Exception as exc:
        console.print(f"[bold red]Failed to build .eln:[/bold red] {exc}")
        raise typer.Exit(1)

    console.print(f"[green].eln written →[/green] {out_path}")


# ======================================================================
# Stream-log helper
# ======================================================================


def _prepare_stream_log(config: "SessionConfig", session_dir: Path) -> Optional[Path]:
    """Create the stream log path for an autonomous claude-code session.

    Writes a synthetic user record with the task prompt so history includes it,
    then returns the path for ``start_autonomous`` to append raw JSON lines.
    Only used for agent types whose autonomous output is stream-json
    (currently claude-code).
    """
    if config.agent_type != "claude-code":
        return None
    log_path = (
        session_dir / "logs" / "projects" / "-autonomous-run" / f"{config.name}.jsonl"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if config.task:
        from datetime import timezone

        task_record = {
            "type": "user",
            "message": {"role": "user", "content": config.task},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        log_path.write_text(json.dumps(task_record) + "\n", encoding="utf-8")
    else:
        log_path.write_text("", encoding="utf-8")
    return log_path


# ======================================================================
# Agent-args helpers
# ======================================================================


def _agent_args_dict_to_tokens(mapping: dict[str, Any]) -> list[str]:
    """Convert a config ``agent-args`` mapping into ``--agent-args`` token form.

    ``True`` becomes the bare ``key`` (boolean flag); everything else becomes
    ``key=value`` so it flows through the same validation as CLI-passed tokens.
    """
    tokens: list[str] = []
    for key, value in mapping.items():
        if value is True:
            tokens.append(str(key))
        else:
            tokens.append(f"{key}={value}")
    return tokens


def _merge_agent_args(
    agent_args_cfg: Optional[dict[str, Any]],
    cli_raw: list[str],
    agent_backend: Any,
) -> tuple[dict[str, Any], list[str]]:
    """Merge config-file ``agent-args`` with CLI ``--agent-args``.

    The config mapping supplies the base; individual CLI keys override individual
    config keys (rather than the CLI discarding the whole config mapping). Both
    are validated/coerced through :func:`_parse_agent_args`.

    Returns ``(merged, overridden)`` where ``overridden`` is the sorted list of
    config keys the CLI replaced (so the caller can warn about them).
    """
    cli_agent_args = _parse_agent_args(cli_raw, agent_backend)
    if agent_args_cfg is None:
        return cli_agent_args, []
    cfg_agent_args = _parse_agent_args(
        _agent_args_dict_to_tokens(agent_args_cfg), agent_backend
    )
    overridden = sorted(set(cfg_agent_args) & set(cli_agent_args))
    return {**cfg_agent_args, **cli_agent_args}, overridden


def _parse_agent_args(raw: list[str], agent_backend: Any) -> dict[str, Any]:
    """Parse and type-coerce ``--agent-args`` tokens into a dict.

    Each token is either ``key=value`` or ``key`` (treated as boolean True).
    """
    if not raw:
        return {}
    declared = {arg.name: arg for arg in agent_backend.get_agent_args()}
    result: dict[str, Any] = {}
    for token in raw:
        if "=" in token:
            key, _, value_str = token.partition("=")
            bare = False
        else:
            key, value_str, bare = token.strip(), None, True
        if key not in declared:
            available = ", ".join(declared) or "(none)"
            console.print(
                f"[bold red]Unknown agent arg '{key}'. Valid: {available}[/bold red]"
            )
            raise typer.Exit(1)
        spec = declared[key]
        if spec.type is bool:
            if bare or (value_str or "").lower() in ("1", "true", "yes", ""):
                result[key] = True
            elif (value_str or "").lower() in ("0", "false", "no"):
                result[key] = False
            else:
                console.print(
                    f"[bold red]Arg '{key}': invalid boolean '{value_str}'.[/bold red]"
                )
                raise typer.Exit(1)
        elif spec.type is int:
            try:
                result[key] = int(value_str or "")
            except ValueError:
                console.print(
                    f"[bold red]Arg '{key}': '{value_str}' is not an integer.[/bold red]"
                )
                raise typer.Exit(1)
        else:
            if bare:
                console.print(
                    f"[bold red]Arg '{key}' requires a value, e.g. {key}=somevalue.[/bold red]"
                )
                raise typer.Exit(1)
            if spec.choices and value_str not in spec.choices:
                console.print(
                    f"[bold red]Arg '{key}': '{value_str}' not valid. "
                    f"Choose: {', '.join(spec.choices)}[/bold red]"
                )
                raise typer.Exit(1)
            result[key] = value_str
    return result


def _prompt_required_agent_args(
    agent_backend: Any, parsed: dict[str, Any], mode: str
) -> dict[str, Any]:
    """Prompt for any required agent args that are missing."""
    result = dict(parsed)
    for arg in agent_backend.get_agent_args():
        is_required = arg.required or (
            arg.required_for_autonomous and mode == "autonomous"
        )
        if is_required and arg.name not in result:
            suffix = f" ({'/'.join(arg.choices)})" if arg.choices else ""
            raw_val = Prompt.ask(f"{arg.description}{suffix}", password=arg.is_secret)
            if arg.type is bool:
                # Empty input (bare Enter) must NOT count as True — require an
                # explicit affirmative for a bool the user is being asked for.
                result[arg.name] = raw_val.lower() in ("1", "true", "yes", "y")
            elif arg.type is int:
                # Re-prompt instead of crashing with a ValueError traceback.
                while True:
                    try:
                        result[arg.name] = int(raw_val)
                        break
                    except ValueError:
                        console.print(
                            f"[bold red]'{raw_val}' is not an integer.[/bold red]"
                        )
                        raw_val = Prompt.ask(
                            f"{arg.description}{suffix}", password=arg.is_secret
                        )
            else:
                result[arg.name] = raw_val
    return result


# ======================================================================
# Interactive wizard helpers
# ======================================================================


def _prompt_agent() -> str:
    """Prompt the user to choose an agent backend."""
    available = list_agents()
    console.print("\n[bold]Available agents:[/bold]")
    for i, name in enumerate(available, 1):
        console.print(f"  {i}. {name}")
    choice = Prompt.ask(
        "Which agent?",
        default=available[0] if available else "claude-code",
    )
    # Accept either the number or the name.
    if choice.isdigit() and 1 <= int(choice) <= len(available):
        return available[int(choice) - 1]
    if choice in available:
        return choice
    console.print(
        f"[yellow]Unknown agent '{choice}', defaulting to '{available[0]}'.[/yellow]"
    )
    return available[0]


def _prompt_path(
    label: str,
    must_exist: bool = False,
    suffix: Optional[str] = None,
    default: Optional[Path] = None,
) -> Path:
    """Prompt until the user provides a valid path."""
    while True:
        if default is not None:
            raw = Prompt.ask(label, default=str(default))
        else:
            raw = Prompt.ask(label)
        p = Path(raw).expanduser().resolve()
        if must_exist and not p.exists():
            console.print(f"[red]Path does not exist: {p}[/red]")
            continue
        if suffix and p.suffix != suffix:
            console.print(f"[red]File must have a {suffix} extension.[/red]")
            continue
        return p


def _prompt_optional_path(label: str) -> Optional[Path]:
    """Prompt for an optional directory path (Enter to skip)."""
    raw = Prompt.ask(label, default="")
    if not raw:
        return None
    p = Path(raw).expanduser().resolve()
    if not p.exists():
        console.print(f"[yellow]Warning: path does not exist yet: {p}[/yellow]")
    return p


def _prompt_optional_str(label: str) -> Optional[str]:
    """Prompt for an optional string (Enter to skip)."""
    raw = Prompt.ask(label, default="")
    return raw if raw else None


def _prompt_session_name() -> str:
    """Prompt for a session name or auto-generate one."""
    default = generate_session_name()
    raw = Prompt.ask("Session name", default=default)
    return raw


def _pick_session(prompt: str) -> Optional[str]:
    """Show available sessions and let the user pick one."""
    sessions = SessionMetadata.list_sessions()
    if not sessions:
        console.print("[dim]No sessions found.[/dim]")
        return None

    console.print(f"\n[bold]{prompt}[/bold]")
    for i, meta in enumerate(sessions, 1):
        cfg = meta.config
        console.print(f"  {i}. {cfg.name}  ({cfg.agent_type}, {meta.status})")

    choice = Prompt.ask("Enter number or name")
    if choice.isdigit() and 1 <= int(choice) <= len(sessions):
        return sessions[int(choice) - 1].config.name
    # Try matching by name.
    for meta in sessions:
        if meta.config.name == choice:
            return choice
    console.print(f"[red]No session matching '{choice}'.[/red]")
    return None


# OAuth tokens minted by `claude setup-token` look like
# ``sk-ant-oat01-<base64url>``.  Match the prefix and the contiguous token body.
_OAUTH_TOKEN_RE = re.compile(r"sk-ant-oat\d{2}-[A-Za-z0-9_-]+")


def _extract_oauth_token(recording: str) -> Optional[str]:
    """Pull the OAuth token out of a ``claude setup-token`` session recording.

    The recording (captured via ``script``) contains terminal control codes, but
    the token is a single contiguous run of token characters preceded by an ANSI
    colour code and followed by a carriage return, so a direct regex search is
    reliable.
    """
    match = _OAUTH_TOKEN_RE.search(recording)
    return match.group(0) if match else None


def _bootstrap_login_in_container(
    docker_mgr, agent_backend, image_tag
) -> Optional[str]:
    """Mint a Claude OAuth token via an interactive login in a throwaway container.

    Used for autonomous runs when the host has no Claude credentials: a TTY
    container runs ``claude setup-token`` (under ``script`` so it gets a PTY).
    The user opens the printed URL, signs in, and pastes the code back; the
    command prints a long-lived token and exits on its own.  We harvest the
    recorded session via ``docker cp`` and extract the token.

    Returns the OAuth token string on success, or ``None`` if the user aborted /
    no token was produced.  The caller handles MCP shutdown / exit on ``None``.
    The throwaway container is always removed before returning.
    """
    console.print(
        "[bold]No Claude credentials found.[/bold] Launching a one-time login inside the container …"
    )
    console.print(
        "[dim]Open the printed URL, sign in, and paste the code back. Login "
        "finishes and the autonomous run starts automatically — nothing to quit.[/dim]\n"
    )
    login_container = None
    try:
        login_container = docker_mgr.create_login_container(image_tag, agent_backend)
        docker_mgr.start_interactive(login_container)  # blocks until setup-token exits
        recording = docker_mgr.read_file_from_container(
            login_container.id, "/home/agent/.setup-token.log"
        )
        if not recording:
            return None
        return _extract_oauth_token(recording)
    finally:
        if login_container is not None:
            try:
                docker_mgr.remove_container(login_container.id, force=True)
            except Exception:
                pass


def _get_claude_credentials() -> Optional[str]:
    """Read Claude Code OAuth credentials from the host's native secret store.

    Returns the raw JSON string ``{"claudeAiOauth": {...}}``, or ``None``.

    - macOS:   reads from the macOS Keychain via ``security find-generic-password``
    - Linux:   reads from ``~/.claude/.credentials.json``
    - Windows: reads from Windows Credential Manager via PowerShell
    """
    import platform
    import subprocess

    system = platform.system()

    if system == "Darwin":
        result = subprocess.run(
            [
                "security",
                "find-generic-password",
                "-s",
                "Claude Code-credentials",
                "-w",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()

    elif system == "Windows":
        ps = (
            "$v=New-Object Windows.Security.Credentials.PasswordVault;"
            "$c=$v.FindAllByResource('Claude Code-credentials');"
            "if($c.Count){$c[0].RetrievePassword();Write-Output $c[0].Password}"
        )
        result = subprocess.run(
            ["powershell", "-Command", ps],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()

    # Linux + universal fallback
    creds_file = Path.home() / ".claude" / ".credentials.json"
    if creds_file.exists():
        return creds_file.read_text(encoding="utf-8")

    return None


def _write_system_prompt(config: SessionConfig, agent: BaseAgent) -> None:
    """Write the agent's environment system prompt into the workspace.

    Both entrypoint scripts read ``/agent/workspace/system_prompt.txt`` as the
    base prompt, so the prose lives in one place (``BaseAgent.get_system_prompt``)
    instead of being duplicated in each shell script.  A trailing newline keeps
    the entrypoints' subsequent appends (autonomous instruction, tool-info
    files) on their own lines.
    """
    config.workspace_dir.mkdir(parents=True, exist_ok=True)
    (config.workspace_dir / "system_prompt.txt").write_text(
        agent.get_system_prompt(config) + "\n", encoding="utf-8"
    )


_RELOAD_INFO = """\

## Reloading tools

The tools file lives on the host and is edited by the user, not by you. When the
user tells you they have updated it, call the MCP tool
`mcp__experiment-tools__reload_tools` (no arguments) to pick up the changes.
Only call it when the user asks you to reload — never on your own. After the MCP
connection reconnects, refresh your MCP tool list; if Python tools changed,
re-import /agent/workspace/tools_client.py.
"""


def _sync_reload_info(config: SessionConfig, reload_available: bool) -> None:
    """Write (or remove) reload_info.txt so the prompt only mentions reload_tools
    when it actually exists this run.

    ``reload_tools`` is an MCP tool registered only when ``--update-tools`` is
    active, and resume does not register it — so the guidance must not appear
    otherwise.  Both entrypoints inject ``/agent/workspace/reload_info.txt`` into
    the system prompt when present.  The file is removed when reload is
    unavailable so a stale copy from an earlier start does not leak into resume.
    """
    path = config.workspace_dir / "reload_info.txt"
    if reload_available:
        config.workspace_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(_RELOAD_INFO, encoding="utf-8")
    elif path.exists():
        path.unlink()


def _start_reload_monitor(mcp_holder, config, mcp_port, auth_token, stop_event):
    """Restart the MCP server when the agent calls ``reload_tools``.

    Runs in a daemon thread.  *mcp_holder* is a one-element list holding the live
    :class:`ExperimentMCPServer` so the thread can swap in the restarted server.
    Shared by ``start`` and ``resume`` whenever ``--update-tools`` is active.
    """
    from safe_lab_agents.mcp.client_generator import generate_client_files

    def _watcher_print(msg: str) -> None:
        sys.stderr.write("\r\n")
        sys.stderr.flush()
        console.print(msg)

    def _reload_monitor() -> None:
        import queue as _queue

        while not stop_event.is_set():
            try:
                mcp_holder[0].reload_queue.get(timeout=1.0)
            except _queue.Empty:
                continue
            if stop_event.is_set():
                break
            time.sleep(
                0.5
            )  # let the HTTP response reach the agent before socket closes
            _watcher_print(
                "[yellow]Agent requested tools reload – restarting MCP server …[/yellow]"
            )
            try:
                mcp_holder[0].stop()
                time.sleep(0.5)  # allow OS to release the port
                new_server = ExperimentMCPServer(
                    tools_file=config.tools_file,
                    port=mcp_port,
                    predefined_servers=config.predefined_servers,
                    shared_dir=config.shared_dir,
                    update_tools=True,
                    auth_token=auth_token,
                )
                new_server.start()
                mcp_holder[0] = new_server
                if wait_for_server(mcp_port):
                    generate_client_files(config.tools_file, config.workspace_dir)
                    _watcher_print(
                        "[green]MCP server restarted – updated tools are now available.[/green]"
                    )
                else:
                    _watcher_print("[red]MCP server did not restart in time.[/red]")
            except Exception as exc:
                _watcher_print(f"[red]Tools reload error: {exc}[/red]")

    threading.Thread(
        target=_reload_monitor, daemon=True, name="tools-reload-monitor"
    ).start()


def _write_auto_log_info(workspace_dir: Path, container_log_dir: str) -> None:
    """Write auto_log_info.txt into the workspace so entrypoints can inject it into the system prompt."""
    content = f"""\

## Experiment auto-logging

All MCP tool calls and Python tool calls are automatically recorded to
{container_log_dir}/ as structured ELN records. Each record captures the
tool name, call parameters, return values, and timestamps. Numpy arrays are
stored as HDF5 datasets. You do not need to save experiment results manually.

### Grouping calls into a batch

For sweeps, optimisation loops, or any multi-step protocol, use the batch
helpers from /agent/workspace/auto_log_client.py to group all related calls
into a single merged ELN record:

    import sys; sys.path.insert(0, "/agent/workspace")
    from auto_log_client import start_batch, stop_batch

    start_batch("Voltage sweep 0–5 V")
    for v in voltages:
        measure(v)          # logged automatically
    stop_batch()            # writes one merged record for the whole sweep

Use batches when:
- Running a parameter sweep (voltage, temperature, frequency, concentration)
- Running an optimisation loop (Bayesian optimisation, grid search, …)
- Executing a multi-step protocol (calibrate → acquire → verify)
- Repeating a measurement N times for statistics

Without a batch, each tool call creates its own individual record — fine for
one-off measurements.

### Writing and running Python scripts

ALWAYS save every Python script to a file in /agent/shared/scripts before executing
it. Never run analysis code as a one-liner or inline snippet. Saving first means:
- The script can be passed verbatim to log_analysis(script=...) for full
  reproducibility — open(__file__).read() only works in a saved file.
- Scripts are preserved in the session workspace even if the analysis is re-run.

Naming convention: use descriptive names, e.g. fit_voltage_sweep.py,
plot_spectrum.py. Save to /agent/shared/scripts/ so they persist across sessions.

### Recording analysis results

After running an analysis script, call log_analysis() to add a structured
analysis entry to the ELN. log_analysis() must always be called from inside
a saved script file — never from an interactive one-liner — so that
open(__file__).read() captures the full analysis code correctly.

### Log everything, not just successes

A failed attempt is data, not noise. Record it — do NOT silently retry and
throw the failure away. Every analysis you run, every debugging detour, every
hypothesis you form, and every non-obvious decision you make should leave a
log_analysis() entry behind. A reviewer reading the log afterwards should be
able to reconstruct not just what worked, but what you tried, what broke, and
why you made the choices you did.

Tag each entry with the `kind` parameter so the record is self-describing:

- kind="analysis"    — a successful result or conclusion (the default)
- kind="hypothesis"  — what you expect before a measurement/sweep and why you
                       chose these parameters or ranges
- kind="decision"    — the rationale for an approach, a plan change, or a
                       decision to stop or continue
- kind="debug"       — a debugging step, or a failed-then-fixed iteration
                       (what broke, what you changed, whether it helped)
- kind="failed"      — an attempt that did not succeed: a script that errored,
                       a fit that didn't converge, an instrument that returned
                       garbage. Paste the traceback / error into `text` and the
                       failing code into `script`.
- kind="observation" — an anomaly, an unexpected or negative result worth
                       keeping, or a "what to measure next" note

When a script errors, capture the traceback and log it before moving on, e.g.:

    import sys, traceback
    sys.path.insert(0, "/agent/workspace")
    from auto_log_client import log_analysis
    try:
        ...                       # the analysis that might fail
    except Exception:
        log_analysis(
            title="Gaussian fit of peak did not converge",
            kind="failed",
            text="curve_fit raised RuntimeError — initial guess for the width "
                 "was too small. Next: widen p0 and bound sigma > 0.\\n\\n"
                 + traceback.format_exc(),
            script=open(__file__).read(),
            references=["exp_20260522_111149_616781"],
        )
        raise

Workflow:
  1. Write the analysis script and save it to /agent/shared/scripts/my_analysis.py
  2. Run the script: python /agent/shared/scripts/my_analysis.py
  3. The script calls log_analysis() as its last step (or in an except block on failure)

    import sys; sys.path.insert(0, "/agent/workspace")
    from auto_log_client import log_analysis, AUTO_LOG_DIR

Signature:
    log_analysis(title, text="", data={{}}, references=[], script="", figures=[], kind="analysis")

Parameters:
- title (str): Short label for this analysis, e.g. "Linear fit — voltage sweep"
- text (str): Your written interpretation — conclusions, observations, confidence
  in the result, what should be measured next. Write this as you would in a lab
  notebook: what does the result mean physically?
- data (dict): Computed quantities to preserve. Scalars and strings go directly
  into the JSON. Numpy arrays are saved to HDF5 automatically — include things
  like fitted parameters, residuals, processed spectra, derived arrays.
- references (list[str]): IDs of the raw ELN entries this analysis is based on.
  Open the relevant exp_*.json or batch_*.json files in {container_log_dir}/ and
  copy the top-level "id" field, e.g.:
      ["exp_20260522_111149_616781", "batch_20260522_114500_000001"]
  This links the analysis permanently to the raw data it was derived from.
  Always include references — they are the chain of provenance.
- script (str): Always pass open(__file__).read() — this captures the full
  source of the analysis script and stores it verbatim in the ELN entry,
  making the analysis exactly reproducible from the record alone.
- figures (list[str]): Filenames of plots produced by this analysis.
  Figures MUST be saved to AUTO_LOG_DIR before calling log_analysis —
  files outside AUTO_LOG_DIR are inaccessible to the host.
  Pass only the filename (not the full path), e.g. ["fit.png"].
- kind (str): The sort of record this is — one of "analysis" (default),
  "hypothesis", "decision", "debug", "failed", or "observation". See
  "Log everything" above. Use it so failures and reasoning are distinguishable
  from successful results in the log.

Full example — save this as /agent/shared/scripts/fit_voltage_sweep.py, then run it:

    import sys; sys.path.insert(0, "/agent/workspace")
    from auto_log_client import log_analysis, AUTO_LOG_DIR
    import json, h5py, numpy as np, matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1. Load raw data from the ELN log directory
    rec = json.load(open(f"{{AUTO_LOG_DIR}}/batch_20260522_111100_000001.json"))
    voltages = np.array([e["result"]["voltage"] for e in rec["experiments"]])
    with h5py.File(f"{{AUTO_LOG_DIR}}/batch_20260522_111100_000001.h5") as f:
        powers = np.array([f[e["id"] + "/power"][:] for e in rec["experiments"]])

    # 2. Fit
    slope, intercept = np.polyfit(voltages, powers, 1)
    residuals = powers - (slope * voltages + intercept)

    # 3. Figure
    fig, ax = plt.subplots()
    ax.scatter(voltages, powers, label="data")
    ax.plot(voltages, slope * voltages + intercept, "r-", label="fit")
    ax.set_xlabel("Voltage (V)"); ax.set_ylabel("Power (W)"); ax.legend()
    fig.savefig(f"{{AUTO_LOG_DIR}}/voltage_fit.png"); plt.close(fig)

    # 4. Record — always the last step in the script
    log_analysis(
        title="Linear fit — voltage sweep",
        text=(
            "Power scales linearly with voltage across 0–3 V (R²=0.998). "
            "Slope 0.023 W/V, intercept 1.84 mW. Residuals are within noise — "
            "no nonlinearity visible. Safe to use this calibration for power control."
        ),
        data={{"slope": slope, "intercept": intercept, "residuals": residuals}},
        references=["batch_20260522_111100_000001"],
        script=open(__file__).read(),
        figures=["voltage_fit.png"],
    )

Call log_analysis liberally — at minimum:
- Whenever a script or fit FAILS (kind="failed") or you debug one (kind="debug")
- Before a sweep/optimisation, to record what you expect (kind="hypothesis")
- When you make a non-obvious choice about how to proceed (kind="decision")
- On any anomaly, negative result, or next-step note (kind="observation")
- After fitting a model, computing statistics, or processing raw arrays
- Any time you draw a conclusion from the data that is worth preserving

A session_summary.json and session_summary.zip are written automatically
when the session ends, collecting all records into a single file.
"""
    (workspace_dir / "auto_log_info.txt").write_text(content, encoding="utf-8")


def _write_auto_log_client(workspace_dir: Path, container_log_dir: str) -> None:
    """Write auto_log_client.py into the workspace for use inside Docker."""
    content = f'''\
"""Auto-generated client for experiment auto-log tools.

Import inside Docker to group tool calls into batches or record analysis results.

Example::

    import sys; sys.path.insert(0, "/agent/workspace")
    from auto_log_client import start_batch, stop_batch, log_analysis, AUTO_LOG_DIR
"""
import base64 as _base64
import io as _io
import json as _json
import os as _os
import pickle as _pickle
import shutil as _shutil
import urllib.request as _urllib_request
from datetime import datetime as _datetime, timezone as _timezone
from pathlib import Path as _Path

try:
    import h5py as _h5py
    _H5PY = True
except ImportError:
    _H5PY = False

try:
    import numpy as _np
    _NUMPY = True
except ImportError:
    _NUMPY = False

AUTO_LOG_DIR = {container_log_dir!r}

_HOST = _os.environ.get("MCP_HOST", "host.docker.internal")
_PORT = _os.environ["MCP_PORT"]
_URL = f"http://{{_HOST}}:{{_PORT}}/invoke"
_HEADERS = {{"Content-Type": "application/json"}}
_TOKEN = _os.environ.get("MCP_AUTH_TOKEN", "")
if _TOKEN:
    _HEADERS["Authorization"] = f"Bearer {{_TOKEN}}"


def _encode_arg(obj):
    if _NUMPY and isinstance(obj, _np.ndarray):
        buf = _io.BytesIO()
        _np.save(buf, obj)
        return {{"__type__": "ndarray", "data": _base64.b64encode(buf.getvalue()).decode()}}
    if _NUMPY and isinstance(obj, _np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {{k: _encode_arg(v) for k, v in obj.items()}}
    if isinstance(obj, (list, tuple)):
        return [_encode_arg(x) for x in obj]
    return obj


def _invoke(tool_name: str, **kwargs) -> str:
    body = _json.dumps({{"tool": tool_name, "args": {{k: _encode_arg(v) for k, v in kwargs.items()}}}}).encode()
    req = _urllib_request.Request(_URL, data=body, headers=_HEADERS)
    with _urllib_request.urlopen(req) as r:
        return _pickle.loads(r.read())


def start_batch(label: str, description: str = "") -> str:
    """Start collecting experiment results into a single ELN batch record.

    All tool calls until stop_batch() are grouped into one merged record
    instead of creating individual records per call.

    Use this when running parameter sweeps, optimisation loops, multi-step
    protocols, or repeated measurements — any set of calls that logically
    form one experiment.

    Args:
        label: Short label, e.g. "Voltage sweep 0-5 V".
        description: Optional longer description of the batch.
    """
    return _invoke("start_batch", label=label, description=description)


def stop_batch() -> str:
    """Finalise the active batch and write a merged ELN record to disk.

    Returns a summary string with the output file path and experiment count.
    """
    return _invoke("stop_batch")


def log_analysis(
    title: str,
    text: str = "",
    data: dict = None,
    references: list = None,
    script: str = "",
    figures: list = None,
    kind: str = "analysis",
) -> str:
    """Record analysis results as an ELN entry and push to Kadi4Mat if configured.

    Args:
        title: Short title, e.g. "Linear fit of voltage sweep".
        text: Free-text narrative, observations, or conclusions (markdown OK).
        data: Dict of analysis results. numpy arrays are saved to HDF5
              automatically. Scalars and strings are stored as JSON metadata.
              **Note:** values must be JSON-serializable or numpy arrays —
              other types (DataFrames, arbitrary objects) are not supported.
        references: List of exp_*/batch_*/analysis_* IDs this analysis is
                    based on.
        script: Python source code used to produce this analysis.
        figures: Filenames of figures already saved to AUTO_LOG_DIR.
                 **Note:** figures must be saved to AUTO_LOG_DIR before calling
                 log_analysis — files outside AUTO_LOG_DIR are not accessible.
                 Pass only the filename, not the full path, e.g. ``["fit.png"]``.
        kind: What sort of record this is. One of "analysis" (default, a
              successful result), "hypothesis", "decision", "debug" (a
              debugging step or failed-then-fixed iteration), "failed" (an
              attempt that did not succeed), or "observation". Record failures
              and debug steps too — a failed attempt is data, not noise.

    Returns:
        Confirmation string with the output file name.

    Example::

        from auto_log_client import log_analysis, AUTO_LOG_DIR
        import numpy as np, matplotlib.pyplot as plt

        slope, intercept = np.polyfit(voltages, powers, 1)
        residuals = powers - (slope * voltages + intercept)

        fig, ax = plt.subplots()
        ax.scatter(voltages, powers)
        ax.plot(voltages, slope * voltages + intercept, "r-")
        fig.savefig(f"{{AUTO_LOG_DIR}}/fit.png")
        plt.close(fig)

        log_analysis(
            title="Linear fit",
            text="Power is linear with voltage (R²=0.998).",
            data={{"slope": slope, "residuals": residuals}},
            references=["exp_20260522_111149_616781"],
            script=open(__file__).read(),
            figures=["fit.png"],
        )
    """
    return _invoke("log_analysis",
        title=title, text=text, data=data or {{}},
        references=references or [], script=script,
        figures=figures or [], kind=kind,
    )
'''
    (workspace_dir / "auto_log_client.py").write_text(content, encoding="utf-8")


def _validate_config(config: SessionConfig) -> None:
    """Validate the session config and exit with a helpful error if invalid."""
    if not config.tools_file.exists():
        console.print(f"[bold red]Tools file not found: {config.tools_file}[/bold red]")
        raise typer.Exit(1)
    if config.tools_file.suffix != ".py":
        console.print("[bold red]Tools file must be a .py file.[/bold red]")
        raise typer.Exit(1)
    if "kadi4mat" in config.predefined_servers and not config.kadi4mat_project:
        console.print(
            "[bold red]--project is required when using --server kadi4mat.[/bold red]"
        )
        raise typer.Exit(1)
    if config.context_dir and not config.context_dir.exists():
        console.print(
            f"[bold red]Context directory not found: {config.context_dir}[/bold red]"
        )
        raise typer.Exit(1)
    if config.requirements_file and not config.requirements_file.exists():
        console.print(
            f"[bold red]Requirements file not found: {config.requirements_file}[/bold red]"
        )
        raise typer.Exit(1)


# ======================================================================
# Entry point
# ======================================================================

if __name__ == "__main__":
    app()
