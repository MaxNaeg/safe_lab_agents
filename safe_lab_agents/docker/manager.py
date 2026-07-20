"""Docker container lifecycle management for agent sessions.

Handles creating, starting, stopping, committing, and resuming containers.
Volume mounts connect the agent's workspace, context, and shared directories
to the host filesystem.
"""

from __future__ import annotations

import logging
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Optional

import docker
import requests
from docker.models.containers import Container
from rich.console import Console

from safe_lab_agents.agents.base import BaseAgent
from safe_lab_agents.config import SessionConfig
from safe_lab_agents.docker.build import ensure_base_image
from safe_lab_agents.docker.runtime import (
    container_cli,
    find_docker_desktop_windows,
    podman_windows_gateway_ip,
)

logger = logging.getLogger(__name__)

# Docker image prefix for committed session images.
_SESSION_IMAGE_PREFIX = "safe-lab-agents-session-"

# Seconds to wait for Docker to become ready after launching it.
_DOCKER_START_TIMEOUT = 60

# Container hardening applied to every container we create — defense-in-depth
# on top of the non-root ``agent`` user baked into the images. The agent is
# untrusted, so we:
#   - drop all Linux capabilities (bash/python/HTTP need none; add back via
#     cap_add only if a tool genuinely requires one, e.g. NET_RAW for ping),
#   - forbid gaining privileges via setuid binaries (no-new-privileges), and
#   - cap the live process count to contain fork bombs (PIDs are a host-global
#     resource, so an unbounded container can wedge the whole host).
_HARDENING_CAP_DROP = ["ALL"]
_HARDENING_SECURITY_OPT = ["no-new-privileges:true"]
_HARDENING_PIDS_LIMIT = 512

# Capabilities added back on top of ``cap_drop ALL``. Every container starts
# as root only so the entrypoint can run its root phase, then drops to the
# unprivileged ``agent`` user via setpriv — which needs SETUID/SETGID.
# Session containers additionally get NET_ADMIN so the entrypoint can install
# the egress firewall (scope host access to the MCP port) before the drop.
# After the drop the capabilities survive only in the bounding set, where the
# ``no-new-privileges`` flag makes them unreacquirable.
_LOGIN_CAP_ADD = ["SETUID", "SETGID"]
_SESSION_CAP_ADD = ["NET_ADMIN", *_LOGIN_CAP_ADD]

# Floor for the default memory limit. The default is half the RAM visible to
# the container runtime, but never less than this — scientific workloads
# (numpy arrays, plots) need real headroom, and a too-small limit would turn
# ordinary analyses into OOM kills.
_MIN_MEM_LIMIT_BYTES = 2 * 1024**3


def _make_agent_writable(path: Path) -> None:
    """Recursively make a host bind-mount tree writable by the container ``agent``.

    The workspace/shared dirs are created here on the host, so they're owned by
    the host user.  On Linux a bind mount shares the host inode verbatim, and the
    in-container ``agent`` user's UID essentially never matches the host owner:

    * rootful Docker / Podman — files show up owned by the host UID, not
      ``agent``;
    * rootless Podman — the host user maps to *container-root*, so they show up
      **root-owned** inside the container.

    Either way ``agent`` falls into the "other" permission class and cannot write
    with the default ``0775``/``0644``.  We widen dirs to ``0777`` and files to
    add ``rw`` for everyone, *without* changing ownership — so the host side
    (e.g. the auto-log writer, which runs as the host user) keeps its own access.
    The mode bit is metadata on the shared inode, so the container sees it
    immediately; no in-container privilege juggling required.  (On macOS /
    Windows the mounts go through a VM file-sharing layer that already fakes
    ownership, so this is a harmless no-op there.)

    Applied recursively so pre-existing content (seeded data, files left by an
    earlier session) becomes writable too.  Best-effort: files owned by another
    user that the host can't chmod (e.g. left by a session under a *different*
    container runtime, owned by a Podman subuid) are logged and skipped rather
    than aborting the whole session.
    """
    for target in (path, *path.rglob("*")):
        try:
            if target.is_symlink():
                continue  # don't chase symlinks out of the tree
            mode = target.stat().st_mode
            if target.is_dir():
                target.chmod(0o777)
            else:
                # add rw for all, keep existing execute bits.
                target.chmod((mode & 0o777) | 0o666)
        except OSError as exc:
            logger.warning("Could not widen permissions on %s: %s", target, exc)


def _connect_or_start_docker() -> docker.DockerClient:
    """Return a Docker client, starting Docker Desktop automatically if needed.

    On macOS, tries ``open -a Docker`` to launch Docker Desktop.
    On Linux, tries ``sudo systemctl start docker``.
    On Windows, tries to start Docker Desktop via its executable.

    Waits up to :data:`_DOCKER_START_TIMEOUT` seconds for the daemon to
    become ready before giving up.

    Raises:
        RuntimeError: If Docker cannot be started or does not become ready
            within the timeout.
    """
    # Fast path — Docker is already running. from_env() is lazy and does NOT
    # contact the daemon, so it succeeds even when Docker is stopped; ping()
    # forces a real round-trip so a stopped daemon is detected here (routing us
    # into auto-start) instead of surfacing as an opaque error on the first API
    # call. A dead socket raises requests' ConnectionError, not a
    # DockerException, so both are caught.
    try:
        client = docker.from_env()
        client.ping()
        return client
    except (docker.errors.DockerException, requests.exceptions.RequestException) as exc:
        if os.environ.get("DOCKER_HOST"):
            raise RuntimeError(
                f"Could not connect to container runtime at {os.environ['DOCKER_HOST']}. "
                "If using Podman, ensure it is running (or pass --podman to start it automatically)."
            )
        # Detect permission error: Docker is running but the socket is not accessible.
        # This happens when the user is not in the 'docker' group on Linux.
        if platform.system() == "Linux" and "Permission denied" in str(exc):
            raise RuntimeError(
                "Cannot connect to Docker: permission denied on the Docker socket.\n"
                "Add your user to the 'docker' group and re-login:\n"
                "  sudo usermod -aG docker $USER\n"
                "  newgrp docker"
            ) from exc

    system = platform.system()
    logger.info("Docker is not running — attempting to start it (%s) …", system)
    print("Docker is not running — starting it automatically …", flush=True)

    try:
        if system == "Darwin":
            subprocess.Popen(["open", "-a", "Docker"])
        elif system == "Linux":
            subprocess.run(
                ["sudo", "systemctl", "start", "docker"],
                check=True,
                capture_output=True,
            )
        elif system == "Windows":
            exe = find_docker_desktop_windows()
            if exe is None:
                raise RuntimeError("Could not find Docker Desktop executable.")
            subprocess.Popen([str(exe)])
        else:
            raise RuntimeError(f"Unsupported platform: {system}")
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"Failed to start Docker: {exc}") from exc

    # Poll until the daemon responds. ping() (not just from_env(), which would
    # succeed instantly) is what actually confirms the daemon is up, so the wait
    # reflects Docker Desktop's real boot time.
    deadline = time.monotonic() + _DOCKER_START_TIMEOUT
    while time.monotonic() < deadline:
        try:
            client = docker.from_env()
            client.ping()
            print("Docker is ready.", flush=True)
            return client
        except (docker.errors.DockerException, requests.exceptions.RequestException):
            remaining = int(deadline - time.monotonic())
            print(f"  Waiting for Docker … ({remaining}s remaining)", end="\r", flush=True)
            time.sleep(2)

    raise RuntimeError(
        f"Docker did not become ready within {_DOCKER_START_TIMEOUT} seconds. "
        "Please start Docker manually and try again."
    )


class DockerManager:
    """Manage the Docker container for an agent session.

    Args:
        docker_client: An existing Docker client, or ``None`` to create one
            from the default environment.
    """

    def __init__(self, docker_client: Optional[docker.DockerClient] = None) -> None:
        self.client = docker_client or _connect_or_start_docker()

    # ------------------------------------------------------------------
    # Image management
    # ------------------------------------------------------------------

    def ensure_image(
        self,
        agent_type: str,
        requirements_file: Optional[Path] = None,
        rebuild: bool = False,
    ) -> str:
        """Ensure the base Docker image for *agent_type* is built.

        Pass ``rebuild=True`` to force a full ``--no-cache --pull`` rebuild,
        ignoring the content-hash up-to-date check (picks up upstream toolchain
        updates the hash cannot see).

        Returns the image tag.
        """
        return ensure_base_image(
            agent_type,
            requirements_file=requirements_file,
            docker_client=self.client,
            rebuild=rebuild,
        )

    # ------------------------------------------------------------------
    # Container creation
    # ------------------------------------------------------------------

    def create_container(
        self,
        config: SessionConfig,
        agent: BaseAgent,
        mcp_port: int,
        image_tag: Optional[str] = None,
        resume: bool = False,
        extra_env: Optional[dict[str, str]] = None,
        tty: bool = True,
    ) -> Container:
        """Create a new Docker container for the session.

        The container is created but **not** started.  Call
        :meth:`start_interactive` or :meth:`start_autonomous` afterwards.

        Args:
            config: Session configuration.
            agent: The agent backend instance.
            mcp_port: The TCP port of the host-side MCP server.
            image_tag: Override the Docker image tag (used when resuming from
                a committed session image).
            resume: If ``True``, use the agent's resume command.

        Returns:
            The created (but not yet started) container.
        """
        if image_tag is None:
            image_tag = ensure_base_image(
                config.agent_type,
                requirements_file=config.requirements_file,
                docker_client=self.client,
            )

        volumes = self._build_volumes(config)
        environment = agent.get_environment_variables(config, mcp_port)
        if extra_env:
            environment.update(extra_env)
        mcp_host = self._resolve_mcp_host(config)
        if mcp_host:
            environment["MCP_HOST"] = mcp_host
        # Tell the entrypoint whether to install the egress firewall before
        # dropping privileges (see firewall.sh). The host-gateway mapping
        # itself must stay: the entrypoint's MCP URL and the firewall's
        # host-IP resolution both rely on the host.docker.internal name
        # (rootless Podman does not define it on its own), and the firewall —
        # not the absence of the name — is what scopes host access.
        environment["EGRESS_LOCKDOWN"] = (
            "true" if config.egress_lockdown else "false"
        )
        command = (
            agent.get_resume_command() if resume else agent.get_entrypoint_command()
        )

        # User-configured memory/CPU limits override the runtime-sized
        # defaults applied in _containers_create (setdefault semantics).
        resource_overrides: dict = {}
        if config.mem_limit is not None:
            resource_overrides["mem_limit"] = config.mem_limit
        if config.cpu_limit is not None:
            resource_overrides["nano_cpus"] = int(config.cpu_limit * 1_000_000_000)

        container = self._containers_create(
            image=image_tag,
            name=f"safe-lab-agents-{config.name}",
            command=command,
            environment=environment,
            volumes=volumes,
            extra_hosts={"host.docker.internal": "host-gateway"},
            cap_add=_SESSION_CAP_ADD,
            tty=tty,
            stdin_open=tty,
            detach=True,
            **resource_overrides,
        )

        logger.info(
            "Created container %s (%s) from image %s",
            container.name,
            container.short_id,
            image_tag,
        )
        return container

    def create_login_container(self, image_tag: str, agent: BaseAgent) -> Container:
        """Create a throwaway TTY container that runs an interactive login then exits.

        Used by the autonomous login-bootstrap: the container runs the agent's
        login command (``claude setup-token`` for Claude Code), the user
        completes the OAuth flow over the attached PTY, and the resulting
        credential is written to a file the caller harvests via
        :meth:`read_file_from_container`.  No volumes or task env are mounted —
        its only job is to produce a credential.

        The container is created but **not** started; run it with
        :meth:`start_interactive`.

        Unlike session containers it gets no ``host.docker.internal`` mapping
        and no ``NET_ADMIN`` — the OAuth flow only needs the public internet,
        so the host is not exposed to it at all (and with ``EGRESS_LOCKDOWN``
        unset the entrypoint skips the firewall). SETUID/SETGID remain so the
        entrypoint can drop from root to the ``agent`` user.
        """
        container = self._containers_create(
            image=image_tag,
            name=f"safe-lab-agents-login-{os.getpid()}",
            command=agent.get_login_command(),
            environment={"TERM": os.environ.get("TERM", "xterm-256color")},
            cap_add=_LOGIN_CAP_ADD,
            tty=True,
            stdin_open=True,
            detach=True,
        )
        logger.info("Created login container %s (%s)", container.name, container.short_id)
        return container

    def read_file_from_container(self, container_id: str, src_path: str) -> Optional[str]:
        """Return the text contents of a file inside a (stopped) container, or None.

        Uses ``docker cp <id>:<src> -`` (tar stream to stdout) so nothing is
        written to the host filesystem.  Podman-compatible via
        :func:`container_cli`.
        """
        import io
        import tarfile

        result = subprocess.run(
            [container_cli(), "cp", f"{container_id}:{src_path}", "-"],
            capture_output=True,
        )
        if result.returncode != 0 or not result.stdout:
            logger.warning(
                "Could not read %s from container %s: %s",
                src_path,
                container_id,
                result.stderr.decode(errors="replace"),
            )
            return None
        try:
            with tarfile.open(fileobj=io.BytesIO(result.stdout)) as tar:
                member = next((m for m in tar.getmembers() if m.isfile()), None)
                if member is None:
                    return None
                extracted = tar.extractfile(member)
                return extracted.read().decode("utf-8", errors="replace") if extracted else None
        except (tarfile.TarError, OSError) as exc:
            logger.warning("Could not extract %s from tar stream: %s", src_path, exc)
            return None

    def _with_reconnect(self, operation):
        """Run an SDK *operation*, reconnecting once if the connection went stale.

        The SDK keeps a pooled connection to the runtime. While a long image
        build runs or an interactive session is open (both outside the SDK),
        that connection sits idle and the daemon may close it — notably Podman
        over an npipe/socket. The next SDK call then fails with
        ``ConnectionError`` / ``RemoteDisconnected``. Reconnect and retry once.

        *operation* is a zero-arg callable that performs its SDK work through
        ``self.client`` so the retry uses the freshly reconnected client.
        """
        try:
            return operation()
        except requests.exceptions.ConnectionError:
            logger.info(
                "Runtime connection went stale — reconnecting and retrying …"
            )
            self.client = _connect_or_start_docker()
            return operation()

    def _resource_limit_defaults(self) -> dict:
        """Return default memory/CPU limits sized from the runtime's resources.

        Sized against what the *daemon* reports (``client.info()``), not the
        physical host: on macOS/Windows that is the Docker Desktop / Podman VM
        (whose size already caps total usage — the in-container limit stops one
        session from wedging the VM), while on native Linux it is the host
        itself — the case where a runaway agent could otherwise exhaust host
        RAM mid-experiment.

        Defaults: half the visible RAM (floor :data:`_MIN_MEM_LIMIT_BYTES`,
        capped at the total) and all-but-one visible CPU core — the spared
        core keeps the host-side MCP server (which drives the real lab
        hardware) responsive. Returns ``{}`` with a warning if the runtime
        does not report its resources.
        """
        try:
            info = self.client.info()
            mem_total = int(info.get("MemTotal") or 0)
            ncpu = int(info.get("NCPU") or 0)
        except Exception as exc:
            logger.warning("Could not query runtime resources: %s", exc)
            mem_total, ncpu = 0, 0
        if mem_total <= 0 or ncpu <= 0:
            logger.warning(
                "Container runtime did not report its RAM/CPUs — creating the "
                "container without memory/CPU limits."
            )
            return {}
        return {
            "mem_limit": min(max(mem_total // 2, _MIN_MEM_LIMIT_BYTES), mem_total),
            "nano_cpus": max(1, ncpu - 1) * 1_000_000_000,
        }

    def _containers_create(self, **kwargs) -> Container:
        """Create a container, reconnecting once if the connection went stale.

        Applies the hardening defaults (drop all capabilities, no-new-privileges,
        PID cap, memory/CPU limits) to every container unless the caller
        explicitly overrides them.  ``memswap_limit`` always follows
        ``mem_limit`` so the container gets **no swap**: with swap available a
        runaway agent would not OOM inside the container but swap-thrash the
        whole host — itself a DoS.  Both Docker and Podman honour these options.

        The memory/CPU limits are availability hardening, not a security
        boundary, so they degrade gracefully: if the runtime rejects them
        (rootless Podman without cgroups-v2 delegation), the create is retried
        once without limits and a warning is printed.
        """
        kwargs.setdefault("cap_drop", _HARDENING_CAP_DROP)
        kwargs.setdefault("security_opt", _HARDENING_SECURITY_OPT)
        kwargs.setdefault("pids_limit", _HARDENING_PIDS_LIMIT)
        limits = self._resource_limit_defaults()
        if limits:
            kwargs.setdefault("mem_limit", limits["mem_limit"])
            kwargs.setdefault("nano_cpus", limits["nano_cpus"])
        if kwargs.get("mem_limit") is not None:
            # Same value, same unit string if the caller passed one — docker's
            # SDK parses both fields identically.
            kwargs.setdefault("memswap_limit", kwargs["mem_limit"])

        def _create() -> Container:
            try:
                return self.client.containers.create(**kwargs)
            except docker.errors.APIError as exc:
                limit_keys = [
                    k
                    for k in ("mem_limit", "memswap_limit", "nano_cpus")
                    if kwargs.get(k) is not None
                ]
                if not limit_keys:
                    raise
                logger.warning(
                    "Container runtime rejected the resource limits (%s) — "
                    "retrying without memory/CPU limits. A runaway agent can "
                    "then exhaust host resources; on rootless Podman, enable "
                    "cgroups-v2 delegation to make limits enforceable. Error: %s",
                    ", ".join(limit_keys),
                    exc,
                )
                for key in limit_keys:
                    kwargs.pop(key)
                return self.client.containers.create(**kwargs)

        return self._with_reconnect(_create)

    def engine_is_podman(self) -> bool:
        """Return True if the connected container engine is actually Podman.

        Used to detect when the ``docker`` endpoint is served by Podman (e.g.
        Podman Desktop's Docker-compatibility mode) so we can warn the user
        rather than silently behaving as Docker. Returns ``False`` if the
        engine cannot be identified.
        """
        try:
            version = self.client.version()
        except Exception:
            return False
        if "podman" in str(version.get("Platform", {}).get("Name", "")).lower():
            return True
        for component in version.get("Components") or []:
            if "podman" in str(component.get("Name", "")).lower():
                return True
        return False

    @staticmethod
    def _resolve_mcp_host(config: SessionConfig) -> Optional[str]:
        """Return the address the container should use to reach the host MCP server.

        Only Podman on Windows needs an override: the container lives in a WSL2
        VM behind NAT and reaches the host via the WSL default gateway, not
        ``host.docker.internal`` (which resolves to the in-VM Podman bridge).
        Everywhere else, returning ``None`` keeps the client's built-in default
        (``host.docker.internal``), which works via the bridge gateway.
        """
        if config.container_runtime != "podman" or platform.system() != "Windows":
            return None
        gateway = podman_windows_gateway_ip()
        if gateway is None:
            logger.warning(
                "Could not determine the WSL gateway IP for Podman on Windows; "
                "falling back to host.docker.internal (the container may be unable "
                "to reach the tool server)."
            )
        return gateway

    # ------------------------------------------------------------------
    # Log extraction
    # ------------------------------------------------------------------

    def copy_agent_logs(self, container_id: str, session_dir: Path, agent_type: str) -> bool:
        """Copy native agent logs from a stopped container to the host filesystem.

        Uses ``docker cp`` directly on the stopped container — no new container
        or image is created.  The destination layout matches what
        :meth:`~BaseAgent.parse_conversation_history` expects:

        * Claude Code: ``<session_dir>/logs/projects/**/*.jsonl``
        * OpenClaw:    ``<session_dir>/logs/.openclaw/**``

        Args:
            container_id: ID or name of the (stopped) container.
            session_dir: Host-side session root
                (``~/.safe_lab_agents/sessions/<name>``).
            agent_type: ``'claude-code'`` or ``'openclaw'``.

        Returns:
            ``True`` if the copy succeeded, ``False`` otherwise.
        """
        logs_dir = session_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        if agent_type == "claude-code":
            src = f"{container_id}:/home/agent/.claude/projects"
            dest = logs_dir
        elif agent_type == "openclaw":
            # Copy only the session-log subtree, not the whole ~/.openclaw.
            # The full directory also contains OpenClaw's bundled npm install
            # tree (e.g. .openclaw/npm/.../node_modules/.bin/codex), which is
            # full of symlinks. Recreating those on a Windows host fails with
            # "A required privilege is not held by the client" (the user lacks
            # SeCreateSymbolicLinkPrivilege), aborting the whole copy. The JSONL
            # logs we actually parse live under .openclaw/agents/, so copy just
            # that into a pre-created .openclaw/ to preserve the expected layout.
            dest = logs_dir / ".openclaw"
            dest.mkdir(parents=True, exist_ok=True)
            src = f"{container_id}:/home/agent/.openclaw/agents"
        else:
            logger.info("No log copy defined for agent type '%s'.", agent_type)
            return False

        result = subprocess.run(
            [container_cli(), "cp", src, str(dest)],
            capture_output=True,
        )
        if result.returncode != 0:
            logger.warning(
                "Could not copy agent logs from container %s: %s",
                container_id,
                result.stderr.decode(errors="replace"),
            )
            return False

        logger.info("Copied agent logs from container %s to %s", container_id, logs_dir)
        return True


    # ------------------------------------------------------------------
    # Running modes
    # ------------------------------------------------------------------

    def start_interactive(self, container: Container) -> None:
        """Start an already-created container interactively.

        Uses ``docker start -ai`` to hand the terminal over to the container
        via fork+exec, giving correct PTY/raw-mode behaviour for both fresh
        sessions and resumes. Blocks until the container exits.
        """
        logger.info("Starting container %s (interactive) …", container.name)

        cli = container_cli()
        cid = container.id
        if cid is None:
            raise RuntimeError("Container has no id; cannot start it.")
        if hasattr(os, "fork"):
            # Self-pipe idiom: the write end is close-on-exec (Python's default),
            # so a SUCCESSFUL execvp closes it and the parent reads EOF. On exec
            # failure the child writes the error and terminates with os._exit —
            # NOT a raised exception — so it never unwinds into the caller's
            # `finally: _cleanup()`, which in the forked child would race the
            # parent on commit/remove/MCP-shutdown (the _cleaned_up re-entrancy
            # guard is per-process and not shared across the fork).
            err_r, err_w = os.pipe()
            pid = os.fork()
            if pid == 0:  # child
                os.close(err_r)
                try:
                    os.execvp(cli, [cli, "start", "-ai", cid])
                except BaseException as exc:  # noqa: BLE001 - must never propagate
                    try:
                        os.write(err_w, str(exc).encode("utf-8", "replace")[:500])
                    except BaseException:
                        pass
                    os._exit(127)
            else:  # parent
                os.close(err_w)
                err = b""
                while True:
                    chunk = os.read(err_r, 4096)
                    if not chunk:
                        break
                    err += chunk
                os.close(err_r)
                os.waitpid(pid, 0)
                if err:
                    raise RuntimeError(
                        f"Failed to launch '{cli}': {err.decode('utf-8', 'replace')}"
                    )
        else:
            subprocess.run([cli, "start", "-ai", cid])

    def start_autonomous(
        self,
        container: Container,
        agent: BaseAgent,
        stream_log_file: Optional[Path] = None,
    ) -> None:
        """Start the container and stream its formatted output to the terminal.

        Each line of container output is passed through the agent's
        :meth:`~BaseAgent.format_autonomous_line` method before printing, so
        agents can render their native log format (e.g. ``stream-json``) in a
        human-friendly way.

        If *stream_log_file* is given, every raw JSON line is also written
        there so it can be re-parsed as conversation history later.

        Blocks until the container exits.
        """
        container.start()
        logger.info("Streaming output from container %s …", container.name)
        console = Console()
        log_fh = stream_log_file.open("a", encoding="utf-8") if stream_log_file else None
        try:
            # Buffer across chunks so we only process complete newline-delimited
            # lines. Docker delivers logs in small chunks; splitting each chunk
            # individually would break multi-byte sequences and JSON records.
            buf = ""
            for chunk in container.logs(stream=True, follow=True):
                buf += chunk.decode(errors="replace")
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    if log_fh and line.strip().startswith("{"):
                        log_fh.write(line.strip() + "\n")
                    formatted = agent.format_autonomous_line(line)
                    if formatted is not None:
                        console.print(formatted, markup=True, highlight=False)
            if buf.strip():
                if log_fh and buf.strip().startswith("{"):
                    log_fh.write(buf.strip() + "\n")
                formatted = agent.format_autonomous_line(buf)
                if formatted is not None:
                    console.print(formatted, markup=True, highlight=False)
        finally:
            if log_fh:
                log_fh.close()

    # ------------------------------------------------------------------
    # Lifecycle operations
    # ------------------------------------------------------------------

    def stop_container(self, container_id: str, timeout: int = 10) -> None:
        """Stop a running container gracefully.

        Args:
            container_id: Container ID or name.
            timeout: Seconds to wait before force-killing.
        """
        self._with_reconnect(
            lambda: self.client.containers.get(container_id).stop(timeout=timeout)
        )
        logger.info("Container %s stopped.", container_id)

    def commit_container(self, container_id: str, session_name: str) -> str:
        """Commit the container's filesystem state to a new Docker image.

        This preserves the agent's conversation state, installed packages,
        and any files created inside the container (outside of volumes).

        Args:
            container_id: Container ID or name.
            session_name: The session name used for the image tag.

        Returns:
            The image tag of the committed image.
        """
        image_tag = self._session_image_tag(session_name)
        repo, tag = image_tag.rsplit(":", 1)

        def _commit() -> None:
            container = self.client.containers.get(container_id)
            container.commit(repository=repo, tag=tag, message=f"Session {session_name}")

        self._with_reconnect(_commit)
        logger.info("Committed container %s as image %s", container_id, image_tag)
        return image_tag

    def remove_container(self, container_id: str, force: bool = False) -> None:
        """Remove a container.

        Args:
            container_id: Container ID or name.
            force: Force removal even if the container is running.
        """
        self._with_reconnect(
            lambda: self.client.containers.get(container_id).remove(force=force)
        )
        logger.info("Removed container %s", container_id)

    # ------------------------------------------------------------------
    # Session queries
    # ------------------------------------------------------------------

    @staticmethod
    def _session_image_tag(session_name: str) -> str:
        """Return the Docker image tag for a committed session.

        Docker image names must be lowercase, so the session name is
        normalised here regardless of how it was originally entered.
        """
        return f"{_SESSION_IMAGE_PREFIX}{session_name.lower()}:latest"

    def get_session_image_tag(self, session_name: str) -> Optional[str]:
        """Return the image tag for a committed session, or ``None``."""
        tag = self._session_image_tag(session_name)
        try:
            self.client.images.get(tag)
            return tag
        except docker.errors.ImageNotFound:
            return None

    def list_session_images(self) -> list[dict[str, str]]:
        """Return a list of committed session images with their tags and creation dates.

        Normalises registry prefixes (e.g. ``docker.io/library/``) so that the
        returned ``image_tag`` values match the format produced by
        :meth:`_session_image_tag` regardless of whether Docker or Podman is in use.
        """
        results: list[dict[str, str]] = []
        for image in self.client.images.list():
            for tag in (image.tags or []):
                # Podman returns fully-qualified tags such as
                # "docker.io/library/safe-lab-agents-session-foo:latest".
                # Strip everything up to and including the last "/" so we always
                # compare against the bare "name:tag" form.
                short_tag = tag.rsplit("/", 1)[-1] if "/" in tag else tag
                if short_tag.startswith(_SESSION_IMAGE_PREFIX):
                    session_name = short_tag.removeprefix(_SESSION_IMAGE_PREFIX).removesuffix(":latest")
                    results.append({
                        "session_name": session_name,
                        "image_tag": short_tag,
                        "created": image.attrs.get("Created", ""),
                    })
        return results

    def remove_session_image(self, session_name: str) -> None:
        """Remove the committed image for a session."""
        tag = self._session_image_tag(session_name)
        self.client.images.remove(tag)
        logger.info("Removed session image %s", tag)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_volumes(config: SessionConfig) -> dict[str, dict[str, str]]:
        """Construct the Docker volume mount dictionary from *config*."""
        volumes: dict[str, dict[str, str]] = {}

        # Workspace — always mounted read-write.
        config.workspace_dir.mkdir(parents=True, exist_ok=True)
        _make_agent_writable(config.workspace_dir)
        volumes[str(config.workspace_dir)] = {
            "bind": "/agent/workspace",
            "mode": "rw",
        }

        # Context directory — read-only.
        if config.context_dir is not None:
            volumes[str(config.context_dir)] = {
                "bind": "/agent/context",
                "mode": "ro",
            }

        # Shared directory — read-write.
        if config.shared_dir is not None:
            config.shared_dir.mkdir(parents=True, exist_ok=True)
            # Pre-create the scripts dir the agent is told to use (see the system
            # prompt) so it exists host-owned and world-writable from the start —
            # persisting across sessions and container runtimes — rather than
            # being created ad-hoc by the container agent, which would own it as a
            # runtime-specific UID/subuid that other runtimes can't write to.
            (config.shared_dir / "scripts").mkdir(exist_ok=True)
            _make_agent_writable(config.shared_dir)
            volumes[str(config.shared_dir)] = {
                "bind": "/agent/shared",
                "mode": "rw",
            }

        return volumes
