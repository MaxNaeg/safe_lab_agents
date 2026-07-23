"""Tests for Docker container lifecycle management."""

from __future__ import annotations

import os
import types
from pathlib import Path

import docker
import pytest
import requests

from safe_lab_agents.config import SessionConfig
from safe_lab_agents.docker import manager as manager_mod
from safe_lab_agents.docker.manager import DockerManager


def _config(runtime: str) -> SessionConfig:
    return SessionConfig(
        name="s",
        agent_type="claude-code",
        tools_file=Path("tools.py"),
        workspace_dir=Path("ws"),
        container_runtime=runtime,
    )


class _FakeContainers:
    """Stand-in for ``client.containers`` that can fail its first create call."""

    def __init__(self, fail_first: bool) -> None:
        self.fail_first = fail_first
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        if self.fail_first and self.calls == 1:
            raise requests.exceptions.ConnectionError("Connection aborted.")
        return "created-container"


def test_containers_create_reconnects_on_dropped_connection(monkeypatch) -> None:
    """A stale-connection ConnectionError triggers one reconnect-and-retry."""
    stale_client = types.SimpleNamespace(containers=_FakeContainers(fail_first=True))
    fresh_client = types.SimpleNamespace(containers=_FakeContainers(fail_first=False))
    monkeypatch.setattr(manager_mod, "_connect_or_start_docker", lambda: fresh_client)

    mgr = DockerManager(docker_client=stale_client)
    result = mgr._containers_create(image="img", name="c")

    assert result == "created-container"
    assert mgr.client is fresh_client  # reconnected to a fresh client


def test_containers_create_no_retry_when_healthy(monkeypatch) -> None:
    """When the first call succeeds, no reconnect happens."""
    healthy = _FakeContainers(fail_first=False)
    client = types.SimpleNamespace(containers=healthy)

    def _should_not_be_called():
        raise AssertionError("reconnect must not run when the connection is healthy")

    monkeypatch.setattr(manager_mod, "_connect_or_start_docker", _should_not_be_called)

    mgr = DockerManager(docker_client=client)
    result = mgr._containers_create(image="img", name="c")

    assert result == "created-container"
    assert healthy.calls == 1
    assert mgr.client is client


class _CapturingContainers:
    """Records the kwargs passed to the most recent ``create`` call."""

    def __init__(self) -> None:
        self.kwargs: dict | None = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return types.SimpleNamespace(name=kwargs.get("name", "c"), short_id="abc123")


def test_containers_create_applies_hardening_defaults() -> None:
    """Every container is created non-privileged: caps dropped, no-new-privs, PID cap."""
    capturing = _CapturingContainers()
    mgr = DockerManager(docker_client=types.SimpleNamespace(containers=capturing))

    mgr._containers_create(image="img", name="c")

    assert capturing.kwargs["cap_drop"] == ["ALL"]
    assert capturing.kwargs["security_opt"] == ["no-new-privileges:true"]
    assert capturing.kwargs["pids_limit"] == 512


def test_containers_create_caller_can_override_hardening() -> None:
    """An explicit value from the caller wins over the hardening default."""
    capturing = _CapturingContainers()
    mgr = DockerManager(docker_client=types.SimpleNamespace(containers=capturing))

    mgr._containers_create(image="img", name="c", cap_drop=["ALL"], cap_add=["NET_RAW"])

    assert capturing.kwargs["cap_add"] == ["NET_RAW"]
    # The other hardening defaults are still applied.
    assert capturing.kwargs["security_opt"] == ["no-new-privileges:true"]
    assert capturing.kwargs["pids_limit"] == 512


class _StubAgent:
    """Minimal agent backend for exercising container creation."""

    def get_environment_variables(self, config, mcp_port):
        return {"MCP_PORT": str(mcp_port), "MODE": "interactive"}

    def get_entrypoint_command(self):
        return None

    def get_resume_command(self):
        return None

    def get_login_command(self):
        return ["--login"]


def _session_config(tmp_path: Path, **overrides) -> SessionConfig:
    return SessionConfig(
        name="s",
        agent_type="claude-code",
        tools_file=tmp_path / "tools.py",
        workspace_dir=tmp_path / "ws",
        container_runtime="docker",
        **overrides,
    )


def test_create_container_scopes_host_access(tmp_path: Path) -> None:
    """Session containers get the host mapping plus the caps and env the
    entrypoint needs to firewall egress (NET_ADMIN) and drop to 'agent'
    (SETUID/SETGID); the lockdown is on by default."""
    capturing = _CapturingContainers()
    mgr = DockerManager(docker_client=types.SimpleNamespace(containers=capturing))

    mgr.create_container(
        _session_config(tmp_path), _StubAgent(), mcp_port=5000, image_tag="img"
    )

    assert capturing.kwargs["extra_hosts"] == {"host.docker.internal": "host-gateway"}
    assert capturing.kwargs["cap_add"] == ["NET_ADMIN", "SETUID", "SETGID"]
    assert capturing.kwargs["cap_drop"] == ["ALL"]
    assert capturing.kwargs["environment"]["EGRESS_LOCKDOWN"] == "true"


def test_create_container_egress_lockdown_can_be_disabled(tmp_path: Path) -> None:
    """--no-egress-lockdown propagates to the entrypoint as EGRESS_LOCKDOWN=false."""
    capturing = _CapturingContainers()
    mgr = DockerManager(docker_client=types.SimpleNamespace(containers=capturing))

    mgr.create_container(
        _session_config(tmp_path, egress_lockdown=False),
        _StubAgent(),
        mcp_port=5000,
        image_tag="img",
    )

    assert capturing.kwargs["environment"]["EGRESS_LOCKDOWN"] == "false"


def test_create_login_container_is_not_host_exposed() -> None:
    """The login container gets no host.docker.internal mapping and no NET_ADMIN —
    only the SETUID/SETGID caps needed for the entrypoint's privilege drop."""
    capturing = _CapturingContainers()
    mgr = DockerManager(docker_client=types.SimpleNamespace(containers=capturing))

    mgr.create_login_container("img", _StubAgent())

    assert "extra_hosts" not in capturing.kwargs
    assert capturing.kwargs["cap_add"] == ["SETUID", "SETGID"]
    assert capturing.kwargs["cap_drop"] == ["ALL"]
    assert "EGRESS_LOCKDOWN" not in capturing.kwargs["environment"]


_GB = 1024**3


def _client_with_info(containers, mem_total: int = 16 * _GB, ncpu: int = 8):
    """Client stub whose ``info()`` reports the runtime's RAM and CPU count."""
    return types.SimpleNamespace(
        containers=containers, info=lambda: {"MemTotal": mem_total, "NCPU": ncpu}
    )


def test_containers_create_applies_default_resource_limits() -> None:
    """Defaults: half the runtime's RAM (swap disabled: memswap == mem) and
    all-but-one of its CPU cores."""
    capturing = _CapturingContainers()
    mgr = DockerManager(docker_client=_client_with_info(capturing))

    mgr._containers_create(image="img", name="c")

    assert capturing.kwargs["mem_limit"] == 8 * _GB
    assert capturing.kwargs["memswap_limit"] == 8 * _GB
    assert capturing.kwargs["nano_cpus"] == 7_000_000_000


def test_default_mem_limit_has_floor() -> None:
    """On a small runtime (3 GB), half would starve scientific workloads —
    the 2 GB floor applies (still capped at the total)."""
    capturing = _CapturingContainers()
    mgr = DockerManager(docker_client=_client_with_info(capturing, mem_total=3 * _GB, ncpu=2))

    mgr._containers_create(image="img", name="c")

    assert capturing.kwargs["mem_limit"] == 2 * _GB
    assert capturing.kwargs["memswap_limit"] == 2 * _GB
    assert capturing.kwargs["nano_cpus"] == 1_000_000_000


def test_no_limits_when_runtime_does_not_report_resources() -> None:
    """A runtime whose info() is unavailable gets no limits (warned, not fatal)."""
    capturing = _CapturingContainers()
    mgr = DockerManager(docker_client=types.SimpleNamespace(containers=capturing))

    mgr._containers_create(image="img", name="c")

    assert "mem_limit" not in capturing.kwargs
    assert "memswap_limit" not in capturing.kwargs
    assert "nano_cpus" not in capturing.kwargs


def test_create_container_resource_overrides(tmp_path: Path) -> None:
    """--mem-limit/--cpu-limit override the runtime-sized defaults; memswap
    follows the overridden mem_limit so swap stays disabled."""
    capturing = _CapturingContainers()
    mgr = DockerManager(docker_client=_client_with_info(capturing))

    mgr.create_container(
        _session_config(tmp_path, mem_limit="1g", cpu_limit=2.5),
        _StubAgent(),
        mcp_port=5000,
        image_tag="img",
    )

    assert capturing.kwargs["mem_limit"] == "1g"
    assert capturing.kwargs["memswap_limit"] == "1g"
    assert capturing.kwargs["nano_cpus"] == 2_500_000_000


class _LimitRejectingContainers:
    """Rejects the first create (as a limits-incapable runtime would), then accepts."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(dict(kwargs))
        if len(self.calls) == 1:
            raise docker.errors.APIError("crun: the requested cgroup controller is not available")
        return types.SimpleNamespace(name=kwargs.get("name", "c"), short_id="abc123")


def test_containers_create_retries_without_limits_on_rejection() -> None:
    """If the runtime rejects the limits (e.g. rootless Podman on cgroups v1),
    the create is retried once without them — availability over enforcement."""
    rejecting = _LimitRejectingContainers()
    mgr = DockerManager(docker_client=_client_with_info(rejecting))

    mgr._containers_create(image="img", name="c")

    assert len(rejecting.calls) == 2
    assert rejecting.calls[0]["mem_limit"] == 8 * _GB
    for key in ("mem_limit", "memswap_limit", "nano_cpus"):
        assert key not in rejecting.calls[1]
    # The security hardening is NOT dropped on retry.
    assert rejecting.calls[1]["cap_drop"] == ["ALL"]
    assert rejecting.calls[1]["security_opt"] == ["no-new-privileges:true"]


def test_build_volumes_makes_writable_mounts_agent_writable(tmp_path: Path) -> None:
    """Workspace and shared mounts (and their pre-existing content) are widened so
    the container's non-root 'agent' user (a mismatched UID, or container-root
    under rootless Podman) can write; the read-only context mount is untouched."""
    ws = tmp_path / "ws"
    shared = tmp_path / "sh"
    ctx = tmp_path / "ctx"
    ctx.mkdir()
    ctx.chmod(0o755)
    # Pre-existing content in shared, with perms that block the 'agent' user.
    shared.mkdir()
    (shared / "old").mkdir()
    (shared / "old").chmod(0o755)
    seeded = shared / "old" / "data.csv"
    seeded.write_text("x")
    seeded.chmod(0o644)
    config = SessionConfig(
        name="s",
        agent_type="claude-code",
        tools_file=Path("tools.py"),
        workspace_dir=ws,
        shared_dir=shared,
        context_dir=ctx,
        container_runtime="docker",
    )

    volumes = DockerManager._build_volumes(config)

    assert (ws.stat().st_mode & 0o777) == 0o777
    assert (shared.stat().st_mode & 0o777) == 0o777
    # The scripts dir is pre-created host-side and world-writable so it persists,
    # writable, across sessions and runtimes.
    assert (shared / "scripts").is_dir()
    assert ((shared / "scripts").stat().st_mode & 0o777) == 0o777
    # Pre-existing nested content is widened too (dirs 0777, files rw-for-all).
    assert ((shared / "old").stat().st_mode & 0o777) == 0o777
    assert (seeded.stat().st_mode & 0o666) == 0o666
    # Context is read-only and never widened.
    assert (ctx.stat().st_mode & 0o777) == 0o755
    assert volumes[str(ws)]["mode"] == "rw"
    assert volumes[str(ctx)]["mode"] == "ro"


def test_make_agent_writable_tolerates_partial_chmod_failure(tmp_path, monkeypatch) -> None:
    """An entry the host can't chmod (e.g. owned by a Podman subuid) is skipped
    (logged at debug); the rest of the tree is still widened, nothing raises."""
    (tmp_path / "ok").mkdir()
    blocked = tmp_path / "blocked.bin"
    blocked.write_text("x")

    real_chmod = Path.chmod

    def selective(self, mode, *a, **k):
        if self.name == "blocked.bin":
            raise PermissionError(1, "Operation not permitted")
        return real_chmod(self, mode, *a, **k)

    monkeypatch.setattr(Path, "chmod", selective)
    manager_mod._make_agent_writable(tmp_path)  # must not raise

    assert (tmp_path.stat().st_mode & 0o777) == 0o777
    assert ((tmp_path / "ok").stat().st_mode & 0o777) == 0o777


def test_with_reconnect_retries_once_on_connection_error(monkeypatch) -> None:
    """A stale-connection error reconnects once and re-runs the operation."""
    fresh_client = types.SimpleNamespace()
    monkeypatch.setattr(manager_mod, "_connect_or_start_docker", lambda: fresh_client)
    mgr = DockerManager(docker_client=types.SimpleNamespace())

    calls = {"n": 0}

    def op():
        calls["n"] += 1
        if calls["n"] == 1:
            raise requests.exceptions.ConnectionError("Connection aborted.")
        return "committed"

    assert mgr._with_reconnect(op) == "committed"
    assert calls["n"] == 2
    assert mgr.client is fresh_client


def test_with_reconnect_passes_through_when_healthy(monkeypatch) -> None:
    """A healthy operation runs once with no reconnect."""
    def _should_not_reconnect():
        raise AssertionError("must not reconnect when the operation succeeds")

    monkeypatch.setattr(manager_mod, "_connect_or_start_docker", _should_not_reconnect)
    mgr = DockerManager(docker_client=types.SimpleNamespace())
    assert mgr._with_reconnect(lambda: "ok") == "ok"


class _CommitCapture:
    """Fake client whose container records the kwargs passed to commit()."""

    def __init__(self) -> None:
        self.commit_kwargs: dict = {}

        class _Container:
            def commit(inner, **kwargs):  # noqa: N805 - fake, self is outer via closure
                self.commit_kwargs = kwargs

        self.containers = types.SimpleNamespace(get=lambda _id: _Container())


def test_commit_blanks_secret_env_keys(monkeypatch) -> None:
    """commit_container turns each secret env key into an `ENV KEY=` blank."""
    monkeypatch.setattr(manager_mod, "_connect_or_start_docker", lambda: None)
    client = _CommitCapture()
    mgr = DockerManager(docker_client=client)

    tag = mgr.commit_container("cid", "MySession", scrub_env_keys=["MCP_AUTH_TOKEN", "LLM_API_KEY"])

    assert tag == "safe-lab-agents-session-mysession:latest"
    assert client.commit_kwargs["changes"] == ["ENV MCP_AUTH_TOKEN=", "ENV LLM_API_KEY="]


def test_commit_without_scrub_keys_sends_no_changes(monkeypatch) -> None:
    """With no secret keys the commit sends changes=None (unchanged behaviour)."""
    monkeypatch.setattr(manager_mod, "_connect_or_start_docker", lambda: None)
    client = _CommitCapture()
    mgr = DockerManager(docker_client=client)

    mgr.commit_container("cid", "s")
    assert client.commit_kwargs["changes"] is None


def test_resolve_mcp_host_none_for_docker() -> None:
    """Docker needs no override — the client default (host.docker.internal) is used."""
    assert DockerManager._resolve_mcp_host(_config("docker")) is None


def test_resolve_mcp_host_none_for_podman_non_windows(monkeypatch) -> None:
    """Podman on Linux/macOS reaches the host via the bridge gateway — no override."""
    monkeypatch.setattr(manager_mod.platform, "system", lambda: "Linux")
    assert DockerManager._resolve_mcp_host(_config("podman")) is None


def test_resolve_mcp_host_returns_gateway_for_podman_windows(monkeypatch) -> None:
    """Podman on Windows injects the resolved WSL gateway IP as MCP_HOST."""
    monkeypatch.setattr(manager_mod.platform, "system", lambda: "Windows")
    monkeypatch.setattr(manager_mod, "podman_windows_gateway_ip", lambda: "172.26.80.1")
    assert DockerManager._resolve_mcp_host(_config("podman")) == "172.26.80.1"


def test_resolve_mcp_host_none_when_gateway_unresolved(monkeypatch) -> None:
    """If the gateway cannot be discovered, fall back to the client default."""
    monkeypatch.setattr(manager_mod.platform, "system", lambda: "Windows")
    monkeypatch.setattr(manager_mod, "podman_windows_gateway_ip", lambda: None)
    assert DockerManager._resolve_mcp_host(_config("podman")) is None


class _VersionClient:
    def __init__(self, version_info: dict) -> None:
        self._version_info = version_info

    def version(self) -> dict:
        return self._version_info


def test_engine_is_podman_detects_component() -> None:
    """A 'Podman Engine' component is recognised as Podman."""
    client = _VersionClient({"Components": [{"Name": "Podman Engine"}]})
    assert DockerManager(docker_client=client).engine_is_podman() is True


def test_engine_is_podman_detects_platform_name() -> None:
    """A platform name containing 'podman' is recognised as Podman."""
    client = _VersionClient({"Platform": {"Name": "Podman Engine"}})
    assert DockerManager(docker_client=client).engine_is_podman() is True


def test_engine_is_podman_false_for_real_docker() -> None:
    """A real Docker daemon is not flagged as Podman."""
    client = _VersionClient(
        {"Platform": {"Name": "Docker Engine - Community"}, "Components": [{"Name": "Engine"}]}
    )
    assert DockerManager(docker_client=client).engine_is_podman() is False


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires os.fork")
def test_start_interactive_exec_failure_raises_not_unwinds(monkeypatch) -> None:
    """A missing container CLI makes execvp fail in the forked child. The child
    must os._exit (surfaced to the parent as a RuntimeError) instead of unwinding
    into the caller's `finally: _cleanup()`, which would run teardown in the child
    and race the parent (double commit/remove/MCP-shutdown)."""
    monkeypatch.setattr(
        manager_mod, "container_cli", lambda: "safe-lab-no-such-binary-xyz"
    )
    mgr = DockerManager(docker_client=types.SimpleNamespace())
    container = types.SimpleNamespace(id="fakeid", name="test")

    with pytest.raises(RuntimeError) as exc:
        mgr.start_interactive(container)
    assert "safe-lab-no-such-binary-xyz" in str(exc.value)


def _pingable_client(ping_error: Exception | None = None):
    """A minimal docker-client stub whose ping() succeeds or raises *ping_error*."""

    def ping():
        if ping_error is not None:
            raise ping_error
        return True

    return types.SimpleNamespace(ping=ping)


def test_connect_pings_and_returns_running_client(monkeypatch) -> None:
    """The fast path pings the daemon and returns the client without auto-starting."""
    client = _pingable_client()
    monkeypatch.setattr(manager_mod.docker, "from_env", lambda: client)

    def _no_start(*a, **k):
        raise AssertionError("must not try to start Docker when it is already up")

    monkeypatch.setattr(manager_mod.subprocess, "Popen", _no_start)

    assert manager_mod._connect_or_start_docker() is client


def test_connect_treats_ping_failure_as_not_running(monkeypatch) -> None:
    """from_env() succeeds but ping() fails (stopped daemon) → routed into the
    start path, not returned as a working client."""
    bad = _pingable_client(requests.exceptions.ConnectionError("daemon down"))
    monkeypatch.setattr(manager_mod.docker, "from_env", lambda: bad)
    monkeypatch.delenv("DOCKER_HOST", raising=False)
    # Force an unsupported platform so the start path fails fast and observably.
    monkeypatch.setattr(manager_mod.platform, "system", lambda: "Plan9")

    with pytest.raises(RuntimeError, match="Unsupported platform"):
        manager_mod._connect_or_start_docker()


def test_connect_ping_failure_with_docker_host_raises(monkeypatch) -> None:
    """A ping failure with DOCKER_HOST set surfaces the remote-runtime message
    (no auto-start of a remote daemon)."""
    bad = _pingable_client(requests.exceptions.ConnectionError("daemon down"))
    monkeypatch.setattr(manager_mod.docker, "from_env", lambda: bad)
    monkeypatch.setenv("DOCKER_HOST", "tcp://remote:2375")

    with pytest.raises(RuntimeError, match="Could not connect to container runtime"):
        manager_mod._connect_or_start_docker()


class _FakeAutonomousContainer:
    """Container stand-in whose output stream is delivered as raw byte chunks.

    ``chunks`` is the exact byte sequence a robust demux would yield; the test
    controls where chunk boundaries fall (mid-line, mid-UTF-8) to prove the
    reader reassembles complete lines regardless.
    """

    name = "safe-lab-agents-s"

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks
        self.started = False
        self.attach_kwargs: dict | None = None

    def start(self) -> None:
        self.started = True

    def attach(self, **kwargs):
        self.attach_kwargs = kwargs
        return iter(self._chunks)

    # start_autonomous must not fall back to the buggy logs() demux path.
    def logs(self, **kwargs):  # pragma: no cover - guards against regression
        raise AssertionError("start_autonomous must use attach(), not logs()")


def _claude_code_agent():
    from safe_lab_agents.agents import get_agent

    return get_agent("claude-code")


def test_start_autonomous_uses_attach_not_logs() -> None:
    """The fix routes streaming through attach() (robust framing), never logs()."""
    container = _FakeAutonomousContainer([b""])
    mgr = DockerManager(docker_client=object())

    mgr.start_autonomous(container, _claude_code_agent())

    assert container.started
    assert container.attach_kwargs == {
        "stream": True,
        "logs": True,
        "stdout": True,
        "stderr": True,
    }


def test_start_autonomous_reassembles_split_lines(capsys) -> None:
    """A JSON line split across chunk boundaries (incl. mid-UTF-8) renders intact."""
    # A stream-json assistant line carrying a multi-byte char (°), deliberately
    # sliced so one chunk boundary lands in the middle of the UTF-8 sequence.
    line = (
        '{"type":"assistant","message":{"role":"assistant","content":'
        '[{"type":"text","text":"lab is 22.5 °C"}]}}'
    )
    raw = (line + "\n").encode("utf-8")
    deg_pos = raw.index(b"\xc2\xb0")  # split inside the 2-byte ° sequence
    chunks = [raw[: deg_pos + 1], raw[deg_pos + 1 : -3], raw[-3:]]

    container = _FakeAutonomousContainer(chunks)
    mgr = DockerManager(docker_client=object())

    mgr.start_autonomous(container, _claude_code_agent())

    # The reassembled JSON parses, so the assistant text is rendered to stdout
    # for the live display with its multi-byte char intact.
    assert "lab is 22.5 °C" in capsys.readouterr().out
