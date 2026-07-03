"""Tests for Docker container lifecycle management."""

from __future__ import annotations

import types
from pathlib import Path

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
        return "created-container"


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
    with a warning; the rest of the tree is still widened, nothing raises."""
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
