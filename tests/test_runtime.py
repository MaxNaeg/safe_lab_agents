"""Tests for container runtime helpers."""

from __future__ import annotations

from safe_lab_agents.docker import runtime
from safe_lab_agents.docker.runtime import (
    _CONTAINER_CLI_ENV,
    MCP_FIREWALL_RULE_NAME,
    _npipe_url,
    _parse_default_gateway,
    container_cli,
    find_docker_desktop_windows,
    is_runtime_installed,
    mcp_firewall_setup_command,
)


def test_npipe_url_converts_native_windows_path() -> None:
    r"""A native ``\\.\pipe\name`` path becomes a forward-slash npipe URL.

    docker-py's ``parse_host`` keeps only the URL path component; backslashes
    get swallowed into the discarded netloc, so the value must use forward
    slashes to survive parsing.
    """
    assert (
        _npipe_url(r"\\.\pipe\podman-machine-default")
        == "npipe:////./pipe/podman-machine-default"
    )


def test_npipe_url_strips_existing_scheme() -> None:
    """A path that already carries the ``npipe://`` scheme is normalised once."""
    assert (
        _npipe_url(r"npipe://\\.\pipe\podman-machine-default")
        == "npipe:////./pipe/podman-machine-default"
    )


def test_npipe_url_passes_through_forward_slash_path() -> None:
    """An already-canonical forward-slash path is returned unchanged."""
    assert (
        _npipe_url("//./pipe/podman-machine-default")
        == "npipe:////./pipe/podman-machine-default"
    )


def test_container_cli_defaults_to_docker(monkeypatch) -> None:
    """Without the env var set, the CLI binary defaults to docker."""
    monkeypatch.delenv(_CONTAINER_CLI_ENV, raising=False)
    assert container_cli() == "docker"


def test_container_cli_honours_env_override(monkeypatch) -> None:
    """setup_podman_host sets the env var; container_cli reads it back."""
    monkeypatch.setenv(_CONTAINER_CLI_ENV, "podman")
    assert container_cli() == "podman"


def test_is_runtime_installed_true_when_on_path(monkeypatch) -> None:
    """A runtime whose CLI binary is on PATH is reported as installed."""
    monkeypatch.setattr(runtime.platform, "system", lambda: "Linux")
    monkeypatch.setattr(runtime.shutil, "which", lambda b: f"/usr/bin/{b}")
    assert is_runtime_installed("docker") is True
    assert is_runtime_installed("podman") is True


def test_is_runtime_installed_false_when_absent_non_windows(monkeypatch) -> None:
    """On non-Windows, a runtime not on PATH is reported as not installed."""
    monkeypatch.setattr(runtime.platform, "system", lambda: "Linux")
    monkeypatch.setattr(runtime.shutil, "which", lambda b: None)
    assert is_runtime_installed("podman") is False


def test_is_runtime_installed_windows_falls_back_to_search(monkeypatch) -> None:
    """On Windows, a missing PATH binary falls back to the install-location search."""
    monkeypatch.setattr(runtime.platform, "system", lambda: "Windows")
    monkeypatch.setattr(runtime.shutil, "which", lambda b: None)
    monkeypatch.setattr(runtime, "_find_podman_windows", lambda: object())
    monkeypatch.setattr(runtime, "find_docker_desktop_windows", lambda: None)
    assert is_runtime_installed("podman") is True
    assert is_runtime_installed("docker") is False


def test_parse_default_gateway_extracts_via_ip() -> None:
    """The gateway IP is taken from the token after 'via'."""
    out = "default via 172.26.80.1 dev eth0 proto kernel\n"
    assert _parse_default_gateway(out) == "172.26.80.1"


def test_parse_default_gateway_returns_none_without_route() -> None:
    """Unparseable output yields None so the caller can fall back."""
    assert _parse_default_gateway("") is None
    assert _parse_default_gateway("10.88.0.0/16 dev eth0 proto kernel") is None


def test_mcp_firewall_command_uses_discovered_alias() -> None:
    """The rule is scoped to the given WSL adapter alias."""
    cmd = mcp_firewall_setup_command("vEthernet (WSL (Hyper-V firewall))")
    assert MCP_FIREWALL_RULE_NAME in cmd
    assert "-InterfaceAlias 'vEthernet (WSL (Hyper-V firewall))'" in cmd
    assert "-Direction Inbound" in cmd


def test_mcp_firewall_command_falls_back_to_default_alias() -> None:
    """When the alias is unknown, the literal 'vEthernet (WSL)' is used."""
    cmd = mcp_firewall_setup_command(None)
    assert "-InterfaceAlias 'vEthernet (WSL)'" in cmd


def test_mcp_firewall_command_recreate_removes_before_adding() -> None:
    """Recreate mode removes the stale rule before adding the re-bound one."""
    cmd = mcp_firewall_setup_command("vEthernet (WSL)", recreate=True)
    assert "Remove-NetFirewallRule" in cmd
    assert cmd.index("Remove-NetFirewallRule") < cmd.index("New-NetFirewallRule")


def test_firewall_status_missing_when_no_rule(monkeypatch) -> None:
    """No rule by that name → 'missing'."""
    monkeypatch.setattr(runtime, "_powershell", lambda *a, **k: "MISSING")
    assert runtime.windows_firewall_rule_status("172.26.80.1") == "missing"


def test_firewall_status_stale_when_disabled(monkeypatch) -> None:
    """A disabled rule → 'stale' (no need to inspect its interface)."""
    monkeypatch.setattr(runtime, "_powershell", lambda *a, **k: "False")
    assert runtime.windows_firewall_rule_status("172.26.80.1") == "stale"


def test_firewall_status_stale_when_alias_mismatch(monkeypatch) -> None:
    """Enabled but pinned to a dead GUID alias that ≠ the live adapter → 'stale'."""
    replies = iter(
        [
            "True",  # enabled
            "a4880530-4eca-42b8-ab8a-d9449221f9ec",  # rule's stale alias
            "vEthernet (WSL (Hyper-V firewall))",  # live gateway adapter alias
        ]
    )
    monkeypatch.setattr(runtime, "_powershell", lambda *a, **k: next(replies))
    assert runtime.windows_firewall_rule_status("172.26.80.1") == "stale"


def test_firewall_status_ok_when_alias_matches(monkeypatch) -> None:
    """Enabled and bound to the adapter that holds the gateway → 'ok'."""
    replies = iter(
        [
            "True",
            "vEthernet (WSL (Hyper-V firewall))",
            "vEthernet (WSL (Hyper-V firewall))",
        ]
    )
    monkeypatch.setattr(runtime, "_powershell", lambda *a, **k: next(replies))
    assert runtime.windows_firewall_rule_status("172.26.80.1") == "ok"


def test_firewall_status_ok_when_scoped_to_any(monkeypatch) -> None:
    """A rule scoped to 'Any' interface matches everywhere → 'ok'."""
    replies = iter(["True", "Any"])
    monkeypatch.setattr(runtime, "_powershell", lambda *a, **k: next(replies))
    assert runtime.windows_firewall_rule_status("172.26.80.1") == "ok"


def test_find_docker_desktop_windows_per_user_install(tmp_path, monkeypatch) -> None:
    """The per-user %LOCALAPPDATA% Docker Desktop install is discovered."""
    exe = tmp_path / "Programs" / "DockerDesktop" / "Docker Desktop.exe"
    exe.parent.mkdir(parents=True)
    exe.write_text("")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setenv("PROGRAMFILES", str(tmp_path / "absent"))
    monkeypatch.setenv("PROGRAMFILES(X86)", str(tmp_path / "absent86"))
    assert find_docker_desktop_windows() == exe


def test_find_docker_desktop_windows_returns_none_when_absent(tmp_path, monkeypatch) -> None:
    """Returns None when no Docker Desktop launcher exists in any search location."""
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "empty"))
    monkeypatch.setenv("PROGRAMFILES", str(tmp_path / "empty-pf"))
    monkeypatch.setenv("PROGRAMFILES(X86)", str(tmp_path / "empty-pf86"))
    assert find_docker_desktop_windows() is None


def test_find_podman_linux_socket_skips_inaccessible_candidate(monkeypatch) -> None:
    """A candidate whose stat() raises PermissionError is skipped, not fatal.

    The rootful socket lives in a root-owned dir (/run/podman, mode 0700) that a
    non-root user cannot stat; the loop must fall through instead of crashing.
    """
    from pathlib import Path

    denied = Path("/run/podman/podman.sock")
    rootless = Path("/run/user/1000/podman/podman.sock")

    def fake_exists(self: Path) -> bool:
        if self == denied:
            raise PermissionError(13, "Permission denied")
        return self == rootless

    monkeypatch.setattr(runtime, "_linux_socket_candidates", lambda: [rootless, denied])
    monkeypatch.setattr(Path, "exists", fake_exists)
    monkeypatch.setattr(runtime, "_unix_socket_responsive", lambda p: True)

    assert runtime._find_podman_linux_socket() == str(rootless)


def test_find_podman_linux_socket_none_when_only_inaccessible(monkeypatch) -> None:
    """When the sole candidate can't be stat'd, return None rather than raising."""
    from pathlib import Path

    denied = Path("/run/podman/podman.sock")

    def fake_exists(self: Path) -> bool:
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(runtime, "_linux_socket_candidates", lambda: [denied])
    monkeypatch.setattr(Path, "exists", fake_exists)

    assert runtime._find_podman_linux_socket() is None
