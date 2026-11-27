"""Tests for Docker image build helpers."""

from __future__ import annotations

import re

import pytest

from safe_lab_agents.docker import build as build_mod
from safe_lab_agents.docker.build import _write_text_lf, ensure_base_image


def test_write_text_lf_emits_lf_only(tmp_path) -> None:
    """A string with LF newlines is written verbatim, never translated to CRLF."""
    path = tmp_path / "entrypoint.sh"
    _write_text_lf(path, "#!/usr/bin/env bash\necho hi\n")
    assert path.read_bytes() == b"#!/usr/bin/env bash\necho hi\n"


def test_write_text_lf_strips_crlf(tmp_path) -> None:
    """CRLF (and lone CR) input is normalised to LF so the shebang stays valid."""
    path = tmp_path / "entrypoint.sh"
    _write_text_lf(path, "#!/usr/bin/env bash\r\necho hi\r\n")
    assert b"\r" not in path.read_bytes()
    assert path.read_bytes() == b"#!/usr/bin/env bash\necho hi\n"


def test_rebuild_forces_build_even_when_up_to_date(monkeypatch) -> None:
    """rebuild=True ignores the up-to-date hash and rebuilds with no-cache + pull."""
    monkeypatch.setattr(build_mod, "_image_up_to_date", lambda *a, **k: True)
    calls: dict = {}

    def fake_build(
        agent_type, tag, content_hash, requirements_file,
        no_cache=False, pull=False, agent_version=None,
    ):
        calls["no_cache"] = no_cache
        calls["pull"] = pull

    monkeypatch.setattr(build_mod, "_build_image", fake_build)
    ensure_base_image("claude-code", docker_client=object(), rebuild=True)
    assert calls == {"no_cache": True, "pull": True}


def test_no_rebuild_reuses_up_to_date_image(monkeypatch) -> None:
    """Without rebuild, an up-to-date image is reused and no build runs."""
    monkeypatch.setattr(build_mod, "_image_up_to_date", lambda *a, **k: True)
    built = False

    def fake_build(*a, **k):
        nonlocal built
        built = True

    monkeypatch.setattr(build_mod, "_build_image", fake_build)
    ensure_base_image("claude-code", docker_client=object(), rebuild=False)
    assert built is False


def test_build_image_passes_no_cache_and_pull(monkeypatch) -> None:
    """_build_image forwards --no-cache and --pull to the builder CLI."""
    captured: dict = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd

    monkeypatch.setattr(build_mod.subprocess, "run", fake_run)
    build_mod._build_image("claude-code", "tag:latest", "abc123", None, no_cache=True, pull=True)
    assert "--no-cache" in captured["cmd"]
    assert "--pull" in captured["cmd"]


def test_compute_hash_covers_firewall_script(monkeypatch) -> None:
    """A change to the shared firewall.sh must change the content hash, so the
    image is rebuilt even though the Dockerfile/entrypoint are untouched."""

    def read_with(firewall_content: str):
        def fake_read(filename: str) -> str:
            return firewall_content if filename == "firewall.sh" else "static"
        return fake_read

    monkeypatch.setattr(build_mod, "_read_packaged_file", read_with("v1"))
    hash_v1 = build_mod._compute_hash("claude-code", None)
    monkeypatch.setattr(build_mod, "_read_packaged_file", read_with("v2"))
    hash_v2 = build_mod._compute_hash("claude-code", None)
    assert hash_v1 != hash_v2


def test_build_image_ships_firewall_script_in_context(monkeypatch) -> None:
    """firewall.sh is copied into the build context so the Dockerfiles' COPY works."""
    from pathlib import Path

    seen: dict = {}

    def fake_run(cmd, **kwargs):
        context_dir = Path(cmd[-1])
        seen["files"] = sorted(p.name for p in context_dir.iterdir())
        seen["firewall_bytes"] = (context_dir / "firewall.sh").read_bytes()

    monkeypatch.setattr(build_mod.subprocess, "run", fake_run)
    build_mod._build_image("claude-code", "tag:latest", "abc123", None)
    assert "firewall.sh" in seen["files"]
    # LF-only, or the shebang breaks when the build runs on Windows.
    assert b"\r" not in seen["firewall_bytes"]


def test_claude_code_image_disables_claude_ai_connectors() -> None:
    """The image ships a managed setting that turns off claude.ai connectors."""
    import json
    from pathlib import Path

    dockerfile = (
        Path(build_mod.__file__).parent / "dockerfiles" / "Dockerfile.claude-code"
    ).read_text()
    m = re.search(r"printf '([^']*)' > /etc/claude-code/managed-settings\.json", dockerfile)
    assert m, "managed-settings.json is not written in Dockerfile.claude-code"
    assert json.loads(m.group(1).replace("\\n", "")) == {"disableClaudeAiConnectors": True}


# ---- Agent version pinning -------------------------------------------------


@pytest.mark.parametrize("agent_type", ["claude-code", "openclaw"])
def test_dockerfile_pins_a_concrete_agent_version(agent_type: str) -> None:
    """Each agent Dockerfile pins a real version number by default — not a moving
    tag — so a rebuild cannot silently change the agent's behaviour."""
    pinned = build_mod.pinned_agent_version(agent_type)
    assert pinned is not None
    assert re.fullmatch(r"\d+\.\d+\.\d+(-\d+)?", pinned), pinned
    build_mod.validate_agent_version(pinned)


def test_agent_version_override_changes_hash() -> None:
    """Overriding the agent version must rebuild; not overriding must not."""
    default = build_mod._compute_hash("claude-code", None)
    assert build_mod._compute_hash("claude-code", None, None) == default
    latest = build_mod._compute_hash("claude-code", None, "latest")
    other = build_mod._compute_hash("claude-code", None, "2.1.0")
    assert len({default, latest, other}) == 3


def test_build_image_passes_agent_version_build_arg(monkeypatch) -> None:
    captured: dict = {}
    monkeypatch.setattr(
        build_mod.subprocess, "run", lambda cmd, **kw: captured.update(cmd=cmd)
    )

    build_mod._build_image("claude-code", "tag:latest", "abc", None, agent_version="latest")
    cmd = captured["cmd"]
    assert cmd[cmd.index("AGENT_VERSION=latest") - 1] == "--build-arg"

    build_mod._build_image("claude-code", "tag:latest", "abc", None)
    assert not any(str(part).startswith("AGENT_VERSION=") for part in captured["cmd"])


def test_ensure_base_image_forwards_agent_version(monkeypatch) -> None:
    """The override reaches both the up-to-date check (via the hash) and the build."""
    seen: dict = {}

    def fake_up_to_date(client, tag, content_hash):
        seen["hash"] = content_hash
        return False

    def fake_build(agent_type, tag, content_hash, requirements_file, **kwargs):
        seen["build_hash"] = content_hash
        seen["agent_version"] = kwargs["agent_version"]

    monkeypatch.setattr(build_mod, "_image_up_to_date", fake_up_to_date)
    monkeypatch.setattr(build_mod, "_build_image", fake_build)

    ensure_base_image("claude-code", docker_client=object(), agent_version="2.1.0")

    assert seen["agent_version"] == "2.1.0"
    assert seen["hash"] == seen["build_hash"]
    assert seen["hash"] == build_mod._compute_hash("claude-code", None, "2.1.0")


@pytest.mark.parametrize("bad", ["", "1.0; rm -rf /", "$(id)", "-x", "a b", "x" * 80])
def test_invalid_agent_version_is_rejected_before_building(monkeypatch, bad: str) -> None:
    """The version is spliced into a build command, so anything that is not a
    plain version or tag is refused and no build is attempted."""
    built = []
    monkeypatch.setattr(build_mod, "_image_up_to_date", lambda *a, **k: False)
    monkeypatch.setattr(build_mod, "_build_image", lambda *a, **k: built.append(1))

    with pytest.raises(ValueError):
        ensure_base_image("claude-code", docker_client=object(), agent_version=bad)
    assert built == []
