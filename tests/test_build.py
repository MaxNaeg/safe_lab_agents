"""Tests for Docker image build helpers."""

from __future__ import annotations

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

    def fake_build(agent_type, tag, content_hash, requirements_file, no_cache=False, pull=False):
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
