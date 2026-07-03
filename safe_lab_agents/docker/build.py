"""Docker image building for agent containers.

Reads Dockerfile templates from the package's bundled ``dockerfiles/``
directory, optionally injects a user-supplied ``requirements.txt``, and
builds (or reuses) a Docker image tagged
``safe-lab-agents-<agent_type>:latest``.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import subprocess
import tempfile
from importlib.resources import files as pkg_files
from pathlib import Path
from typing import Optional

import docker

from safe_lab_agents.docker.runtime import container_cli

logger = logging.getLogger(__name__)

# Label used to store a content hash so we know when to rebuild.
_HASH_LABEL = "safe_lab_agents.build_hash"


def ensure_base_image(
    agent_type: str,
    requirements_file: Optional[Path] = None,
    docker_client: Optional[docker.DockerClient] = None,
    rebuild: bool = False,
) -> str:
    """Build the base Docker image for *agent_type* if it does not already exist.

    If the image exists and neither the Dockerfile nor the requirements file
    have changed since the last build, the existing image is reused.

    The content hash only covers local build inputs (Dockerfile, entrypoint,
    requirements) — it does NOT track upstream drift in the base image or the
    unpinned agent/Python toolchain, so an up-to-date image can still carry a
    stale toolchain.  Pass ``rebuild=True`` to force a rebuild that ignores the
    hash and additionally re-pulls the base image and bypasses the builder's
    layer cache, so the latest upstream versions are fetched.

    Args:
        agent_type: Agent identifier (e.g. ``"claude-code"``).
        requirements_file: Optional path to a ``requirements.txt`` with extra
            Python packages to install inside the container.
        docker_client: Docker client instance (created from env if ``None``).
        rebuild: Force a full rebuild (``--no-cache --pull``), ignoring the
            content-hash up-to-date check.

    Returns:
        The full image tag, e.g. ``"safe-lab-agents-claude-code:latest"``.
    """
    client = docker_client or docker.from_env()
    tag = _image_tag(agent_type)

    # Compute a content hash to decide whether rebuilding is necessary.
    content_hash = _compute_hash(agent_type, requirements_file)

    if not rebuild and _image_up_to_date(client, tag, content_hash):
        logger.info("Image %s is up-to-date — skipping build.", tag)
        return tag

    if rebuild:
        logger.info("Rebuilding Docker image %s (--no-cache --pull) …", tag)
    else:
        logger.info("Building Docker image %s …", tag)
    _build_image(
        agent_type, tag, content_hash, requirements_file,
        no_cache=rebuild, pull=rebuild,
    )
    return tag


def _image_tag(agent_type: str) -> str:
    """Return the Docker image tag for *agent_type*."""
    return f"safe-lab-agents-{agent_type}:latest"


def _compute_hash(agent_type: str, requirements_file: Optional[Path]) -> str:
    """Compute a SHA-256 hash over the Dockerfile and optional requirements."""
    h = hashlib.sha256()

    # Hash the Dockerfile template.
    dockerfile_name = f"Dockerfile.{agent_type}"
    dockerfile_content = _read_packaged_file(dockerfile_name)
    h.update(dockerfile_content.encode())

    # Hash the entrypoint script.
    entrypoint_name = f"entrypoint.{agent_type}.sh"
    entrypoint_content = _read_packaged_file(entrypoint_name)
    h.update(entrypoint_content.encode())

    # Hash the requirements file if present.
    if requirements_file is not None and requirements_file.exists():
        h.update(requirements_file.read_bytes())

    return h.hexdigest()[:16]


def _image_up_to_date(client: docker.DockerClient, tag: str, content_hash: str) -> bool:
    """Return ``True`` if an image with *tag* and matching hash already exists."""
    try:
        image = client.images.get(tag)
        labels = image.labels or {}
        return labels.get(_HASH_LABEL) == content_hash
    except docker.errors.ImageNotFound:
        return False


def _read_packaged_file(filename: str) -> str:
    """Read a file from the bundled ``dockerfiles/`` package data."""
    resource = pkg_files("safe_lab_agents.docker.dockerfiles").joinpath(filename)
    return resource.read_text(encoding="utf-8")


def _write_text_lf(path: Path, content: str) -> None:
    r"""Write *content* to *path* using LF newlines regardless of host OS.

    On Windows, ``Path.write_text`` translates ``\n`` to ``\r\n``. A CRLF
    shebang makes the container try to exec ``bash\r`` and fail with
    ``/usr/bin/env: 'bash\r': No such file or directory``. Shell scripts and
    Dockerfiles copied into the build context must therefore keep Unix line
    endings even when the build runs on Windows.
    """
    normalized = content.replace("\r\n", "\n").replace("\r", "\n")
    path.write_text(normalized, newline="\n", encoding="utf-8")


def _build_image(
    agent_type: str,
    tag: str,
    content_hash: str,
    requirements_file: Optional[Path],
    no_cache: bool = False,
    pull: bool = False,
) -> None:
    """Build the Docker image in a temporary directory.

    Copies the Dockerfile, entrypoint script, and optional requirements into
    a temp directory, then invokes ``docker build`` via subprocess.

    Using the CLI instead of ``client.images.build()`` ensures BuildKit is
    used (same as running ``docker build`` manually).  The Python SDK's
    ``images.build()`` falls back to the legacy builder which OOM-kills on
    large install steps such as the Claude Code binary download.

    Args:
        no_cache: Pass ``--no-cache`` so layers (e.g. the unpinned toolchain
            installs) are rebuilt instead of served from the builder's cache.
        pull: Pass ``--pull`` so the base image is re-fetched from the registry
            rather than reused from local storage.
    """
    with tempfile.TemporaryDirectory(prefix="safe_lab_agents_build_") as tmpdir:
        tmp = Path(tmpdir)

        # Copy Dockerfile (LF newlines — see _write_text_lf).
        dockerfile_name = f"Dockerfile.{agent_type}"
        _write_text_lf(tmp / dockerfile_name, _read_packaged_file(dockerfile_name))

        # Copy entrypoint script (LF newlines, or the shebang breaks on Windows).
        entrypoint_name = f"entrypoint.{agent_type}.sh"
        _write_text_lf(tmp / entrypoint_name, _read_packaged_file(entrypoint_name))

        # Copy requirements file (or create an empty .dockerignore as fallback
        # so the COPY instruction in the Dockerfile does not fail).
        buildargs: dict[str, str] = {}
        if requirements_file is not None and requirements_file.exists():
            shutil.copy2(requirements_file, tmp / "requirements.txt")
            buildargs["REQUIREMENTS_FILE"] = "requirements.txt"
        else:
            (tmp / ".dockerignore").write_text("")

        cli = container_cli()
        cmd = [cli, "build"]
        if no_cache:
            cmd.append("--no-cache")
        if pull:
            cmd.append("--pull")
        if cli == "docker":
            # Load the build result into the local image store. Required for the
            # docker-container buildx driver, no-op for the default driver.
            # Podman's native builder writes to local storage directly and does
            # not accept --load.
            cmd.append("--load")
        cmd += [
            "--file", str(tmp / dockerfile_name),
            "--tag", tag,
            "--label", f"{_HASH_LABEL}={content_hash}",
        ]
        for key, val in buildargs.items():
            cmd += ["--build-arg", f"{key}={val}"]
        cmd.append(str(tmp))

        subprocess.run(cmd, check=True)

    logger.info("Image %s built successfully.", tag)
