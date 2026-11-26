"""Docker image building for agent containers.

Reads Dockerfile templates from the package's bundled ``dockerfiles/``
directory, optionally injects a user-supplied ``requirements.txt``, and
builds (or reuses) a Docker image tagged
``safe-lab-agents-<agent_type>:latest``.
"""

from __future__ import annotations

import hashlib
import logging
import re
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

# Build argument (declared with a pinned default in every agent Dockerfile) that
# selects the agent CLI version to install.
_AGENT_VERSION_ARG = "AGENT_VERSION"
_AGENT_VERSION_DEFAULT_RE = re.compile(
    rf"^ARG\s+{_AGENT_VERSION_ARG}=(\S+)\s*$", re.MULTILINE
)
# A version is spliced into a build command line, so accept only what a version
# number or dist-tag can look like ("2.1.332", "2026.7.1-2", "latest", "stable").
_AGENT_VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,63}$")


def validate_agent_version(version: str) -> str:
    """Return *version* if it is a plausible version number or tag, else raise."""
    if not _AGENT_VERSION_RE.fullmatch(version):
        raise ValueError(
            f"Invalid agent version {version!r}: expected a version number such as "
            "'2.1.332' or a tag such as 'latest'."
        )
    return version


def pinned_agent_version(agent_type: str) -> Optional[str]:
    """Return the agent CLI version pinned in *agent_type*'s Dockerfile.

    This is the default of the Dockerfile's ``ARG AGENT_VERSION=…`` line — the
    version installed unless a session overrides it. ``None`` if the Dockerfile
    declares no such argument.
    """
    match = _AGENT_VERSION_DEFAULT_RE.search(
        _read_packaged_file(f"Dockerfile.{agent_type}")
    )
    return match.group(1).strip("\"'") if match else None


def ensure_base_image(
    agent_type: str,
    requirements_file: Optional[Path] = None,
    docker_client: Optional[docker.DockerClient] = None,
    rebuild: bool = False,
    agent_version: Optional[str] = None,
) -> str:
    """Build the base Docker image for *agent_type* if it does not already exist.

    If the image exists and neither the Dockerfile, the requirements file nor
    the requested agent version have changed since the last build, the existing
    image is reused.

    The agent CLI version is pinned in the Dockerfile and can be overridden
    with *agent_version*; either way it is part of the content hash, so
    changing it rebuilds. The hash does NOT track upstream drift in the base
    image, the Python toolchain, or a moving agent tag such as ``latest``, so an
    up-to-date image can still carry a stale toolchain.  Pass ``rebuild=True``
    to force a rebuild that ignores the hash and additionally re-pulls the base
    image and bypasses the builder's layer cache, so the latest upstream
    versions are fetched.

    Args:
        agent_type: Agent identifier (e.g. ``"claude-code"``).
        requirements_file: Optional path to a ``requirements.txt`` with extra
            Python packages to install inside the container.
        docker_client: Docker client instance (created from env if ``None``).
        rebuild: Force a full rebuild (``--no-cache --pull``), ignoring the
            content-hash up-to-date check.
        agent_version: Agent CLI version or tag to install instead of the one
            pinned in the Dockerfile (``None`` = the pinned version).

    Returns:
        The full image tag, e.g. ``"safe-lab-agents-claude-code:latest"``.
    """
    client = docker_client or docker.from_env()
    tag = _image_tag(agent_type)

    # Compute a content hash to decide whether rebuilding is necessary.
    if agent_version is not None:
        validate_agent_version(agent_version)
    content_hash = _compute_hash(agent_type, requirements_file, agent_version)

    if not rebuild and _image_up_to_date(client, tag, content_hash):
        logger.info("Image %s is up-to-date — skipping build.", tag)
        return tag

    if rebuild:
        logger.info("Rebuilding Docker image %s (--no-cache --pull) …", tag)
    else:
        logger.info("Building Docker image %s …", tag)
    _build_image(
        agent_type, tag, content_hash, requirements_file,
        no_cache=rebuild, pull=rebuild, agent_version=agent_version,
    )
    return tag


def _image_tag(agent_type: str) -> str:
    """Return the Docker image tag for *agent_type*."""
    return f"safe-lab-agents-{agent_type}:latest"


def _compute_hash(
    agent_type: str,
    requirements_file: Optional[Path],
    agent_version: Optional[str] = None,
) -> str:
    """Compute a SHA-256 hash over the build inputs.

    Covers the Dockerfile (and with it the pinned agent version), the
    entrypoint, the firewall script, the optional requirements file, and an
    *agent_version* override when one is given.
    """
    h = hashlib.sha256()

    # Hash the Dockerfile template.
    dockerfile_name = f"Dockerfile.{agent_type}"
    dockerfile_content = _read_packaged_file(dockerfile_name)
    h.update(dockerfile_content.encode())

    # Hash the entrypoint script.
    entrypoint_name = f"entrypoint.{agent_type}.sh"
    entrypoint_content = _read_packaged_file(entrypoint_name)
    h.update(entrypoint_content.encode())

    # Hash the shared egress-firewall script (baked into every agent image).
    h.update(_read_packaged_file("firewall.sh").encode())

    # Hash the requirements file if present.
    if requirements_file is not None and requirements_file.exists():
        h.update(requirements_file.read_bytes())

    # An overridden agent version changes what gets installed without changing
    # any file above. (No override leaves the hash as it was: the pinned default
    # lives in the Dockerfile, which is already hashed.)
    if agent_version is not None:
        h.update(f"\0{_AGENT_VERSION_ARG}={agent_version}".encode())

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
    agent_version: Optional[str] = None,
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
        agent_version: Passed as the ``AGENT_VERSION`` build argument to
            override the version pinned in the Dockerfile.
    """
    with tempfile.TemporaryDirectory(prefix="safe_lab_agents_build_") as tmpdir:
        tmp = Path(tmpdir)

        # Copy Dockerfile (LF newlines — see _write_text_lf).
        dockerfile_name = f"Dockerfile.{agent_type}"
        _write_text_lf(tmp / dockerfile_name, _read_packaged_file(dockerfile_name))

        # Copy entrypoint script (LF newlines, or the shebang breaks on Windows).
        entrypoint_name = f"entrypoint.{agent_type}.sh"
        _write_text_lf(tmp / entrypoint_name, _read_packaged_file(entrypoint_name))

        # Copy the shared egress-firewall script (COPY'd by both Dockerfiles).
        _write_text_lf(tmp / "firewall.sh", _read_packaged_file("firewall.sh"))

        # Copy requirements file (or create an empty .dockerignore as fallback
        # so the COPY instruction in the Dockerfile does not fail).
        buildargs: dict[str, str] = {}
        if requirements_file is not None and requirements_file.exists():
            shutil.copy2(requirements_file, tmp / "requirements.txt")
            buildargs["REQUIREMENTS_FILE"] = "requirements.txt"
        else:
            (tmp / ".dockerignore").write_text("")
        if agent_version is not None:
            buildargs[_AGENT_VERSION_ARG] = validate_agent_version(agent_version)

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
