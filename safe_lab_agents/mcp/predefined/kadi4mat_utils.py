"""Pure utility functions for Kadi4Mat ELN integration.

These functions have no dependency on kadi-apy and can be tested in isolation.
They handle Kadi4Mat identifier generation (slugs, record/collection ids).

Format-neutral serialization helpers (``json_safe``, the quantity convention,
numpy-array extraction) live in :mod:`safe_lab_agents.mcp.predefined.records`.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

__all__ = [
    "slugify",
    "make_user_slug",
    "make_collection_identifier",
    "make_record_identifier",
]

# Kadi4Mat enforces a maximum identifier length of 50 characters.
# kadi-apy silently truncates longer identifiers, which would destroy
# our uniqueness guarantees.  All identifier functions respect this limit.
_KADI_MAX_IDENTIFIER_LENGTH = 50


def slugify(text: str, max_length: int = 30) -> str:
    """Convert text to a URL-safe slug.

    Lowercases, replaces non-alphanumeric characters with hyphens,
    collapses consecutive hyphens, strips leading/trailing hyphens,
    and truncates to *max_length*.
    """
    slug = text.lower()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    slug = re.sub(r"-{2,}", "-", slug)
    slug = slug.strip("-")
    return slug[:max_length].rstrip("-")


def make_user_slug(raw_username: str, max_length: int = 8) -> str:
    """Derive a short user slug from a Kadi4Mat login username.

    Takes the local part of the login (before ``@``), sanitises it,
    and truncates to *max_length* characters.  This is short enough to
    fit inside the 50-char identifier budget while still separating
    records from different users on a shared instance.

    Example::

        >>> make_user_slug("mpt240@uni-erlangen.de")
        'mpt240'
        >>> make_user_slug("john.doe@kit.edu")
        'johndoe'
    """
    # Take the local part (before @) for brevity.
    local = raw_username.split("@")[0] if "@" in raw_username else raw_username
    return slugify(local, max_length=max_length)


def make_collection_identifier(user_slug: str, project: str) -> str:
    """Build a globally unique Kadi4Mat collection identifier.

    Format: ``{user_slug}-{project_slug}``

    Guaranteed to be at most 50 characters.
    """
    # Budget: user_slug (already truncated) + 1 hyphen + project.
    project_budget = _KADI_MAX_IDENTIFIER_LENGTH - len(user_slug) - 1
    project_slug = slugify(project, max_length=max(project_budget, 4))
    return f"{user_slug}-{project_slug}"


def make_record_identifier(user_slug: str, project: str, title: str) -> str:
    """Build a globally unique Kadi4Mat record identifier.

    Format: ``{user_slug}-{project_slug}-{YYYYMMDD}-{HHMMSS}-{microseconds}``

    The microsecond-precision timestamp makes same-user collisions
    physically impossible for a single-threaded MCP server (two tool
    calls cannot be processed in the same microsecond).  Cross-user
    collisions require two users with the same short slug, the same
    short project slug, calling a tool in the same microsecond —
    effectively impossible.

    The *title* is stored in the record's ``title`` field (not length-
    limited) but is **not** included in the identifier, because the
    50-character Kadi4Mat limit does not leave room for it alongside
    the user, project, and microsecond timestamp.

    The timestamp uses **UTC** so identifiers are consistent regardless
    of the user's timezone or daylight saving time transitions.
    """
    project_slug = slugify(project, max_length=8)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    identifier = f"{user_slug}-{project_slug}-{timestamp}"

    # Safety check: ensure we never exceed the Kadi4Mat limit.
    if len(identifier) > _KADI_MAX_IDENTIFIER_LENGTH:
        identifier = identifier[:_KADI_MAX_IDENTIFIER_LENGTH]

    return identifier
