"""Rendering helpers shared by the terminal and HTML conversation views.

:mod:`safe_lab_agents.history.display` (Rich/terminal) and
:mod:`safe_lab_agents.history.html` (self-contained HTML) render the same
normalized :class:`~safe_lab_agents.agents.base.ConversationEntry` schema, so
the format-neutral bits — guessing a code block's language and detecting an
image tool result — live here once instead of being copy-pasted in both.
"""

from __future__ import annotations

import re

_IMAGE_TYPE_RE = re.compile(r"['\"]type['\"]\s*:\s*['\"]image['\"]")


def guess_language(key: str, value: str) -> str:
    """Best-effort syntax-highlight language for a tool-argument value.

    Keyed first on the argument name (``command`` → bash, ``code``/``script``/
    ``source`` → python), then falls back to sniffing a Python preamble.
    """
    if key == "command":
        return "bash"
    if key in ("code", "script", "source"):
        return "python"
    stripped = value.strip()
    if stripped.startswith(
        ("import ", "from ", "def ", "class ", "#!/usr/bin/env python")
    ):
        return "python"
    return "text"


def is_image_content(content: str) -> bool:
    """True if *content* looks like an image tool result (``{"type": "image"...``)."""
    return bool(_IMAGE_TYPE_RE.search(content[:200]))
