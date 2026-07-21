"""Build a self-contained HTML viewer for a session's conversation history.

This is the HTML counterpart to :mod:`safe_lab_agents.history.display` (which
pretty-prints to the terminal).  It renders a list of normalized
:class:`~safe_lab_agents.agents.base.ConversationEntry` objects — the same
schema produced for both ``claude-code`` and ``openclaw`` — into one
``conversation.html`` with inline CSS + vanilla JS.  All filtering, search, and
block collapsing run in the browser, so the file opens offline and can be
emailed or archived as-is.

The visual style deliberately mirrors
:mod:`safe_lab_agents.report.builder` so the conversation viewer and the
experiment-log report look like one family.
"""

from __future__ import annotations

import html
import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

from safe_lab_agents.agents.base import ConversationEntry
from safe_lab_agents.config import SessionMetadata
from safe_lab_agents.history.render_utils import guess_language, is_image_content

logger = logging.getLogger(__name__)

# Badge colours keyed by conversation role — hex equivalents of the Rich
# ``_ROLE_STYLES`` used by the terminal display.
_ROLE_COLORS = {
    "user": "#0969da",  # blue
    "assistant": "#1a7f37",  # green
    "tool_use": "#bf8700",  # amber
    "tool_result": "#0a6c74",  # teal
    "system": "#57606a",  # grey
}

# Preferred filter/checkbox order; roles outside this list sort after, A→Z.
_ROLE_ORDER = ["user", "assistant", "tool_use", "tool_result", "system"]


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _render_text(text: str) -> str:
    """Escape free text and preserve line breaks."""
    return _esc(text).replace("\n", "<br>")


def _collapsible(label: str, inner: str, open: bool = True) -> str:
    """Wrap *inner* in a ``<details>`` block with *label* as its summary."""
    open_attr = " open" if open else ""
    return (
        f"<details class='block'{open_attr}><summary>{_esc(label)}</summary>"
        f"<div class='block-body'>{inner}</div></details>"
    )


def _render_markdown(text: str) -> str:
    """Render assistant markdown to sanitized HTML.

    Assistant content is model output that can be steered by untrusted data
    (prompt injection), so python-markdown's raw-HTML passthrough would embed
    live ``<script>``/``onerror`` payloads into the report.  The rendered HTML
    is therefore passed through an allowlist sanitizer (``nh3``) that keeps
    formatting tags but strips scripts, event handlers, and other active
    content.  If either dependency is unavailable we fall back to fully escaped
    text rather than emit unsanitized HTML.
    """
    if not text:
        return ""
    try:
        import markdown  # local imports: keep the module importable without them
        import nh3
    except Exception:  # pragma: no cover - exercised only when a dep is missing
        return f"<div class='text'>{_render_text(text)}</div>"
    rendered = markdown.markdown(text, extensions=["fenced_code", "tables"])
    # Preserve the language ``class`` on code/table tags so styling survives.
    attributes = {tag: set(attrs) for tag, attrs in nh3.ALLOWED_ATTRIBUTES.items()}
    for tag in ("code", "pre", "span", "div", "table", "th", "td"):
        attributes[tag] = attributes.get(tag, set()) | {"class"}
    safe = nh3.clean(rendered, attributes=attributes)
    return f"<div class='text md'>{safe}</div>"


def _render_tool_input(tool_input: Optional[dict]) -> str:
    """Render tool arguments as a key/value table (ported from display.py)."""
    if not tool_input:
        return "<div class='dim'>(no arguments)</div>"
    rows = []
    for key, value in tool_input.items():
        if isinstance(value, str) and "\n" in value:
            cls = guess_language(key, value)
            cell = f"<pre class='lang-{cls}'>{_esc(value.strip())}</pre>"
        elif isinstance(value, (dict, list)):
            cell = f"<pre>{_esc(json.dumps(value, indent=2, default=str))}</pre>"
        else:
            cell = _esc(value)
        rows.append(f"<tr><td class='k'>{_esc(key)}</td><td>{cell}</td></tr>")
    return f"<table class='kv'>{''.join(rows)}</table>"


_IMG_DATA_RE = re.compile(r"['\"]data['\"]\s*:\s*['\"]([A-Za-z0-9+/]{20,}={0,2})['\"]")
_IMG_MIME_RE = re.compile(r"['\"]media_type['\"]\s*:\s*['\"]([\w/.+-]+)['\"]")


def _render_images(content: str) -> str:
    """Inline any base64 image blobs in *content* as ``<img>`` tags.

    Agent image reads are stored as ``{'type': 'image', 'source': {'type':
    'base64', 'data': ..., 'media_type': ...}}``.  Returns an empty string if no
    decodable image is found (caller falls back to a size summary)."""
    data_blobs = _IMG_DATA_RE.findall(content)
    mimes = _IMG_MIME_RE.findall(content)
    imgs = []
    for i, data in enumerate(data_blobs):
        mime = mimes[i] if i < len(mimes) else "image/png"
        imgs.append(
            f"<img class='figure' src='data:{_esc(mime)};base64,{data}' "
            f"alt='agent-read image'>"
        )
    return "".join(imgs)


def _render_tool_output(content: str) -> str:
    """Render tool output, inlining images and truncating large text blobs
    (ported from display.py)."""
    if not content:
        return "<div class='dim'>(no output)</div>"
    if is_image_content(content):
        imgs = _render_images(content)
        if imgs:
            return f"<div class='figures'>{imgs}</div>"
        # Fall back to a size summary when the blob can't be decoded.
        m = re.search(r"['\"]data['\"]\s*:\s*['\"]([A-Za-z0-9+/]{20,})", content)
        kb = len(m.group(1)) * 3 // 4 // 1024 if m else 0
        size_str = f"~{kb} KB" if kb else "binary"
        return f"<div class='dim'>[image {_esc(size_str)}]</div>"
    if len(content) > 2000:
        content = content[:2000] + "\n… (truncated)"
    return f"<pre>{_esc(content)}</pre>"


def _present_roles(entries: list[ConversationEntry]) -> list[str]:
    present = {e.role for e in entries}
    ordered = [r for r in _ROLE_ORDER if r in present]
    ordered += sorted(present - set(_ROLE_ORDER))
    return ordered


# ---------------------------------------------------------------------------
# Card rendering
# ---------------------------------------------------------------------------


def _render_card(entry: ConversationEntry, index: int) -> str:
    color = _ROLE_COLORS.get(entry.role, _ROLE_COLORS["system"])
    role_label = entry.role.replace("_", " ")
    timestamp = entry.timestamp.strftime("%Y-%m-%d %H:%M:%S")
    is_error = bool(entry.metadata.get("is_error"))

    if entry.role in ("tool_use", "tool_result") and entry.tool_name:
        title = f"{role_label}: {entry.tool_name}"
    else:
        title = role_label

    parts: list[str] = []
    parts.append(
        f"<div class='card-head'>"
        f"<span class='badge' style='background:{color}'>{_esc(role_label)}</span>"
        f"<span class='card-title'>{_esc(title)}</span>"
        f"<span class='card-time'>{_esc(timestamp)}</span>"
        f"</div>"
    )

    if entry.role == "assistant":
        parts.append(_collapsible("message", _render_markdown(entry.content)))
    elif entry.role == "tool_use":
        parts.append(_collapsible("arguments", _render_tool_input(entry.tool_input)))
    elif entry.role == "tool_result":
        label = "error" if is_error else "output"
        parts.append(
            _collapsible(label, _render_tool_output(entry.tool_output or entry.content))
        )
    else:  # user / system / unknown
        parts.append(
            _collapsible("message", f"<div class='text'>{_render_text(entry.content)}</div>")
        )

    search_blob = _esc(
        f"{title} {entry.content} {entry.tool_name or ''}".lower()
    )
    err_cls = " error" if is_error else ""
    return (
        f"<section class='card{err_cls}' id='entry-{index}' "
        f"data-kind='{_esc(entry.role)}' data-search='{search_blob}'>"
        f"{''.join(parts)}</section>"
    )


def _render_session_header(metadata: SessionMetadata) -> str:
    cfg = metadata.config
    rows: list[tuple[str, str]] = [
        ("Session", cfg.name),
        ("Agent", cfg.agent_type),
        ("Container", cfg.container_runtime),
        ("Status", metadata.status),
        ("Created", cfg.created_at.strftime("%Y-%m-%d %H:%M:%S")),
    ]
    if metadata.started_at:
        rows.append(("Started", metadata.started_at.strftime("%Y-%m-%d %H:%M:%S")))
    if metadata.stopped_at:
        rows.append(("Stopped", metadata.stopped_at.strftime("%Y-%m-%d %H:%M:%S")))
    rows.append(("Tools file", str(cfg.tools_file)))
    if cfg.task:
        rows.append(("Task", cfg.task))

    body = "".join(
        f"<tr><td class='k'>{_esc(k)}</td><td>{_esc(v)}</td></tr>" for k, v in rows
    )
    return (
        "<section class='card session-info'>"
        "<div class='card-head'><span class='card-title'>Session info</span></div>"
        f"<table class='kv'>{body}</table>"
        "</section>"
    )


# ---------------------------------------------------------------------------
# Page assembly
# ---------------------------------------------------------------------------

_CSS = """
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body { font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
       margin: 0; background: #f6f8fa; color: #1f2328; }
header { position: sticky; top: 0; z-index: 10; background: #24292f; color: #fff;
         padding: 12px 20px; box-shadow: 0 1px 4px rgba(0,0,0,.2); }
header h1 { margin: 0 0 8px; font-size: 18px; font-weight: 600; }
.toolbar { display: flex; flex-wrap: wrap; gap: 10px 16px; align-items: center; font-size: 13px; }
.toolbar label { display: inline-flex; align-items: center; gap: 4px; cursor: pointer; }
.toolbar input[type=search] { padding: 4px 8px; border-radius: 6px; border: 1px solid #57606a;
         background: #fff; color: #1f2328; min-width: 200px; }
.toolbar button { padding: 4px 10px; border-radius: 6px; border: 1px solid #57606a;
         background: #32383f; color: #fff; cursor: pointer; font-size: 13px; }
.count { margin-left: auto; opacity: .8; }
main { max-width: 920px; margin: 0 auto; padding: 20px; }
.card { background: #fff; border: 1px solid #d0d7de; border-radius: 10px;
        padding: 14px 16px; margin: 0 0 16px; scroll-margin-top: 110px; }
.card.error { border-color: #cf222e; }
.card-head { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
.badge { color: #fff; font-size: 11px; font-weight: 700; text-transform: uppercase;
         letter-spacing: .04em; padding: 2px 8px; border-radius: 999px; }
.card-title { font-weight: 600; font-size: 15px; }
.card-time { margin-left: auto; color: #57606a; font-size: 12px; font-variant-numeric: tabular-nums; }
.text { line-height: 1.5; margin-top: 8px; white-space: normal; }
.text.md > :first-child { margin-top: 0; }
.text.md > :last-child { margin-bottom: 0; }
.text.md pre { background: #f6f8fa; border: 1px solid #d0d7de; border-radius: 8px;
       padding: 12px; overflow: auto; font-size: 12px; line-height: 1.45; }
.text.md code { background: #eaeef2; border-radius: 4px; padding: 1px 4px;
       font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }
.text.md pre code { background: none; padding: 0; }
.text.md table { border-collapse: collapse; }
.text.md th, .text.md td { border: 1px solid #d0d7de; padding: 4px 8px; }
details.block { margin: 10px 0 0; }
details.block > summary { cursor: pointer; font-size: 11px; font-weight: 700; text-transform: uppercase;
            letter-spacing: .04em; color: #57606a; list-style: none; padding: 2px 0; user-select: none; }
details.block > summary::-webkit-details-marker { display: none; }
details.block > summary::before { content: '▸'; display: inline-block; width: 1em; color: #8c959f; }
details.block[open] > summary::before { content: '▾'; }
.block-body { margin-top: 4px; }
.block-body pre { background: #f6f8fa; border: 1px solid #d0d7de; border-radius: 8px;
       padding: 12px; overflow: auto; font-size: 12px; line-height: 1.45; }
table.kv { border-collapse: collapse; width: 100%; font-size: 13px; table-layout: fixed; }
table.kv td { border-top: 1px solid #eaeef2; padding: 3px 6px; vertical-align: top;
       overflow-wrap: anywhere; }
table.kv td.k { color: #57606a; width: 30%; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
table.kv pre { margin: 0; background: #f6f8fa; border: 1px solid #d0d7de; border-radius: 6px;
       padding: 8px; overflow: auto; font-size: 12px; }
.dim { color: #8c959f; font-style: italic; font-size: 13px; }
.figures { display: flex; flex-wrap: wrap; gap: 10px; }
img.figure { max-width: 100%; border: 1px solid #d0d7de; border-radius: 8px; }
.empty { text-align: center; color: #57606a; padding: 60px 20px; }
"""

_JS = """
const cards = Array.from(document.querySelectorAll('.card[data-kind]'));
const search = document.getElementById('search');
const kindBoxes = Array.from(document.querySelectorAll('.kind-filter'));
const countEl = document.getElementById('count');

function applyFilters() {
  const q = (search.value || '').trim().toLowerCase();
  const active = new Set(kindBoxes.filter(b => b.checked).map(b => b.value));
  let shown = 0;
  for (const card of cards) {
    const kindOk = active.has(card.dataset.kind);
    const searchOk = !q || (card.dataset.search || '').includes(q);
    const visible = kindOk && searchOk;
    card.style.display = visible ? '' : 'none';
    if (visible) shown++;
  }
  countEl.textContent = shown + ' / ' + cards.length + ' shown';
}

search.addEventListener('input', applyFilters);
kindBoxes.forEach(b => b.addEventListener('change', applyFilters));

document.getElementById('expand-all').addEventListener('click', () => {
  document.querySelectorAll('details.block').forEach(d => d.open = true);
});
document.getElementById('collapse-all').addEventListener('click', () => {
  document.querySelectorAll('details.block').forEach(d => d.open = false);
});

applyFilters();
"""


def build_conversation_html(
    entries: list[ConversationEntry],
    metadata: Optional[SessionMetadata],
    output_path: Path,
) -> Path:
    """Render *entries* to a self-contained HTML file at *output_path*.

    Returns *output_path*.  Entries with no content and no tool input/output are
    skipped (mirroring the terminal display).  Writes an empty-state page when
    nothing remains.
    """
    output_path = Path(output_path)

    entries = [
        e for e in entries if e.content.strip() or e.tool_input or e.tool_output
    ]

    title = metadata.config.name if metadata else "conversation"

    roles = _present_roles(entries)
    checkboxes = "".join(
        f"<label><input type='checkbox' class='kind-filter' value='{_esc(r)}' checked> "
        f"<span class='badge' style='background:"
        f"{_ROLE_COLORS.get(r, _ROLE_COLORS['system'])}'>{_esc(r.replace('_', ' '))}</span></label>"
        for r in roles
    )

    body_parts: list[str] = []
    if metadata is not None:
        body_parts.append(_render_session_header(metadata))
    if entries:
        body_parts.extend(_render_card(e, i) for i, e in enumerate(entries))
    else:
        body_parts.append("<div class='empty'>No conversation history found.</div>")
    body = "".join(body_parts)

    toolbar = (
        f"{checkboxes}"
        "<input type='search' id='search' placeholder='search conversation…'>"
        "<button id='expand-all'>show all</button>"
        "<button id='collapse-all'>collapse all</button>"
        "<span class='count' id='count'></span>"
    )

    page = f"""<!DOCTYPE html>
<html lang='en'>
<head>
<meta charset='utf-8'>
<meta name='viewport' content='width=device-width, initial-scale=1'>
<title>Safe Lab Agents — {_esc(title)}</title>
<style>{_CSS}</style>
</head>
<body>
<header>
<h1>Safe Lab Agents — {_esc(title)} <span style='font-weight:400;opacity:.7'>({len(entries)} entries)</span></h1>
<div class='toolbar'>{toolbar}</div>
</header>
<main>{body}</main>
<script>{_JS}</script>
</body>
</html>
"""
    output_path.write_text(page, encoding="utf-8")
    logger.info("conversation html: wrote %s (%d entries)", output_path, len(entries))
    return output_path
