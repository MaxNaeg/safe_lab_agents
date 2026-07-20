"""Build a self-contained HTML report from an auto-log folder.

The folder layout is produced by :mod:`safe_lab_agents.mcp.predefined.autolog`:
``exp_*.json`` / ``batch_*.json`` / ``analysis_*.json`` records, optional
``*.h5`` array files, and saved figure images.  This module turns the whole
folder into one ``report.html`` with inline CSS + vanilla JS — all filtering,
search, and script collapsing run in the browser, so the file opens offline and
can be emailed or archived as-is.
"""

from __future__ import annotations

import base64
import html
import json
import logging
import mimetypes
from datetime import datetime
from pathlib import Path
from typing import Any

from safe_lab_agents.mcp.predefined.records import is_quantity, split_quantity
from safe_lab_agents.utils import safe_under

logger = logging.getLogger(__name__)

# Badge colours keyed by the kind/type label shown on each card.  Anything not
# listed here falls back to the neutral "observation" styling.
_KIND_COLORS = {
    "analysis": "#1a7f37",  # green  — successful result
    "failed": "#cf222e",  # red    — attempt that did not succeed
    "debug": "#bf8700",  # amber  — debugging step / failed-then-fixed
    "hypothesis": "#0969da",  # blue   — intent before a measurement
    "decision": "#8250df",  # purple — rationale for a choice
    "observation": "#57606a",  # grey   — anomaly / negative result / next step
    "experiment": "#0a6c74",  # teal   — a single auto-logged tool call
    "batch": "#9a5700",  # brown  — a grouped sweep / protocol
}

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"}


# ---------------------------------------------------------------------------
# Entry loading
# ---------------------------------------------------------------------------


def _load_entries(log_dir: Path) -> list[dict]:
    """Return all ELN entries in *log_dir*, sorted by timestamp.

    Reads the individual record files (``exp_*`` / ``batch_*`` / ``analysis_*``)
    directly — they are the live source of truth and always reflect entries
    added after the original session ended (e.g. analyses logged during a
    ``resume``).  ``session_summary.json`` is only a snapshot written at session
    shutdown, so it is used solely as a fallback for archive-only folders that
    contain no individual record files.
    """
    json_files = sorted(
        [
            *log_dir.glob("exp_*.json"),
            *log_dir.glob("batch_*.json"),
            *log_dir.glob("analysis_*.json"),
        ]
    )
    entries = []
    for path in json_files:
        try:
            entries.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            logger.warning("report: could not parse %s", path.name, exc_info=True)

    if entries:
        entries.sort(key=lambda e: e.get("timestamp") or e.get("started_at") or "")
        return entries

    # Fallback: an archived folder that only kept session_summary.json.
    summary = log_dir / "session_summary.json"
    if summary.exists():
        try:
            data = json.loads(summary.read_text(encoding="utf-8"))
            return data.get("entries", [])
        except Exception:
            logger.warning("report: could not parse %s", summary.name, exc_info=True)

    return []


def _badge_for(entry: dict) -> str:
    """The kind/type label used for an entry's badge and filter group."""
    etype = entry.get("type")
    if etype == "individual":
        return "experiment"
    if etype == "batch":
        return "batch"
    # analysis records carry an explicit kind (default "analysis" for records
    # written before the kind field existed).
    return entry.get("kind") or "analysis"


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _fmt_time(ts: str) -> str:
    """Render an ISO timestamp as a compact local-ish string; pass through on failure."""
    if not ts:
        return ""
    try:
        return datetime.fromisoformat(ts).strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return ts


def _array_stats(log_dir: Path, ref: dict) -> str:
    """Best-effort 'shape × dtype (min/max/mean)' summary for an ndarray reference.

    Reads the HDF5 file lazily; never raises — falls back to shape/dtype only.
    """
    shape = "×".join(str(s) for s in ref.get("shape", [])) or "scalar"
    dtype = ref.get("dtype", "")
    unit = ref.get("unit")
    base = f"{shape} {dtype}".strip()
    if unit:
        base += f" {unit}"
    fname = ref.get("file")
    dataset = ref.get("dataset")
    if not fname or not dataset:
        return base
    # fname comes from an agent-written record; confine to log_dir so a crafted
    # absolute/../ name (or an out-of-tree symlink) can't open host files.
    h5_path = safe_under(log_dir, fname)
    if h5_path is None:
        return base
    try:
        import h5py  # local import: keep the module importable without h5py
        import numpy as np

        with h5py.File(str(h5_path), "r") as f:
            arr = np.asarray(f[dataset.lstrip("/")][()])
        if arr.size and np.issubdtype(arr.dtype, np.number):
            return f"{base} · min {arr.min():.4g}, max {arr.max():.4g}, mean {arr.mean():.4g}"
    except Exception:
        pass
    return base


def _render_value(log_dir: Path, value: Any) -> str:
    """Render a single param/result/data value as an HTML fragment.

    Recurses into plain dicts and lists so an array nested inside them still
    renders as an ``array …`` chip rather than a raw reference-dict JSON blob.
    """
    if isinstance(value, dict) and value.get("_type") == "ndarray":
        return f'<span class="chip">array {_esc(_array_stats(log_dir, value))}</span>'
    if is_quantity(value):
        v, unit, _ = split_quantity(value)
        return f"{_esc(v)}&nbsp;<span class='unit'>{_esc(unit)}</span>"
    if isinstance(value, dict):
        inner = ", ".join(
            f"<span class='mini-k'>{_esc(k)}</span> {_render_value(log_dir, v)}"
            for k, v in value.items()
        )
        return f"{{ {inner} }}" if value else "{}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_render_value(log_dir, v) for v in value) + "]"
    return _esc(value)


def _collapsible(
    label: str, inner: str, open: bool = True, extra_class: str = ""
) -> str:
    """Wrap *inner* in a ``<details>`` block with *label* as its summary.

    Open by default so the report reads top-to-bottom; the toolbar's
    expand/collapse-all buttons toggle every block at once.
    """
    cls = "block" + (f" {extra_class}" if extra_class else "")
    open_attr = " open" if open else ""
    return (
        f"<details class='{cls}'{open_attr}><summary>{_esc(label)}</summary>"
        f"<div class='block-body'>{inner}</div></details>"
    )


def _strip_param(key: str) -> str:
    """Drop the ``param_`` prefix the auto-logger adds to call arguments."""
    return key[6:] if key.startswith("param_") else key


def _inline_kv(log_dir: Path, mapping: dict) -> str:
    """Render a dict as a compact ``key value, key value`` inline fragment."""
    if not mapping:
        return "<span class='dim'>—</span>"
    return ", ".join(
        f"<span class='mini-k'>{_esc(_strip_param(k))}</span> {_render_value(log_dir, v)}"
        for k, v in mapping.items()
    )


def _render_experiments(log_dir: Path, experiments: list) -> str:
    """Render a batch's individual runs as a compact, scannable table."""
    head = (
        "<tr><th>#</th><th>tool</th><th>dur</th>"
        "<th>parameters</th><th>result</th></tr>"
    )
    rows = []
    for i, exp in enumerate(experiments, 1):
        dur = exp.get("duration_ms")
        dur_s = f"{dur} ms" if dur is not None else ""
        result = exp.get("result", {})
        result = result if isinstance(result, dict) else {"result": result}
        rows.append(
            f"<tr><td class='k'>{i}</td>"
            f"<td>{_esc(exp.get('title', ''))}</td>"
            f"<td class='dim'>{_esc(dur_s)}</td>"
            f"<td>{_inline_kv(log_dir, exp.get('parameters', {}))}</td>"
            f"<td>{_inline_kv(log_dir, result)}</td></tr>"
        )
    table = f"<table class='kv exp-table'>{head}{''.join(rows)}</table>"
    return _collapsible(f"Experiments ({len(experiments)})", table)


def _render_kv_table(log_dir: Path, title: str, mapping: dict) -> str:
    if not mapping:
        return ""
    rows = "".join(
        f"<tr><td class='k'>{_esc(k)}</td><td>{_render_value(log_dir, v)}</td></tr>"
        for k, v in mapping.items()
    )
    return _collapsible(title, f"<table class='kv'>{rows}</table>")


def _embed_figure(log_dir: Path, fname: str) -> str:
    """Return an <img>/<a> fragment with the figure base64-embedded, or a note."""
    # Figure names come from agent-written records; confine to log_dir so a
    # crafted absolute/../ name can't base64-embed arbitrary host files.
    path = safe_under(log_dir, fname)
    if path is None or not path.exists():
        return f"<div class='missing'>figure not found: {_esc(fname)}</div>"
    suffix = path.suffix.lower()
    if suffix not in _IMAGE_SUFFIXES:
        # e.g. PDF — can't inline as <img>; just name it.
        return f"<div class='missing'>figure: {_esc(fname)} (not an inline image)</div>"
    try:
        raw = path.read_bytes()
    except Exception:
        return f"<div class='missing'>could not read figure: {_esc(fname)}</div>"
    mime = mimetypes.guess_type(fname)[0] or "image/png"
    b64 = base64.b64encode(raw).decode("ascii")
    data_uri = f"data:{mime};base64,{b64}"
    return f"<a href='{data_uri}' target='_blank'><img class='figure' src='{data_uri}' alt='{_esc(fname)}'></a>"


def _resolve_reference(ref: str, ids: set[str]) -> str | None:
    """Map a reference string onto a known card id, if one matches.

    References may be an exact entry id, or an id with a tool suffix
    (e.g. ``exp_..._123456-measure``) whose card id is the bare ``exp_..._123456``.
    """
    if ref in ids:
        return ref
    for eid in ids:
        if ref.startswith(eid):
            return eid
    return None


def _render_text(text: str) -> str:
    """Escape free text and preserve line breaks."""
    return _esc(text).replace("\n", "<br>")


def _render_card(entry: dict, log_dir: Path, ids: set[str]) -> str:
    entry_id = entry.get("id", "")
    badge = _badge_for(entry)
    color = _KIND_COLORS.get(badge, _KIND_COLORS["observation"])
    title = entry.get("title") or entry.get("label") or entry_id
    ts = _fmt_time(entry.get("timestamp") or entry.get("started_at") or "")

    parts: list[str] = []

    # Header
    parts.append(
        f"<div class='card-head'>"
        f"<span class='badge' style='background:{color}'>{_esc(badge)}</span>"
        f"<span class='card-title'>{_esc(title)}</span>"
        f"<span class='card-time'>{_esc(ts)}</span>"
        f"</div>"
    )

    # Narrative text (analysis records)
    if entry.get("text"):
        parts.append(
            _collapsible(
                "text", f"<div class='text'>{_render_text(entry['text'])}</div>"
            )
        )

    # Batch metadata + the individual runs it grouped
    if entry.get("type") == "batch":
        experiments = entry.get("experiments", [])
        meta: dict = {}
        if entry.get("description"):
            meta["description"] = entry["description"]
        if entry.get("started_at"):
            meta["started"] = _fmt_time(entry["started_at"])
        if entry.get("completed_at"):
            meta["completed"] = _fmt_time(entry["completed_at"])
        meta["experiments"] = entry.get("experiment_count", len(experiments))
        parts.append(_render_kv_table(log_dir, "Batch", meta))
        if experiments:
            parts.append(_render_experiments(log_dir, experiments))

    # Parameters / results / analysis data
    parts.append(_render_kv_table(log_dir, "Parameters", entry.get("parameters", {})))
    if isinstance(entry.get("result"), dict):
        parts.append(_render_kv_table(log_dir, "Result", entry["result"]))
    parts.append(_render_kv_table(log_dir, "Data", entry.get("data", {})))

    # Figures
    figures = entry.get("figures", [])
    if figures:
        imgs = "".join(
            _embed_figure(log_dir, f["file"] if isinstance(f, dict) else f)
            for f in figures
        )
        parts.append(_collapsible("Figures", f"<div class='figures'>{imgs}</div>"))

    # References
    refs = entry.get("references", [])
    if refs:
        chips = []
        for r in refs:
            target = _resolve_reference(r, ids)
            if target:
                chips.append(f"<a class='ref' href='#{_esc(target)}'>{_esc(r)}</a>")
            else:
                chips.append(f"<span class='ref ref-dead'>{_esc(r)}</span>")
        parts.append(
            _collapsible("References", f"<div class='refs'>{''.join(chips)}</div>")
        )

    # Script (collapsed by default — usually the longest block)
    if entry.get("script"):
        parts.append(
            _collapsible(
                "script",
                f"<pre>{_esc(entry['script'])}</pre>",
                open=False,
                extra_class="script",
            )
        )

    # data-search blob powers the client-side search box
    search_blob = _esc(f"{title} {entry.get('text', '')} {badge}".lower())
    return (
        f"<section class='card' id='{_esc(entry_id)}' "
        f"data-kind='{_esc(badge)}' data-search='{search_blob}'>"
        f"{''.join(parts)}</section>"
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
.card-head { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
.badge { color: #fff; font-size: 11px; font-weight: 700; text-transform: uppercase;
         letter-spacing: .04em; padding: 2px 8px; border-radius: 999px; }
.card-title { font-weight: 600; font-size: 15px; }
.card-time { margin-left: auto; color: #57606a; font-size: 12px; font-variant-numeric: tabular-nums; }
.text { line-height: 1.5; }
details.block { margin: 10px 0 0; }
details.block > summary { cursor: pointer; font-size: 11px; font-weight: 700; text-transform: uppercase;
            letter-spacing: .04em; color: #57606a; list-style: none; padding: 2px 0; user-select: none; }
details.block > summary::-webkit-details-marker { display: none; }
details.block > summary::before { content: '▸'; display: inline-block; width: 1em; color: #8c959f; }
details.block[open] > summary::before { content: '▾'; }
.block-body { margin-top: 4px; }
.block-body pre { background: #f6f8fa; border: 1px solid #d0d7de; border-radius: 8px;
       padding: 12px; overflow: auto; font-size: 12px; line-height: 1.45; }
table.kv { border-collapse: collapse; width: 100%; font-size: 13px; }
table.kv td { border-top: 1px solid #eaeef2; padding: 3px 6px; vertical-align: top; }
table.kv td.k { color: #57606a; width: 30%; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
table.kv th { text-align: left; padding: 3px 6px; font-size: 11px; font-weight: 700; color: #57606a;
            text-transform: uppercase; letter-spacing: .04em; border-bottom: 1px solid #d0d7de; }
.exp-table td.k { width: auto; }
.mini-k { color: #57606a; font-size: 11px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
.dim { color: #8c959f; }
.unit { color: #57606a; font-size: 11px; }
.chip { background: #eaeef2; border-radius: 6px; padding: 1px 6px; font-size: 12px;
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
img.figure { max-width: 100%; border: 1px solid #d0d7de; border-radius: 8px; }
.ref { display: inline-block; margin: 2px 6px 2px 0; padding: 1px 8px; border-radius: 6px;
       background: #ddf4ff; color: #0969da; text-decoration: none; font-size: 12px;
       font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
.ref-dead { background: #eaeef2; color: #57606a; }
.missing { color: #cf222e; font-size: 12px; font-style: italic; }
.empty { text-align: center; color: #57606a; padding: 60px 20px; }
"""

_JS = """
const cards = Array.from(document.querySelectorAll('.card'));
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

// Flash a referenced card when navigated to via an anchor link.
window.addEventListener('hashchange', () => {
  const el = document.querySelector(location.hash);
  if (el) { el.style.transition = 'outline .1s'; el.style.outline = '2px solid #0969da';
            setTimeout(() => el.style.outline = '', 1200); }
});

applyFilters();
"""


def build_report(log_dir: Path, output_path: Path) -> Path:
    """Read *log_dir* and write a self-contained HTML report to *output_path*.

    Returns *output_path*.  Raises ``FileNotFoundError`` if *log_dir* does not
    exist; writes a report with an empty-state message if it contains no entries.
    """
    log_dir = Path(log_dir)
    output_path = Path(output_path)
    if not log_dir.is_dir():
        raise FileNotFoundError(f"Log directory not found: {log_dir}")

    entries = _load_entries(log_dir)
    ids = {e.get("id", "") for e in entries if e.get("id")}

    # Filter checkboxes: one per kind/type present, in a stable preferred order.
    present = {_badge_for(e) for e in entries}
    order = [
        "failed",
        "debug",
        "hypothesis",
        "decision",
        "observation",
        "analysis",
        "experiment",
        "batch",
    ]
    ordered_kinds = [k for k in order if k in present] + sorted(present - set(order))

    checkboxes = "".join(
        f"<label><input type='checkbox' class='kind-filter' value='{_esc(k)}' checked> "
        f"<span class='badge' style='background:{_KIND_COLORS.get(k, _KIND_COLORS['observation'])}'>{_esc(k)}</span></label>"
        for k in ordered_kinds
    )

    if entries:
        body = "".join(_render_card(e, log_dir, ids) for e in entries)
    else:
        body = "<div class='empty'>No ELN entries found in this folder.</div>"

    toolbar = (
        f"{checkboxes}"
        "<input type='search' id='search' placeholder='search title / text…'>"
        "<button id='expand-all'>show all</button>"
        "<button id='collapse-all'>collapse all</button>"
        "<span class='count' id='count'></span>"
    )

    page = f"""<!DOCTYPE html>
<html lang='en'>
<head>
<meta charset='utf-8'>
<meta name='viewport' content='width=device-width, initial-scale=1'>
<title>Safe Lab Agents — {_esc(log_dir.name)}</title>
<style>{_CSS}</style>
</head>
<body>
<header>
<h1>Safe Lab Agents — {_esc(log_dir.name)} <span style='font-weight:400;opacity:.7'>({len(entries)} entries)</span></h1>
<div class='toolbar'>{toolbar}</div>
</header>
<main>{body}</main>
<script>{_JS}</script>
</body>
</html>
"""
    output_path.write_text(page, encoding="utf-8")
    logger.info("report: wrote %s (%d entries)", output_path, len(entries))
    return output_path
