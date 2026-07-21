"""Tests for the static HTML log report builder."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from safe_lab_agents.report import build_report

# The repo ships a real auto-log folder we can render end-to-end. This is a
# committed fixture, so the test runs unconditionally: a missing folder is a
# real failure, not a reason to silently skip (a wrong path here previously made
# this end-to-end test skip on every run, hiding report-rendering regressions).
FIXTURE_DIR = (
    Path(__file__).resolve().parent.parent
    / "example_setup"
    / "shared_calibration_example"
    / "auto_log"
)


def test_build_report_from_example_folder(tmp_path: Path):
    out = tmp_path / "report.html"
    result = build_report(FIXTURE_DIR, out)

    assert result == out
    html = out.read_text(encoding="utf-8")

    # Self-contained: no external asset references.
    assert "http://" not in html
    assert "https://" not in html
    assert 'src="http' not in html

    # The agent's figure is embedded as a data URI, not linked.
    assert "data:image/png;base64," in html

    # Every entry id from the folder appears as a card anchor.
    for path in FIXTURE_DIR.glob("analysis_*.json"):
        entry_id = json.loads(path.read_text(encoding="utf-8"))["id"]
        assert f"id='{entry_id}'" in html

    # Kind/type badges and the collapsible script block are present.
    assert "class='badge'" in html
    assert "<details class='block script'>" in html


def _write_record(folder: Path, name: str, record: dict) -> None:
    (folder / name).write_text(json.dumps(record), encoding="utf-8")


def test_resolve_reference_requires_separator() -> None:
    """A reference must not resolve to a shorter card id it merely prefixes."""
    from safe_lab_agents.report.builder import _resolve_reference

    ids = {"exp_1", "exp_10"}
    assert _resolve_reference("exp_1", ids) == "exp_1"
    assert _resolve_reference("exp_10-measure", ids) == "exp_10"
    assert _resolve_reference("exp_1-measure", ids) == "exp_1"
    assert _resolve_reference("exp_99", ids) is None


def test_embed_figure_rejects_absolute_and_traversal(tmp_path: Path) -> None:
    """Figure names come from agent-written records; an absolute or ``../`` name
    must not base64-embed an arbitrary host file into the shared report."""
    from safe_lab_agents.report.builder import _embed_figure

    log_dir = tmp_path / "auto_log"
    log_dir.mkdir()
    (log_dir / "legit.png").write_bytes(b"\x89PNG\r\n")

    secret = tmp_path / "secret.png"  # image suffix so only containment blocks it
    secret.write_bytes(b"top secret host bytes")

    # A legitimate in-tree figure is embedded.
    assert "data:" in _embed_figure(log_dir, "legit.png")

    # Absolute and traversal names are refused with the not-found note, never embedded.
    for bad in (str(secret), "../secret.png", "/etc/hosts"):
        out = _embed_figure(log_dir, bad)
        assert "figure not found" in out
        assert "data:" not in out


def test_embed_figure_rejects_symlink_escaping_log_dir(tmp_path: Path) -> None:
    from safe_lab_agents.report.builder import _embed_figure

    log_dir = tmp_path / "auto_log"
    log_dir.mkdir()
    secret = tmp_path / "secret.png"
    secret.write_bytes(b"top secret host bytes")

    link = log_dir / "evil.png"
    try:
        link.symlink_to(secret)
    except (OSError, NotImplementedError):  # pragma: no cover - platform w/o symlinks
        pytest.skip("symlinks not supported on this platform")

    out = _embed_figure(log_dir, "evil.png")
    assert "figure not found" in out
    assert "data:" not in out


def test_quantity_result_renders_value_and_unit(tmp_path: Path):
    log_dir = tmp_path / "auto_log"
    log_dir.mkdir()
    _write_record(
        log_dir,
        "exp_20260101_000000_000001-measure.json",
        {
            "type": "individual",
            "id": "exp_20260101_000000_000001",
            "title": "measure",
            "timestamp": "2026-01-01T00:00:00+00:00",
            "parameters": {},
            "result": {"power": {"value": 2.5, "unit": "W"}},
        },
    )

    out = log_dir / "report.html"
    build_report(log_dir, out)
    html = out.read_text(encoding="utf-8")

    assert "2.5" in html
    assert "class='unit'>W<" in html


def test_nested_array_renders_as_chip_not_json_blob(tmp_path: Path):
    """An array nested in a dict value renders as an 'array …' chip, not a raw
    reference-dict JSON blob."""
    log_dir = tmp_path / "auto_log"
    log_dir.mkdir()
    _write_record(
        log_dir,
        "exp_20260101_000000_000001-measure.json",
        {
            "type": "individual",
            "id": "exp_20260101_000000_000001",
            "title": "measure",
            "timestamp": "2026-01-01T00:00:00+00:00",
            "parameters": {},
            "result": {
                "scan": {
                    "x": {
                        "_type": "ndarray",
                        "file": "x.h5",
                        "dataset": "/scan/x",
                        "shape": [5],
                        "dtype": "float64",
                    },
                    "n": 5,
                }
            },
        },
    )

    out = log_dir / "report.html"
    build_report(log_dir, out)
    html = out.read_text(encoding="utf-8")

    assert "chip" in html and "array 5" in html  # rendered as an array chip
    assert "_type" not in html  # not dumped as a raw reference dict


def test_failed_kind_gets_its_own_badge_and_filter(tmp_path: Path):
    log_dir = tmp_path / "auto_log"
    log_dir.mkdir()
    _write_record(
        log_dir,
        "analysis_20260101_000000_000001.json",
        {
            "type": "analysis",
            "id": "analysis_20260101_000000_000001",
            "title": "Fit did not converge",
            "kind": "failed",
            "timestamp": "2026-01-01T00:00:00+00:00",
            "text": "RuntimeError from curve_fit.",
            "script": "raise RuntimeError()",
            "references": [],
            "figures": [],
            "data": {},
        },
    )

    out = log_dir / "report.html"
    build_report(log_dir, out)
    html = out.read_text(encoding="utf-8")

    assert "data-kind='failed'" in html
    assert "value='failed'" in html  # filter checkbox exists
    assert "Fit did not converge" in html


def test_reference_resolves_to_card_anchor(tmp_path: Path):
    """A reference with a tool suffix links to the bare experiment card id."""
    log_dir = tmp_path / "auto_log"
    log_dir.mkdir()
    _write_record(
        log_dir,
        "exp_20260101_000000_000001-measure.json",
        {
            "type": "individual",
            "id": "exp_20260101_000000_000001",
            "title": "measure",
            "timestamp": "2026-01-01T00:00:00+00:00",
            "parameters": {},
            "result": {"result": 1.0},
        },
    )
    _write_record(
        log_dir,
        "analysis_20260101_000001_000001.json",
        {
            "type": "analysis",
            "id": "analysis_20260101_000001_000001",
            "title": "Derived",
            "timestamp": "2026-01-01T00:00:01+00:00",
            "text": "uses the raw measurement",
            "references": ["exp_20260101_000000_000001-measure"],
            "figures": [],
            "data": {},
        },
    )

    out = log_dir / "report.html"
    build_report(log_dir, out)
    html = out.read_text(encoding="utf-8")

    # The suffixed reference resolves to the bare card id.
    assert "href='#exp_20260101_000000_000001'" in html


def test_batch_renders_individual_experiments(tmp_path: Path):
    """A batch card must show its individual runs, not just the count."""
    log_dir = tmp_path / "auto_log"
    log_dir.mkdir()
    _write_record(
        log_dir,
        "batch_20260101_000000_000001.json",
        {
            "id": "batch_20260101_000000_000001",
            "type": "batch",
            "label": "Voltage sweep",
            "description": "0–2 V",
            "started_at": "2026-01-01T00:00:00+00:00",
            "completed_at": "2026-01-01T00:00:05+00:00",
            "experiment_count": 2,
            "experiments": [
                {
                    "id": "exp_a",
                    "title": "measure",
                    "duration_ms": 12,
                    "parameters": {"param_voltage": 0.0},
                    "result": {"power": 1.1},
                },
                {
                    "id": "exp_b",
                    "title": "measure",
                    "duration_ms": 13,
                    "parameters": {"param_voltage": 1.0},
                    "result": {"power": 2.2},
                },
            ],
        },
    )

    out = log_dir / "report.html"
    build_report(log_dir, out)
    html = out.read_text(encoding="utf-8")

    assert "Experiments (2)" in html  # the runs section is present
    assert "voltage" in html  # param_ prefix stripped, value shown
    assert "2.2" in html  # a per-run result is rendered
    assert "<table class='kv exp-table'>" in html


def test_empty_folder_writes_placeholder(tmp_path: Path):
    log_dir = tmp_path / "auto_log"
    log_dir.mkdir()
    out = log_dir / "report.html"
    build_report(log_dir, out)
    html = out.read_text(encoding="utf-8")
    assert "No ELN entries found" in html


def test_missing_folder_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        build_report(tmp_path / "does_not_exist", tmp_path / "report.html")


def test_new_analysis_added_after_session_summary_is_included(tmp_path: Path):
    """An analysis logged on resume (after session_summary.json was written) must
    appear in the report — the loader reads live record files, not the snapshot."""
    log_dir = tmp_path / "auto_log"
    log_dir.mkdir()

    # Original session ended: one analysis + a stale summary snapshot of it.
    old = {
        "type": "analysis",
        "id": "analysis_20260101_000000_000001",
        "title": "Original analysis",
        "kind": "analysis",
        "timestamp": "2026-01-01T00:00:00+00:00",
        "text": "from the first session",
        "references": [],
        "figures": [],
        "data": {},
    }
    _write_record(log_dir, "analysis_20260101_000000_000001.json", old)
    (log_dir / "session_summary.json").write_text(
        json.dumps({"type": "session_summary", "entries": [old]}), encoding="utf-8"
    )

    # Resume: a new analysis is logged but the snapshot is NOT regenerated yet.
    _write_record(
        log_dir,
        "analysis_20260102_000000_000001.json",
        {
            "type": "analysis",
            "id": "analysis_20260102_000000_000001",
            "title": "Resumed analysis",
            "kind": "observation",
            "timestamp": "2026-01-02T00:00:00+00:00",
            "text": "added during resume",
            "references": [],
            "figures": [],
            "data": {},
        },
    )

    out = log_dir / "report.html"
    build_report(log_dir, out)
    html = out.read_text(encoding="utf-8")

    assert "Original analysis" in html
    assert "Resumed analysis" in html  # would be missing if the snapshot won


def test_archive_only_folder_falls_back_to_summary(tmp_path: Path):
    """A folder containing only session_summary.json still renders its entries."""
    log_dir = tmp_path / "auto_log"
    log_dir.mkdir()
    (log_dir / "session_summary.json").write_text(
        json.dumps(
            {
                "type": "session_summary",
                "entries": [
                    {
                        "type": "analysis",
                        "id": "analysis_20260101_000000_000001",
                        "title": "Archived analysis",
                        "kind": "analysis",
                        "timestamp": "2026-01-01T00:00:00+00:00",
                        "text": "only the summary survived",
                        "references": [],
                        "figures": [],
                        "data": {},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    out = log_dir / "report.html"
    build_report(log_dir, out)
    assert "Archived analysis" in out.read_text(encoding="utf-8")
