# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-07-24

A container/security hardening pass, cross-platform (Windows/WSL2/Linux) fixes,
autonomous-mode bug fixes, a refactored auto-logging + units pipeline, and a new
end-to-end test harness. 79 commits, +7068/−1661 across 75 files since 0.1.0.

### Security

- **In-container egress firewall** so the agent can only reach the host on the
  MCP port; private/LAN ranges blocked, public internet open for the model API
  (fail-closed, opt out with `--no-egress-lockdown`).
- **Mandatory MCP authentication token** for the server.
- **Best-effort secret scrubbing** from committed Docker images on session exit.
- Fixed potential **code injection into generated HTML reports** and
  **arbitrary host-file inclusion in `.eln` records**.
- `~/.safe_lab_agents` and session logs made **readable only by the current
  user** (Windows included).
- **RAM/CPU resource limits** enforced per container.

### Cross-platform support

- Reliable agent-log capture on Windows/WSL2; UTF-8 stdio for the pipeline driver.
- Pruned deep plugin trees in openclaw log scrub to avoid the Windows `MAX_PATH`
  crash.
- History rendering enabled on Linux; Windows RAM detected via CIM.
- Quieted benign chmod-failure warnings on resume.

### Autonomous-mode & session fixes

- Fixed Claude Code autonomous mode: raw JSON streams, duplicated history
  entries, and un-resumable sessions.
- Fixed resuming autonomous openclaw sessions.
- Reject duplicate session names; ensure `_cleanup` runs even when the container
  fails early (and no longer runs twice).

### Auto-logging, units & reports

- Refactored autolog into an `AutoLogger` class; made it **safe for concurrent
  tool calls**; auto-`stop_batch()` on shutdown.
- Better numpy-array handling (top-level and nested), standardized `np.ndarray`
  summaries, ASCII `degrees_C`/`degC` → `DEG_C` QUDT unitCode.
- Kadi4Mat: correctly enforce per-minute rate limits; fix numeric-scalar
  stringification.
- HTML report: collapse batches by default, count functions called within a
  batch; `.eln` export fixes (mandatory license field, correct file paths,
  entry-filename matching).

### Input validation

- Reject agent-supplied bools for required ints, retry on non-int input, don't
  treat empty input as `true` for booleans, integer coercion for
  `KADI4MAT_MAX_*`.

### Tooling, logging & tests

- Proper logging with an agent-usable `--log-level` flag.
- Fixed `--update-tools` reload edge cases; MCP rebind retry with
  `SO_REUSEADDR`; pinned FastMCP version.
- **New `tests/e2e/` pipeline harness** plus substantial expansion of unit tests
  across agents, config, autolog, kadi4mat, records, report, eln export, and
  manager.

### Packaging

- Exclude demo videos/GIFs (`docs/example_videos`, ~29 MB) from the sdist.

## [0.1.0] - 2026-07-04

- Initial release.

[0.2.0]: https://github.com/MaxNaeg/safe_lab_agents/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/MaxNaeg/safe_lab_agents/releases/tag/v0.1.0
