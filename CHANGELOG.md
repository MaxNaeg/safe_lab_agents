# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-11-27

### Security

- **claude.ai connectors are switched off in Claude Code containers.** With
  `copy-host-credentials`, the account's claude.ai connectors (e.g. Claude Docs,
  Gmail, Drive) showed up as MCP servers next to `experiment-tools`, bypassing
  the host MCP server and auto-logging. The container now sets
  `ENABLE_CLAUDEAI_MCP_SERVERS=false` (also on resume of older sessions) and the
  image ships a root-owned managed setting `disableClaudeAiConnectors`.

### Added

- **Autonomous resume**: `agent resume --task "…"` (or `--task-file`) continues
  a stopped session autonomously with a new task, in the same conversation.
  `agent resume` without a task stays interactive. Sessions created by earlier
  versions can only be resumed interactively (the entrypoint is baked into the
  session image); they are refused with a clear message.

- **Pinned agent versions**: each image installs a fixed, tested version of
  its agent CLI (Claude Code 2.1.332, OpenClaw 2026.7.1-2) instead of whatever
  is newest at build time, and Claude Code's self-updater is off inside the
  container. `agent start --agent-version <version|latest>` (config key
  `agent-version`) overrides the pin; the version is part of the image cache
  key.

- **`--system-prompt` / `--system-prompt-file`** replace the built-in
  "you are controlling a scientific experiment" prompt, e.g. for non-experiment
  use. The directory description, `--no-web`, autonomous-mode and tool/auto-log
  info are always kept. Kept on resume.

- **`--auto-log-prompt full|minimal|none|<path>`** sets how much the agent is
  told about the auto-log helpers (default `full`).

### Fixed

- **Claude Code resume ignored the regenerated system prompt.** Since Claude
  Code 2.1.257 the system prompt is recorded on a conversation's first request
  and reused on resume, so updated tool descriptions (and the autonomous-mode
  instruction) never reached a resumed session. Resume now asks for a fresh
  render (`--system-prompt-snapshot off`) when the installed CLI supports it.
- **Claude Code autonomous runs under accounts that disallow bypass mode.**
  Newer Claude Code versions apply settings delivered to the logged-in account;
  when those refuse `--dangerously-skip-permissions`, a headless run starts in
  `default` mode and every tool call (including the lab's MCP tools) was denied.
  Autonomous runs now also pre-approve the experiment MCP server and the
  built-in shell/file tools (web tools unless `--no-web`) via `--allowedTools`,
  and the CLI prints a notice when Claude Code reports a non-bypass mode.
- **Claude Code `effort` default.** Omitting `effort` was documented as `low`
  but never applied; it now states that Claude Code's own default is used.
- **Python tools info** only explains `Quantity` when a tool actually returns
  one.
- **HTML report** embeds each figure once (no duplicate data-URI link), with a
  lightbox to enlarge it (#6).
- **Windows-encoded prompt files.** `--task-file`, `--system-prompt-file` and
  `--auto-log-prompt <path>` now accept UTF-8 with BOM and UTF-16 (what
  PowerShell 5.1 `>` writes) instead of crashing; other encodings get a clear
  "save it as UTF-8" error.
- **Autonomous runs on Windows with redirected output** (`> run.log`, a
  scheduled task, CI) crashed on the first ▶/✓ status line because redirected
  stdout used cp1252. The CLI now switches non-UTF-8 stdout/stderr to UTF-8.

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

[Unreleased]: https://github.com/MaxNaeg/safe_lab_agents/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/MaxNaeg/safe_lab_agents/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/MaxNaeg/safe_lab_agents/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/MaxNaeg/safe_lab_agents/releases/tag/v0.1.0
