"""Load a YAML config file that supplies defaults for ``agent start``.

The config file lets users keep the flags they pass to ``start`` in a file instead
of retyping them every run. Keys are exactly the CLI flag names with the leading
``--`` stripped (kept hyphenated), so the file reads identically to the command line::

    # safe-lab-agents.config.yaml
    agent: claude-code
    tools: ./tools.py
    shared: ./data
    auto-log: true
    agent-args:
      effort: high

Precedence is handled in the CLI: an explicit command-line flag always wins over a
value from this file, which in turn wins over the interactive wizard / hardcoded
defaults. This module only discovers, parses, validates, and normalizes the file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml

# Default file name auto-discovered in the current directory when --config is omitted.
DEFAULT_CONFIG_NAME = "safe-lab-agents.config.yaml"

# Config-file key (hyphenated flag name, no leading ``--``) -> ``start`` parameter name.
# Derived directly from the flag spellings so it stays in sync with the CLI.
CONFIG_KEY_MAP: dict[str, str] = {
    "agent": "agent",
    "tools": "tools",
    "context": "context",
    "shared": "shared",
    "task": "task",
    "task-file": "task_file",
    "name": "name",
    "server": "server",
    "requirements": "requirements",
    "rebuild": "rebuild",
    "agent-args": "agent_args_raw",
    "kadi4mat-project": "project",
    "kadi-max-per-minute": "kadi_max_per_minute",
    "kadi-max-per-session": "kadi_max_per_session",
    "port": "port",
    "container": "container",
    "no-web": "no_web",
    "update-tools": "update_tools",
    "auto-log": "auto_log",
}

# Config keys whose values are filesystem paths, resolved relative to the config file.
PATH_KEYS: set[str] = {"tools", "context", "shared", "requirements", "task-file"}


def discover_config_path(
    explicit: Optional[Path],
    no_config: bool,
    search_dir: Path,
) -> Optional[Path]:
    """Resolve which config file to load, if any.

    - ``explicit`` and ``no_config`` together is contradictory -> ``ValueError``.
    - ``explicit`` given -> that path (must exist, else ``FileNotFoundError``).
    - ``no_config`` -> ``None`` (auto-discovery disabled).
    - otherwise -> ``search_dir/safe-lab-agents.config.yaml`` if it exists, else ``None``.
    """
    if explicit is not None and no_config:
        raise ValueError("--config and --no-config are mutually exclusive.")
    if explicit is not None:
        if not explicit.exists():
            raise FileNotFoundError(f"Config file not found: {explicit}")
        return explicit
    if no_config:
        return None
    candidate = search_dir / DEFAULT_CONFIG_NAME
    return candidate if candidate.exists() else None


def load_start_config(path: Path) -> dict[str, Any]:
    """Parse, validate, and normalize a start-config file.

    Returns a dict keyed by ``start`` parameter names (already remapped from the
    hyphenated config keys). Path-valued entries are resolved relative to the config
    file's directory. The ``agent-args`` mapping is passed through unchanged under the
    ``agent_args_raw`` key for special handling in the CLI.
    """
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config file {path} must contain a top-level mapping (key: value pairs).")

    unknown = [k for k in raw if k not in CONFIG_KEY_MAP]
    if unknown:
        valid = ", ".join(sorted(CONFIG_KEY_MAP))
        raise ValueError(
            f"Unknown config key(s) in {path}: {', '.join(unknown)}. Valid keys: {valid}"
        )

    base_dir = path.parent
    resolved: dict[str, Any] = {}
    for key, value in raw.items():
        param = CONFIG_KEY_MAP[key]
        if key in PATH_KEYS and value is not None:
            value = (base_dir / Path(value).expanduser()).resolve()
        resolved[param] = value
    return resolved


def resolve_param(
    param: str,
    cli_value: Any,
    cli_explicit: bool,
    cfg: dict[str, Any],
) -> tuple[Any, Optional[tuple[Any, Any]]]:
    """Resolve a single parameter against the config file.

    Precedence: an explicit command-line value wins over the config value, which
    wins over the (default) ``cli_value``.

    Returns ``(resolved_value, override)`` where ``override`` is ``None`` unless an
    explicit CLI flag overrode a config-supplied value, in which case it is the
    ``(config_value, cli_value)`` pair (so the caller can warn about it).
    """
    if param not in cfg:
        return cli_value, None
    cfg_value = cfg[param]
    if cli_explicit:
        return cli_value, (cfg_value, cli_value)
    return cfg_value, None
