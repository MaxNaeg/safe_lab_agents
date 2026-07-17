"""Tests for the start-config file loader (``safe_lab_agents.start_config``)."""

from __future__ import annotations

from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from safe_lab_agents.cli import _flag_passed_on_command_line
from safe_lab_agents.start_config import (
    DEFAULT_CONFIG_NAME,
    discover_config_path,
    load_start_config,
    resolve_param,
)


class TestDiscoverConfigPath:
    """Tests for :func:`discover_config_path`."""

    def test_explicit_path_returned(self, tmp_path: Path) -> None:
        cfg = tmp_path / "my.yaml"
        cfg.write_text("agent: claude-code\n")
        assert discover_config_path(cfg, False, tmp_path) == cfg

    def test_explicit_missing_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            discover_config_path(tmp_path / "nope.yaml", False, tmp_path)

    def test_auto_discovery_hit(self, tmp_path: Path) -> None:
        cfg = tmp_path / DEFAULT_CONFIG_NAME
        cfg.write_text("agent: claude-code\n")
        assert discover_config_path(None, False, tmp_path) == cfg

    def test_auto_discovery_miss(self, tmp_path: Path) -> None:
        assert discover_config_path(None, False, tmp_path) is None

    def test_no_config_disables_discovery(self, tmp_path: Path) -> None:
        (tmp_path / DEFAULT_CONFIG_NAME).write_text("agent: claude-code\n")
        assert discover_config_path(None, True, tmp_path) is None

    def test_config_and_no_config_conflict(self, tmp_path: Path) -> None:
        cfg = tmp_path / "my.yaml"
        cfg.write_text("agent: claude-code\n")
        with pytest.raises(ValueError):
            discover_config_path(cfg, True, tmp_path)


class TestLoadStartConfig:
    """Tests for :func:`load_start_config`."""

    def _write(self, tmp_path: Path, body: str) -> Path:
        cfg = tmp_path / DEFAULT_CONFIG_NAME
        cfg.write_text(body)
        return cfg

    def test_key_remapping(self, tmp_path: Path) -> None:
        cfg = self._write(
            tmp_path,
            "agent: claude-code\n"
            "kadi4mat-project: my-lab\n"
            "kadi-max-per-minute: 5\n"
            "auto-log: true\n"
            "no-web: true\n"
            "egress-lockdown: false\n",
        )
        loaded = load_start_config(cfg)
        assert loaded == {
            "agent": "claude-code",
            "project": "my-lab",
            "kadi_max_per_minute": 5,
            "auto_log": True,
            "no_web": True,
            "egress_lockdown": False,
        }

    def test_unknown_key_raises(self, tmp_path: Path) -> None:
        cfg = self._write(tmp_path, "bogus-key: 1\n")
        with pytest.raises(ValueError, match="Unknown config key"):
            load_start_config(cfg)

    def test_paths_resolved_relative_to_file(self, tmp_path: Path) -> None:
        sub = tmp_path / "project"
        sub.mkdir()
        cfg = sub / DEFAULT_CONFIG_NAME
        cfg.write_text("tools: ./tools.py\nshared: ../data\n")
        loaded = load_start_config(cfg)
        assert loaded["tools"] == (sub / "tools.py").resolve()
        assert loaded["shared"] == (tmp_path / "data").resolve()

    def test_agent_args_passthrough(self, tmp_path: Path) -> None:
        cfg = self._write(
            tmp_path,
            "agent-args:\n  effort: high\n  dangerously-skip-permissions: true\n",
        )
        loaded = load_start_config(cfg)
        assert loaded["agent_args_raw"] == {
            "effort": "high",
            "dangerously-skip-permissions": True,
        }

    def test_server_list(self, tmp_path: Path) -> None:
        cfg = self._write(tmp_path, "server:\n  - lab-notebook\n")
        loaded = load_start_config(cfg)
        assert loaded["server"] == ["lab-notebook"]

    def test_empty_file_returns_empty(self, tmp_path: Path) -> None:
        cfg = self._write(tmp_path, "")
        assert load_start_config(cfg) == {}

    def test_non_mapping_raises(self, tmp_path: Path) -> None:
        cfg = self._write(tmp_path, "- just\n- a\n- list\n")
        with pytest.raises(ValueError, match="top-level mapping"):
            load_start_config(cfg)


class TestResolveParam:
    """Tests for :func:`resolve_param` (precedence logic)."""

    def test_not_in_config_keeps_cli_value(self) -> None:
        assert resolve_param("agent", "cli", False, {}) == ("cli", None)

    def test_config_used_when_not_explicit(self) -> None:
        value, override = resolve_param("agent", None, False, {"agent": "openclaw"})
        assert value == "openclaw"
        assert override is None

    def test_cli_overrides_config_with_warning(self) -> None:
        value, override = resolve_param("agent", "openclaw", True, {"agent": "claude-code"})
        assert value == "openclaw"
        assert override == ("claude-code", "openclaw")

    def test_boolean_config_flips_default(self) -> None:
        # no_web defaults to False; config sets it True and the CLI did not pass it.
        value, override = resolve_param("no_web", False, False, {"no_web": True})
        assert value is True
        assert override is None

    def test_boolean_config_flips_true_default_off(self) -> None:
        # egress_lockdown defaults to True; config disables it, CLI silent.
        value, override = resolve_param(
            "egress_lockdown", True, False, {"egress_lockdown": False}
        )
        assert value is False
        assert override is None


class TestFlagPassedOnCommandLine:
    """Tests for :func:`cli._flag_passed_on_command_line`.

    Runs a real Typer command through ``CliRunner`` so the check is exercised
    against the actually-installed Typer/Click. Guards the regression where a
    Typer that vendors its own Click made an identity comparison against
    ``click.core.ParameterSource`` always False, so ``--agent`` (and every other
    flag) was never seen as explicit and the config value always won.
    """

    @staticmethod
    def _run(*args: str) -> str:
        app = typer.Typer()

        @app.command()
        def cmd(ctx: typer.Context, agent: str = typer.Option(None, "--agent", "-a")) -> None:
            typer.echo("explicit" if _flag_passed_on_command_line(ctx, "agent") else "default")

        result = CliRunner().invoke(app, list(args))
        assert result.exit_code == 0, result.output
        return result.output.strip()

    def test_detects_explicit_flag(self) -> None:
        assert self._run("--agent", "openclaw") == "explicit"

    def test_detects_short_flag(self) -> None:
        assert self._run("-a", "openclaw") == "explicit"

    def test_default_when_flag_absent(self) -> None:
        assert self._run() == "default"
