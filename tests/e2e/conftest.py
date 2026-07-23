"""Gating and parametrization for the end-to-end pipeline tests.

The whole package is inert unless ``SLA_E2E=1``. Each matrix cell additionally
self-skips when its container runtime is unavailable or its agent credentials are
absent, so a partial environment still yields a meaningful green subset.

Environment variables
---------------------
Master switch:
    SLA_E2E=1                      enable e2e collection at all

Narrowing (optional, comma lists):
    SLA_E2E_RUNTIMES=docker,podman restrict runtimes (default: all installed)
    SLA_E2E_AGENTS=claude-code,openclaw  restrict agents (default: all credentialed)

Credentials:
    SLA_E2E_CLAUDE_OAUTH_TOKEN     claude-code (from `claude setup-token`)
    SLA_E2E_OPENCLAW_API_KEY       openclaw provider API key
    SLA_E2E_OPENCLAW_PROVIDER      openclaw provider (default: anthropic)
    SLA_E2E_OPENCLAW_MODEL         openclaw model name (required for openclaw)

Behaviour knobs:
    SLA_E2E_RESUME_CONVERSE=1      drive a real turn on resume (fragile, opt-in)
    SLA_E2E_STRICT_TUI=1           hard-fail (not xfail) when a PTY-driven turn
                                   produces no tool call (default: 1)
"""

from __future__ import annotations

import os

import pytest

from . import _driver

# ----------------------------------------------------------------------
# Master gate: skip the entire package unless explicitly enabled.
# ----------------------------------------------------------------------
E2E_ENABLED = os.environ.get("SLA_E2E") == "1"

if not E2E_ENABLED:  # pragma: no cover - collection-time guard
    collect_ignore_glob = ["*"]


ALL_RUNTIMES = ["docker", "podman"]
ALL_AGENTS = ["claude-code", "openclaw"]
ALL_MODES = ["autonomous", "interactive"]


def _requested(env_var: str, allowed: list[str]) -> list[str]:
    """Return the subset of *allowed* named in *env_var* (all if unset)."""
    raw = os.environ.get(env_var, "").strip()
    if not raw:
        return list(allowed)
    picked = [x.strip() for x in raw.split(",") if x.strip()]
    return [x for x in allowed if x in picked]


def agent_args_for(agent: str) -> list[str] | None:
    """Return the ``--agent-args`` needed to run *agent* non-interactively.

    ``None`` means the required credentials are not configured, so the cell
    should skip.
    """
    if agent == "claude-code":
        token = os.environ.get("SLA_E2E_CLAUDE_OAUTH_TOKEN")
        return [f"oauth-token={token}"] if token else None
    if agent == "openclaw":
        key = os.environ.get("SLA_E2E_OPENCLAW_API_KEY")
        model = os.environ.get("SLA_E2E_OPENCLAW_MODEL")
        provider = os.environ.get("SLA_E2E_OPENCLAW_PROVIDER", "anthropic")
        if not (key and model):
            return None
        return [f"api-key={key}", f"provider={provider}", f"model={model}"]
    return None


def skip_reason(runtime: str, agent: str) -> str | None:
    """Return a human reason to skip this (runtime, agent) cell, or None to run."""
    if runtime not in _requested("SLA_E2E_RUNTIMES", ALL_RUNTIMES):
        return f"runtime {runtime} not in SLA_E2E_RUNTIMES"
    if agent not in _requested("SLA_E2E_AGENTS", ALL_AGENTS):
        return f"agent {agent} not in SLA_E2E_AGENTS"
    if not _driver.runtime_available(runtime):
        return f"{runtime} not installed or daemon unreachable"
    if agent_args_for(agent) is None:
        return f"no credentials configured for {agent}"
    return None


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "e2e: real full-pipeline test (needs SLA_E2E=1, containers, creds)"
    )


def pytest_generate_tests(metafunc):
    """Build the runtime × agent × mode matrix as parametrized, self-skipping cells."""
    if {"runtime", "agent", "mode"} <= set(metafunc.fixturenames):
        params = []
        for runtime in ALL_RUNTIMES:
            for agent in ALL_AGENTS:
                reason = skip_reason(runtime, agent)
                for mode in ALL_MODES:
                    marks = [pytest.mark.e2e]
                    if reason:
                        marks.append(pytest.mark.skip(reason=reason))
                    params.append(
                        pytest.param(
                            runtime,
                            agent,
                            mode,
                            marks=marks,
                            id=f"{runtime}-{agent}-{mode}",
                        )
                    )
        metafunc.parametrize("runtime,agent,mode", params)
