"""Tests for agent registry and implementations."""

from __future__ import annotations

from pathlib import Path

import pytest

from safe_lab_agents.agents import get_agent, list_agents
from safe_lab_agents.agents.base import AGENT_REGISTRY
from safe_lab_agents.config import SessionConfig


class TestAgentRegistry:
    """Tests for the agent registry."""

    def test_claude_code_registered(self) -> None:
        """Claude Code agent is auto-registered."""
        assert "claude-code" in AGENT_REGISTRY

    def test_openclaw_registered(self) -> None:
        """OpenClaw agent is auto-registered."""
        assert "openclaw" in AGENT_REGISTRY

    def test_get_agent(self) -> None:
        """get_agent returns the correct agent instance."""
        agent = get_agent("claude-code")
        assert agent.get_agent_type() == "claude-code"

    def test_get_unknown_agent(self) -> None:
        """get_agent raises for unknown agents."""
        with pytest.raises(ValueError, match="Unknown agent"):
            get_agent("nonexistent")

    def test_list_agents(self) -> None:
        """list_agents returns at least the two built-in agents."""
        agents = list_agents()
        assert "claude-code" in agents
        assert "openclaw" in agents


class TestClaudeCodeAgent:
    """Tests for the Claude Code agent backend."""

    def test_dockerfile_name(self) -> None:
        agent = get_agent("claude-code")
        assert agent.get_dockerfile_name() == "Dockerfile.claude-code"

    def test_login_command(self) -> None:
        agent = get_agent("claude-code")
        assert agent.get_login_command() == ["--login"]

    def test_env_no_api_key(self, tmp_path: Path) -> None:
        """Claude Code does not inject an API key (subscription-based)."""
        agent = get_agent("claude-code")
        cfg = SessionConfig(
            name="test",
            tools_file=tmp_path / "tools.py",
            workspace_dir=tmp_path / "ws",
        )
        env = agent.get_environment_variables(cfg, mcp_port=8000)
        assert "ANTHROPIC_API_KEY" not in env
        assert env["MCP_PORT"] == "8000"
        assert env["MODE"] == "interactive"

    def test_autonomous_mode_env(self, tmp_path: Path) -> None:
        """Task triggers autonomous mode and TASK_PROMPT."""
        agent = get_agent("claude-code")
        cfg = SessionConfig(
            name="test",
            tools_file=tmp_path / "tools.py",
            workspace_dir=tmp_path / "ws",
            task="Run experiment",
        )
        env = agent.get_environment_variables(cfg, mcp_port=8000)
        assert env["MODE"] == "autonomous"
        assert env["TASK_PROMPT"] == "Run experiment"

    def test_get_agent_args(self) -> None:
        """ClaudeCodeAgent declares effort, copy-host-credentials, dangerously-skip-permissions."""
        agent = get_agent("claude-code")
        args = {a.name: a for a in agent.get_agent_args()}
        assert "effort" in args
        assert "copy-host-credentials" in args
        assert not args["copy-host-credentials"].default
        assert "new-login" not in args
        assert "dangerously-skip-permissions" in args
        assert "oauth-token" in args
        assert args["oauth-token"].type is str
        assert args["oauth-token"].is_secret
        assert args["effort"].choices == ["low", "medium", "high", "xhigh", "max"]
        assert not args["effort"].required
        assert not args["effort"].required_for_autonomous

    def test_effort_via_agent_args(self, tmp_path: Path) -> None:
        """effort in agent_args sets CLAUDE_EFFORT env var."""
        agent = get_agent("claude-code")
        cfg = SessionConfig(
            name="test",
            tools_file=tmp_path / "tools.py",
            workspace_dir=tmp_path / "ws",
            agent_args={"effort": "high"},
        )
        env = agent.get_environment_variables(cfg, mcp_port=8000)
        assert env["CLAUDE_EFFORT"] == "high"

    def test_skip_permissions_via_agent_args(self, tmp_path: Path) -> None:
        """dangerously-skip-permissions in agent_args sets SKIP_PERMISSIONS."""
        agent = get_agent("claude-code")
        cfg = SessionConfig(
            name="test",
            tools_file=tmp_path / "tools.py",
            workspace_dir=tmp_path / "ws",
            agent_args={"dangerously-skip-permissions": True},
        )
        env = agent.get_environment_variables(cfg, mcp_port=8000)
        assert env.get("SKIP_PERMISSIONS") == "true"

    def test_parse_history_coerces_input_and_aware_timestamp(self, tmp_path: Path) -> None:
        """A non-dict tool input is coerced to {} and timestamps are tz-aware."""
        import json

        jsonl_dir = tmp_path / "projects" / "abc123"
        jsonl_dir.mkdir(parents=True)
        record = {
            "type": "assistant",
            "timestamp": "2026-04-13T10:00:00Z",
            "message": {
                "content": [
                    {"type": "tool_use", "id": "t1", "name": "foo", "input": "not-a-dict"}
                ]
            },
        }
        (jsonl_dir / "session.jsonl").write_text(json.dumps(record), encoding="utf-8")

        agent = get_agent("claude-code")
        entries = agent.parse_conversation_history(tmp_path)

        tool_uses = [e for e in entries if e.role == "tool_use"]
        assert len(tool_uses) == 1
        assert tool_uses[0].tool_input == {}  # non-dict coerced (no .items() crash)
        assert tool_uses[0].timestamp.tzinfo is not None  # timezone-aware

    def test_parse_history_mixed_naive_and_aware_timestamps(self, tmp_path: Path) -> None:
        """A log mixing a naive ISO timestamp with the aware now()-fallback (missing
        timestamp) must still sort — the parse used to raise TypeError on the mix."""
        import json

        jsonl_dir = tmp_path / "projects" / "abc123"
        jsonl_dir.mkdir(parents=True)
        records = [
            # naive ISO timestamp (no zone) — used to parse to a naive datetime
            {"type": "user", "timestamp": "2026-04-13T10:00:05",
             "message": {"content": "second"}},
            # missing timestamp — falls back to now(), which is tz-aware
            {"type": "user", "message": {"content": "no-timestamp"}},
            # zoned timestamp — aware
            {"type": "user", "timestamp": "2026-04-13T10:00:00Z",
             "message": {"content": "first"}},
        ]
        (jsonl_dir / "session.jsonl").write_text(
            "\n".join(json.dumps(r) for r in records), encoding="utf-8"
        )

        agent = get_agent("claude-code")
        entries = agent.parse_conversation_history(tmp_path)  # must not raise

        # Every timestamp is normalized to tz-aware, so the sort succeeded.
        assert entries  # parse produced entries rather than aborting
        assert all(e.timestamp.tzinfo is not None for e in entries)

    def test_parse_history_missing_projects_dir(self, tmp_path: Path) -> None:
        """With no projects/ subdirectory, parsing returns an empty list."""
        agent = get_agent("claude-code")
        assert agent.parse_conversation_history(tmp_path) == []

    def test_system_prompt(self, tmp_path: Path) -> None:
        """System prompt mentions workspace and context when configured."""
        agent = get_agent("claude-code")
        cfg = SessionConfig(
            name="test",
            tools_file=tmp_path / "tools.py",
            workspace_dir=tmp_path / "ws",
            context_dir=tmp_path / "ctx",
            shared_dir=tmp_path / "shared",
        )
        prompt = agent.get_system_prompt(cfg)
        assert "/agent/workspace" in prompt
        assert "/agent/context" in prompt
        assert "/agent/shared" in prompt

    def test_system_prompt_no_web(self, tmp_path: Path) -> None:
        """The web restriction appears only when --no-web is set."""
        agent = get_agent("claude-code")
        base_kwargs = dict(name="t", tools_file=tmp_path / "t.py", workspace_dir=tmp_path / "ws")
        assert "RESTRICTION" not in agent.get_system_prompt(SessionConfig(**base_kwargs))
        assert "RESTRICTION" in agent.get_system_prompt(SessionConfig(**base_kwargs, no_web=True))

    def test_system_prompt_omits_kadi(self, tmp_path: Path) -> None:
        """The base prompt no longer carries Kadi text (matches the live entrypoints)."""
        agent = get_agent("claude-code")
        cfg = SessionConfig(
            name="t", tools_file=tmp_path / "t.py", workspace_dir=tmp_path / "ws",
            predefined_servers=["kadi4mat"], kadi4mat_project="proj",
        )
        assert "Kadi4Mat" not in agent.get_system_prompt(cfg)


class TestOpenClawAgent:
    """Tests for the OpenClaw agent backend."""

    def test_dockerfile_name(self) -> None:
        agent = get_agent("openclaw")
        assert agent.get_dockerfile_name() == "Dockerfile.openclaw"

    def test_system_prompt_overrides_base(self, tmp_path: Path) -> None:
        """OpenClaw adds its 'prefer MCP tools' guidance and its own no-web wording."""
        agent = get_agent("openclaw")
        cfg = SessionConfig(
            name="t", tools_file=tmp_path / "t.py", workspace_dir=tmp_path / "ws",
            no_web=True,
        )
        prompt = agent.get_system_prompt(cfg)
        assert "experiment-tools" in prompt          # MCP-prefer sentence
        assert "web search or web fetch" in prompt   # OpenClaw's soft no-web wording

    def test_parse_history_strips_sender_metadata_wrapper(self, tmp_path: Path) -> None:
        """OpenClaw's 'Sender (untrusted metadata)' preamble and [timestamp]
        prefix are stripped from user messages."""
        import json

        jsonl_dir = tmp_path / ".openclaw" / "agents" / "a1" / "sessions"
        jsonl_dir.mkdir(parents=True)
        wrapped = (
            "Sender (untrusted metadata):\n```json\n"
            '{\n  "label": "openclaw-tui"\n}\n```\n\n'
            "[Wed 2026-04-22 16:00 UTC] measure voltage"
        )
        record = {
            "type": "message",
            "timestamp": "2026-04-22T16:00:00Z",
            "message": {"role": "user", "content": [{"type": "text", "text": wrapped}]},
        }
        (jsonl_dir / "s.jsonl").write_text(json.dumps(record), encoding="utf-8")

        agent = get_agent("openclaw")
        users = [e for e in agent.parse_conversation_history(tmp_path) if e.role == "user"]
        assert len(users) == 1
        assert users[0].content == "measure voltage"

    def test_parse_history_skips_bundled_plugin_tree(self, tmp_path: Path) -> None:
        """Session logs are parsed, but *.jsonl files inside OpenClaw's bundled
        plugin/npm install trees are ignored — those deeply nested paths
        overflow Windows' MAX_PATH and would otherwise crash the import."""
        import json

        record = {
            "type": "message",
            "timestamp": "2026-04-22T16:00:00Z",
            "message": {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        }
        blob = json.dumps(record)

        sessions = tmp_path / ".openclaw" / "agents" / "main" / "sessions"
        sessions.mkdir(parents=True)
        (sessions / "s.jsonl").write_text(blob, encoding="utf-8")

        # A stray *.jsonl buried inside the pruned plugin tree must be ignored.
        plugin = (
            tmp_path / ".openclaw" / "agents" / "main" / "agent"
            / "codex-home" / ".tmp" / "plugins" / "nvidia" / "skills"
        )
        plugin.mkdir(parents=True)
        (plugin / "manifest.jsonl").write_text(blob, encoding="utf-8")

        agent = get_agent("openclaw")
        users = [e for e in agent.parse_conversation_history(tmp_path) if e.role == "user"]
        assert len(users) == 1

    def test_parse_history_extracts_tool_result_envelope(self, tmp_path: Path) -> None:
        """toolResult content is a {"type":"toolResult","content":"<mcp-json>"}
        envelope, not a plain text block; its inner text must be surfaced."""
        import json

        inner_envelope = json.dumps(
            {
                "content": [{"type": "text", "text": '{"temperature":22.5}'}],
                "structuredContent": {"result": {"temperature": 22.5}},
            }
        )
        record = {
            "type": "message",
            "timestamp": "2026-04-22T16:00:00Z",
            "message": {
                "role": "toolResult",
                "toolName": "experiment-tools.get_temperature",
                "isError": False,
                "content": [{"type": "toolResult", "content": inner_envelope}],
            },
        }
        jsonl_dir = tmp_path / ".openclaw" / "agents" / "main" / "sessions"
        jsonl_dir.mkdir(parents=True)
        (jsonl_dir / "s.jsonl").write_text(json.dumps(record), encoding="utf-8")

        agent = get_agent("openclaw")
        results = [
            e for e in agent.parse_conversation_history(tmp_path) if e.role == "tool_result"
        ]
        assert len(results) == 1
        assert results[0].tool_name == "experiment-tools.get_temperature"
        assert results[0].content == '{"temperature":22.5}'


class TestMergeAgentArgs:
    """Tests for merging config-file agent-args with CLI --agent-args."""

    def test_cli_key_overrides_only_that_config_key(self) -> None:
        """A CLI key overrides its config counterpart while other config keys survive."""
        from safe_lab_agents.cli import _merge_agent_args

        agent = get_agent("claude-code")
        merged, overridden = _merge_agent_args(
            {"model": "opus", "effort": "max"}, ["effort=low"], agent
        )
        assert merged == {"model": "opus", "effort": "low"}
        assert overridden == ["effort"]

    def test_cli_adds_new_key_without_dropping_config(self) -> None:
        """A CLI key absent from the config is added; nothing is reported as overridden."""
        from safe_lab_agents.cli import _merge_agent_args

        agent = get_agent("claude-code")
        merged, overridden = _merge_agent_args(
            {"model": "opus", "effort": "max"},
            ["dangerously-skip-permissions"],
            agent,
        )
        assert merged == {
            "model": "opus",
            "effort": "max",
            "dangerously-skip-permissions": True,
        }
        assert overridden == []

    def test_no_cli_args_keeps_config(self) -> None:
        """With no CLI args, the config mapping is used as-is."""
        from safe_lab_agents.cli import _merge_agent_args

        agent = get_agent("claude-code")
        merged, overridden = _merge_agent_args({"model": "opus"}, [], agent)
        assert merged == {"model": "opus"}
        assert overridden == []

    def test_no_config_uses_cli_only(self) -> None:
        """With no config mapping, only CLI args are parsed."""
        from safe_lab_agents.cli import _merge_agent_args

        agent = get_agent("claude-code")
        merged, overridden = _merge_agent_args(None, ["effort=high"], agent)
        assert merged == {"effort": "high"}
        assert overridden == []
