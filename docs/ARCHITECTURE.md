# `safe_lab_agents` — Architecture Overview

## What it is

`safe_lab_agents` lets a scientist hand a lab experiment to an AI agent that runs **sandboxed in
Docker**. The agent never touches the host directly — it calls **user-defined MCP tools** that the
package serves over HTTP from the host, and those tools drive the real lab hardware. Two
properties define the system:

- **Docker provides isolation.** The agent and its tooling run in a hardened container with three
  bind mounts: `/agent/workspace` (read-write), `/agent/context` (read-only), `/agent/shared`
  (read-write). The container drops all Linux capabilities, runs under runtime-sized memory/CPU
  limits, and installs an in-container egress firewall before the agent starts (see
  [Sandbox hardening](#sandbox-hardening)). On stop, the container is committed to an image so the
  session is **resumable**.
- **MCP provides the controlled interface.** A `FastMCP` server runs in a host subprocess. It
  imports the user's plain-Python tools file at runtime and exposes each public function as an MCP
  tool (for the agent to call) or as a Python-callable tool (invoked from inside the container via
  a generated HTTP client). The server refuses to start without a **mandatory bearer token**: a
  shared secret that the agent and the generated client must present as `Authorization: Bearer …`
  on every request, so the tool server is never exposed unauthenticated. Tool calls can be
  auto-logged to JSON/HDF5 and pushed to a [Kadi4Mat](https://kadi.iam.kit.edu) ELN.

Every run's native agent log is parsed into a standard `ConversationEntry` history. The CLI ties
it all together: collect config → start MCP → build image → run container → on exit, capture logs,
import history, and commit the image. A global `--log-level` flag (also read from `LOG_LEVEL`)
configures logging for the CLI and the MCP subprocess alike (`utils.configure_logging`).

## Sandbox hardening

Beyond the container boundary itself, several defense-in-depth measures constrain what the
(untrusted) agent can reach:

- **Egress firewall.** `docker/dockerfiles/firewall.sh` is baked into every agent image and run by
  the entrypoint before the agent launches. It scopes host access to the MCP tool-server port only,
  blocks private/LAN ranges, and leaves the public internet open (for the model API). It is
  **fail-closed** — if the rules cannot be applied the container refuses to start. Opt out with
  `--no-egress-lockdown` / `SessionConfig.egress_lockdown=False`. Because `firewall.sh` ships in the
  image, its contents are folded into the build hash (`build._compute_hash`).
- **Capabilities + privilege drop.** Containers start from `cap_drop ALL`. A *session* container
  adds back `NET_ADMIN` (to install the firewall) and `SETUID`/`SETGID`; the entrypoint applies the
  firewall, then drops to the unprivileged `agent` user via `setpriv`. The one-off *login*
  container gets no host mapping and no `NET_ADMIN` (its OAuth flow only needs the public internet).
- **Mandatory MCP authentication.** `ExperimentMCPServer` refuses an empty `auth_token`; a
  `_BearerAuthMiddleware` enforces the bearer token, and the generated `tools_client.py` injects it
  on every request.
- **Resource limits.** Every container gets runtime-sized limits: memory = half the runtime-visible
  RAM (swap disabled, so the limit is a hard ceiling), CPUs = all-but-one core. Override via
  `--mem-limit` / `--cpu-limit`; a single warned retry without limits covers runtimes that reject
  them (e.g. rootless Podman on cgroups v1).

## Subsystem map

| Package | Responsibility |
|---|---|
| `cli.py` | Typer app; entry point. Orchestrates `start` / `resume` / `history` / `list` / `report` / `export-eln`. Global `--log-level` option applies across the CLI and MCP subprocess. |
| `start_config.py` | Loads the optional YAML config file (`safe-lab-agents.config.yaml`) that supplies defaults for `start`; precedence is explicit CLI flag > config value > wizard/default. |
| `config.py` | `SessionConfig` (validated user choices) and `SessionMetadata` (persisted state + status). |
| `agents/` | Agent-backend plugin system: `BaseAgent` interface + `claude-code` / `openclaw` backends, registered by name. |
| `mcp/` | MCP server subprocess (bearer-token authenticated), runtime tool loading, the in-container Python client generator, and predefined servers. |
| `docker/` | Container lifecycle: image build + hash caching, create/start/stop/commit, Docker/Podman runtime detection, capability drop + egress firewall, and runtime-sized memory/CPU limits. |
| `history/` | Parse agent-native logs into `ConversationEntry`; persist `history.json`. |
| `report/` | Build a self-contained HTML report from auto-log output. |
| `export/` | Package an auto-log folder as a standard `.eln` (RO-Crate) file for import into other ELNs (`build_eln`). |
| `mcp/predefined/records.py` | Format-neutral record serialization shared by autolog / Kadi / `.eln`: `json_safe`, the `quantity` unit convention, and the canonical HDF5 array extractor. |

## Class-interaction diagram

```mermaid
classDiagram
    direction LR

    %% ---- config ----
    class SessionConfig {
        <<Pydantic>>
        +name, agent_type
        +tools_file, workspace_dir
        +context_dir, shared_dir
        +task, predefined_servers
        +agent_args, auto_log_dir
        +kadi4mat_project
        +egress_lockdown
        +mem_limit, cpu_limit
    }
    class SessionMetadata {
        <<Pydantic>>
        +config: SessionConfig
        +container_id, image_tag
        +status  created→running→stopped→committed
        +save() / load() / list_sessions()
    }
    SessionMetadata *-- SessionConfig : composes

    %% ---- agents ----
    class BaseAgent {
        <<abstract>>
        +get_agent_type()*
        +get_dockerfile_name()*
        +get_environment_variables(config, port)*
        +get_entrypoint_command(mode)*
        +get_resume_command()*
        +parse_conversation_history(dir)*
        +get_system_prompt(config)
        +format_autonomous_line(line)
    }
    class ClaudeCodeAgent {
        @register_agent("claude-code")
    }
    class OpenClawAgent {
        @register_agent("openclaw")
    }
    class AgentArg { <<dataclass>> }
    class ConversationEntry {
        <<dataclass>>
        +timestamp, role, content
        +tool_name, tool_input, tool_output
    }
    BaseAgent <|-- ClaudeCodeAgent
    BaseAgent <|-- OpenClawAgent
    BaseAgent ..> AgentArg : declares
    BaseAgent ..> ConversationEntry : produces

    %% ---- docker ----
    class DockerManager {
        +ensure_image(agent_type, reqs)
        +create_container(config, agent, port)
        +start_interactive() / start_autonomous()
        +copy_agent_logs() / commit_container()
    }
    class build {
        <<module>>
        +ensure_base_image()  hash cache
    }
    class runtime {
        <<module>>
        +setup_podman_host()
        +container_cli()
    }
    DockerManager ..> build : ensures image
    DockerManager ..> runtime : engine detect
    DockerManager ..> BaseAgent : env + entrypoint
    DockerManager ..> SessionConfig : reads mounts

    %% ---- mcp ----
    class ExperimentMCPServer {
        +start() -> port  spawns subprocess
        +stop() / shutdown()
        +auth_token  bearer, mandatory
        +is_running
    }
    class loader {
        <<module>>
        +load_module_exports()
        MCP_TOOLS / PYTHON_TOOLS
        GRACEFUL_EXPERIMENT_SHUTDOWN
    }
    class client_generator {
        <<module>>
        +generate_client_files()  tools_client.py
    }
    class _LazyExperiment {
        experiment(factory, ...)
        lazy hardware proxy
    }
    class PredefinedServer {
        <<abstract>>
        +get_tools()*
        +get_python_tools()
    }
    class LabNotebookServer {
        @register_server("lab-notebook")
    }
    class AutoLogger {
        from_env()  wraps tool calls
        +start_batch() / stop_batch()
        write_session_summary()
    }
    class KadiClient {
        +create_record()  ELN push
    }
    PredefinedServer <|-- LabNotebookServer
    ExperimentMCPServer ..> loader : loads user tools
    ExperimentMCPServer ..> PredefinedServer : get_predefined_server()
    ExperimentMCPServer ..> AutoLogger : wraps tools (if AUTO_LOG_DIR)
    AutoLogger ..> KadiClient : push (if KADI4MAT_PROJECT)
    loader ..> _LazyExperiment : may resolve

    %% ---- history ----
    class HistoryStore {
        +import_from_agent(agent, dir)
        +save_entries() / load_history()
    }
    HistoryStore ..> BaseAgent : delegates parsing
    HistoryStore ..> ConversationEntry : stores

    %% ---- report ----
    class report_builder {
        <<module>>
        +build_report(log_dir, out)
    }

    %% ---- cli orchestrates everything ----
    class cli {
        <<Typer app>>
        start / resume / history / list / report
    }
    cli ..> SessionConfig : builds
    cli ..> SessionMetadata : persists
    cli ..> ExperimentMCPServer : start/stop
    cli ..> client_generator : generate client
    cli ..> DockerManager : build + run
    cli ..> HistoryStore : import on exit
    cli ..> report_builder : report cmd
    cli ..> BaseAgent : get_agent()
```

**Edge legend:** solid triangle `<|--` = inheritance; solid diamond `*--` = composition;
dashed `..>` = "uses / calls". Three registries decouple the wiring: `AGENT_REGISTRY`
(`@register_agent`) and `PREDEFINED_SERVERS` (`@register_server`) map names → classes, and the
user's tools file declares `MCP_TOOLS` / `PYTHON_TOOLS` / `GRACEFUL_EXPERIMENT_SHUTDOWN`.

## Session lifecycle (`start`)

```mermaid
sequenceDiagram
    actor User
    participant CLI as cli.py
    participant Cfg as SessionConfig
    participant MCP as ExperimentMCPServer
    participant Gen as client_generator
    participant DM as DockerManager
    participant Agent as BaseAgent
    participant Ctr as Container (Docker)
    participant Hist as HistoryStore

    User->>CLI: agent start ...
    CLI->>Cfg: build config (prompts + flags)
    CLI->>Agent: get_agent(config.agent_type)
    CLI->>MCP: start()  (subprocess, FastMCP)
    MCP->>MCP: load_module_exports(tools_file)
    MCP-->>CLI: port (poll /health)
    CLI->>Gen: generate_client_files() → tools_client.py
    CLI->>DM: ensure_image(agent_type)  (hash cache → build if stale)
    CLI->>DM: create_container(config, agent, port)
    DM->>Agent: get_environment_variables() + get_entrypoint_command()
    CLI->>Cfg: SessionMetadata(status="running").save()
    CLI->>DM: start_interactive() / start_autonomous()
    Note over Ctr: entrypoint installs egress firewall, then setpriv → agent user
    Ctr->>MCP: HTTP tool calls over /mcp (also /invoke) — Authorization: Bearer <token>
    Note over Ctr,MCP: AutoLogger records calls, KadiClient pushes to ELN
    Ctr-->>CLI: container exits (or Ctrl-C)
    CLI->>DM: copy_agent_logs() → session_dir/logs
    CLI->>Hist: import_from_agent(agent, log_dir)
    Hist->>Agent: parse_conversation_history() → ConversationEntry[]
    CLI->>DM: commit_container() → safe-lab-agents-session-<name>:latest
    CLI->>MCP: stop()  (runs GRACEFUL_EXPERIMENT_SHUTDOWN)
    CLI->>Cfg: SessionMetadata(status="committed").save()
```

`resume` follows the same path but loads the existing `SessionMetadata`, restores the committed
session image, and uses `BaseAgent.get_resume_command()` instead of the fresh entrypoint.

## Class reference

| Class | File | Base | Key members |
|---|---|---|---|
| `SessionConfig` | `config.py` | Pydantic `BaseModel` | `name`, `agent_type`, `tools_file`, `workspace_dir`, `context_dir`, `shared_dir`, `task`, `predefined_servers`, `agent_args`, `auto_log_dir`, `kadi4mat_project`, `egress_lockdown`, `mem_limit`, `cpu_limit` |
| `SessionMetadata` | `config.py` | Pydantic `BaseModel` | composes `config`; `container_id`, `image_tag`, `status`; `save()`, `load()`, `list_sessions()` |
| `BaseAgent` | `agents/base.py` | `ABC` | abstract: `get_agent_type`, `get_dockerfile_name`, `get_environment_variables`, `get_entrypoint_command`, `get_resume_command`, `parse_conversation_history`; concrete: `get_system_prompt`, `get_agent_args`, `format_autonomous_line` |
| `ClaudeCodeAgent` | `agents/claude_code.py` | `BaseAgent` | parses `~/.claude/projects/**/*.jsonl`; renders `stream-json`; OAuth/login handling |
| `OpenClawAgent` | `agents/openclaw.py` | `BaseAgent` | parses `~/.openclaw/agents/**`; provider/api-key/model args |
| `AgentArg` | `agents/base.py` | dataclass | descriptor for an agent CLI flag |
| `ConversationEntry` | `agents/base.py` | dataclass | `timestamp`, `role`, `content`, `tool_name`, `tool_input`, `tool_output`, `metadata` |
| `DockerManager` | `docker/manager.py` | — | `ensure_image`, `create_container`, `start_interactive`, `start_autonomous`, `copy_agent_logs`, `commit_container`, `engine_is_podman` |
| `ExperimentMCPServer` | `mcp/server.py` | — | `start() -> port`, `stop()`, `shutdown()`, `is_running`; mandatory `auth_token` (empty is refused); subprocess runs `_run_server` (FastMCP, `/health`, `/invoke`) behind `_BearerAuthMiddleware` |
| `PredefinedServer` | `mcp/predefined/__init__.py` | `ABC` | `get_tools()`, `get_python_tools()` |
| `LabNotebookServer` | `mcp/predefined/lab_notebook.py` | `PredefinedServer` | `add_entry`, `search_entries`, `list_entries` |
| `AutoLogger` | `mcp/predefined/autolog.py` | — | `from_env()`; wraps tool calls → JSON/HDF5 records; `start_batch()`, `stop_batch()`; module-level `write_session_summary()` |
| `KadiClient` | `mcp/predefined/kadi4mat.py` | — | `create_record()`; rate-limited ELN push, invoked by the `AutoLogger` wrapper |
| `_LazyExperiment` | `mcp/tool_utils.py` | — | `experiment(factory, …)` — defers hardware construction to first use |
| `HistoryStore` | `history/store.py` | — | `import_from_agent`, `save_entries`, `load_history`, `append_entry` |

Module-level orchestration: `docker/build.py` (`ensure_base_image`, SHA-256 hash cache over the
Dockerfile + entrypoint + `firewall.sh` + requirements), `docker/runtime.py` (`setup_podman_host`,
`container_cli`), `mcp/loader.py` (`load_module_exports`), `mcp/client_generator.py`
(`generate_client_files`), `report/builder.py` (`build_report`), `export/eln.py` (`build_eln` —
maps auto-log records to an RO-Crate `.eln`), `mcp/predefined/records.py` (`json_safe`, `quantity`,
`extract_arrays`), `utils.py` (`configure_logging` — honors `--log-level` / `LOG_LEVEL`).

## Extension points

- **Add an agent backend.** Implement `BaseAgent` in `agents/my_agent.py`, decorate with
  `@register_agent("my-agent")`, add `Dockerfile.my-agent` + `entrypoint.my-agent.sh` under
  `docker/dockerfiles/`. Auto-imported via `agents/__init__.py`. (See `CLAUDE.md` → *Adding a New
  Agent*.)
- **Add a predefined MCP server.** Subclass `PredefinedServer` in `mcp/predefined/my_server.py`,
  decorate `@register_server("my-server")`. Auto-imported via `mcp/predefined/__init__.py`.
- **Tools file conventions.** Functions listed in `MCP_TOOLS` become agent-callable MCP tools;
  those in `PYTHON_TOOLS` are exposed via the generated `tools_client.py` inside the container.
  Wrap stateful hardware with `experiment(factory, …)` so it's constructed lazily (the tools file
  is imported in several processes). A module-level `GRACEFUL_EXPERIMENT_SHUTDOWN` callable runs
  cleanup inside the MCP subprocess on reload and shutdown. (See `CLAUDE.md` → *MCP Tool
  Loading*.)
