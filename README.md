<div align="center">
<pre>
╭──────────────────────────────────╮
│  ▄▖▄▖▄▖▄▖  ▖ ▄▖▄▖  ▄▖▄▖▄▖▖▖▄▖▄▖  │
│  ▚ ▌▌▙▖▙▖  ▌ ▌▌▙▘  ▌▌▌▖▙▖▛▌▐ ▚   │
│  ▄▌▛▌▌ ▙▖  ▙▖▛▌▙▘  ▛▌▙▌▙▖▌▌▐ ▄▌  │
│                                  │
│ sandboxed AI agents for lab work │
╰──────────────────────────────────╯
</pre>
</div>

# Safe Lab Agents

Safely run AI agents in a sandbox to autonomously control scientific experiments via an interface you define.

By Maximilian Nägele and Florian Marquardt. Made at the [Max Planck Institute for the Science of Light](https://mpl.mpg.de/divisions/marquardt-division/research) in Germany. Open source (MIT license) and forever free. First release July 2026.

Inspired by our earlier [SciExplorer publication](https://journals.aps.org/prx/abstract/10.1103/xnqc-q6nt) (code also on [github](https://github.com/MaxNaeg/SciExplorer)).

## Overview

**Safe Lab Agents** lets experimental scientists hand control of their experiment to an AI agent — while keeping safety guarantees. The agent runs **sandboxed inside a Docker container** and can only interact with lab hardware through **user-defined MCP tools** running on the host.

<div align="center">
<pre>
┌─────────────────────────────────────────┐
│              Host Machine               │
│                                         │
│  ┌───────────────────────────────────┐  │
│  │  MCP Server (host-side)           │  │
│  │  ┌─────────────────────────────┐  │  │
│  │  │ Your tool functions         │  │  │
│  │  │  - set_angle(angle, comp)   │  │  │
│  │  │  - measure_power()          │  │  │
│  │  │  - get_lab_temperature(pos) │  │  │
│  │  └─────────────────────────────┘  │  │
│  └──────────────▲────────────────────┘  │
│                 │ HTTP                  │
│  ┌──────────────▼────────────────────┐  │
│  │  Docker Container                 │  │
│  │  ┌─────────────────────────────┐  │  │
│  │  │  AI Agent                   │  │  │
│  │  │  (Claude Code / OpenClaw)   │  │  │
│  │  └─────────────────────────────┘  │  │
│  │                                   │  │
│  │  Mounted directories:             │  │
│  │   /agent/context  (read-only)     │  │
│  │   /agent/shared   (read-write)    │  │
│  │   /agent/workspace(read-write)    │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
</pre>
</div>

**Key features:**

- **Safety first** — the agent is sandboxed in Docker; it can only affect your experiment through the MCP tools you define.
- **Easy to use** — install with `pip`, define tools as plain Python functions, run one command.
- **Automatic logging** — every tool call is recorded as a structured record (JSON + HDF5 for arrays) with no changes to your tools.
- **ELN compatibility** — sessions export to the standard `.eln` format, importable into eLabFTW, Kadi4Mat, PASTA, and other electronic lab notebooks.
- **Multiple agents** — supports Claude Code and OpenClaw out of the box.
- **Session persistence** — Docker state is automatically committed when you stop, so you can resume conversations later.
- **Conversation history** — all agent actions are stored and can be viewed with rich terminal formatting or exported as a self-contained HTML viewer.
- **Autonomous mode** — give the agent a task and let it run without interaction.

**Website:**

[safe-lab-agents.org](https://safe-lab-agents.org/)

## Installation

### Prerequisites

- **Python 3.10+**
- **On Windows**, make sure to run 'wsl --install' in an admin PowerShell and reboot before installing Podman/Docker.
- **A container runtime** — either:
  - **Docker** — [Install Docker Desktop](https://docs.docker.com/get-docker/) (or Docker Engine on Linux). ⚠️ **Docker Desktop requires a paid subscription for large organizations** (see [Docker's terms](https://www.docker.com/pricing/)) on Windows and macOS — Podman is a free alternative.
  - **Podman** — a free, open-source alternative to Docker. [Install Podman](https://podman.io/docs/installation).
- **Agent credentials** — a Claude Code subscription (for the Claude Code agent), or an LLM API key such as an Anthropic or OpenAI key (for OpenClaw).

Tested on **macOS 15**, **Windows 11**, and **Ubuntu 24.04**. If you hit runtime issues (especially with Podman), see [Troubleshooting](#troubleshooting) at the end.

### Install the package

We recommend installing into a virtual environment (e.g. `venv` or `conda`) to keep the dependencies isolated.
To install from source:

```bash
git clone git@github.com:MaxNaeg/safe_lab_agents.git
cd safe_lab_agents
pip install -e .
```

or simply use pip:
```bash
pip install safe-lab-agents
```

## Minimal Example

We will go step by step through a minimal example where we pretend to run microwave transmission measurements on some experimental setup. We will have the AI agent analyze that setup from scratch, without any prior knowledge, and document its results.

After installation, create the following python file (here called `my_tools.py`) inside a new folder somewhere on your hard drive. In this example, we define a single function that returns a mock measurement result of a microwave transmission spectrum.

```python my_tools.py
import numpy as np

def measure_transmission(omega: float) -> dict:
   """
Measure the intensity transmission at a given frequency.

Args:
   omega - frequency in GHz
   """
   if omega>1.0 and omega<10.0:
      return { "transmission": np.abs(.5/((omega-3)+.5j))**2, "omega": omega }

PYTHON_TOOLS=[measure_transmission]
```

Now run `mkdir shared` inside the same folder to produce a subfolder in which the agent can place files that you can later access outside the sandbox -- for example documented measurement results.

To launch the AI agent safely inside a sandbox, simply run `agent start` inside your folder!

![Terminal window with command "agent start" entered](https://raw.githubusercontent.com/MaxNaeg/safe_lab_agents/main/docs/figures_mini_example/fig_agent_start.jpg)

You will be asked a few questions by the setup wizard. In this example, we assume you had installed the open `podman` Docker alternative and you have a `Claude Pro (or Max)` subscription (though not necessarily a working installation of claude code) -- otherwise change as needed. Also enter `my_tools.py` when asked for the path to your tools python file, and enter `shared` when asked for the path to the shared folder. Say yes (`y`) when asked about auto-logging the experimental data.

![Terminal window showing a set of interactive prompts and user answers](https://raw.githubusercontent.com/MaxNaeg/safe_lab_agents/main/docs/figures_mini_example/fig_further_questions.jpg)

Now you will observe some startup messages and the installation of various packages (such as numpy). Eventually `Claude Code` will launch. It will ask you to copy a link into a browser for getting the login credentials. This demonstrates that the sandbox works, as the agent cannot just launch a web page inside a browser on your machine (which would be what happens normally at this point). Copy that link to a browser, authorize access, and copy the resulting code back into the terminal. Afterwards, `Claude Code` is ready to take your instructions.

![Terminal window showing a freshly launched Claude Code instance](https://raw.githubusercontent.com/MaxNaeg/safe_lab_agents/main/docs/figures_mini_example/fig_claude_launch.jpg)

Give it instructions like `Characterize this optical setup` and let it run. In this example, it will still ask for permissions, but since you are inside the sandbox, feel free to let it run autonomously without any danger! You can do so by pressing shift-tab several times to reach 'auto' mode. Alternatively, you could have used `agent start --task "Analyze the optical setup!"` in the beginning to directly go to automatic mode.

![Terminal window showing a freshly launched Claude Code instance](https://raw.githubusercontent.com/MaxNaeg/safe_lab_agents/main/docs/figures_mini_example/fig_run.jpg)

The agent will go through a series of measurements, will try to analyze, form hypotheses, run simulations, produce plots, and so on. Finally it converges on the assessment that this device has a Lorentzian transmission spectrum. 

To finish, type `/exit` in `Claude Code`, and then `exit` again in the terminal. This last `exit` takes you out of the sandbox, back to the folder where you started. Now inside the `shared` subfolder you will find all kinds of files, representing the data that were taken. Since we switched on auto-logging, we can generate a nice report of the experiment, using `agent report shared/auto_log/ --open`.

![Terminal window showing a freshly launched Claude Code instance](https://raw.githubusercontent.com/MaxNaeg/safe_lab_agents/main/docs/figures_mini_example/fig_auto_log.jpg)

This will open the electronic lab notebook in a browser, as a searchable html document. That document shows you all the data that have been taken and the various analysis outcomes, including figures and analysis scripts. 

![Terminal window showing a freshly launched Claude Code instance](https://raw.githubusercontent.com/MaxNaeg/safe_lab_agents/main/docs/figures_mini_example/fig_browsing_auto_log.jpg)

Besides this log, there is also the full agent conversation history, including everything it was saying while exploring the experimental setup. In this example, you would use `agent history --name session-20260704-085415 --open` to show the html document for the history.

That's it for now! See below for a more complete example and explanation of the various options.

## Quick Start

Note: A complete, runnable example lives in [`example_setup/`](example_setup/) — a simulated optical bench with an experiment class ([`setup.py`](example_setup/setup.py)), the tools file ([`tools.py`](example_setup/tools.py)), a config file, and a captured agent run.

### 1. Define your tools

Create a folder for your experiment and, inside it, a Python file with functions that control your hardware:

```python
from safe_lab_agents import experiment  # lazy wrapper — avoids opening hardware on import
from safe_lab_agents import quantity, Quantity   # attach units to measurements for richer logging

from setup import ExampleOpticalSetup    # your class that talks to the hardware

# Constructed lazily on first use — see "Stateful experiments" below.
exp = experiment(ExampleOpticalSetup)

def get_current_lab_temperature(position: str) -> dict[str, Quantity|str]:
    """Return the current lab temperature at a given position.

    Args:
        position: One of 'near_laser', 'near_detector', or 'ambient'.
    """
    # Enforce safety checks in the tool — the agent can only call these functions,
    # so it cannot bypass them.
    if position not in ("near_laser", "near_detector", "ambient"):
        raise ValueError(f"Invalid position: {position}.")
    return {"temperature": quantity(22.5, "degrees_C"), "position": position}

# The agent should be able to run efficient sweeps, so expose set_angle/measure_power via
# the Python interface; the one-off temperature reading is a plain MCP tool.
MCP_TOOLS    = [get_current_lab_temperature]      # one-off; agent reasons about the result
PYTHON_TOOLS = [exp.set_angle, exp.measure_power]  # called repeatedly inside sweeps

# Called automatically when the session ends — put hardware into a safe state here.
GRACEFUL_EXPERIMENT_SHUTDOWN = exp.close
```

**Type hints and docstrings are read by the agent** to understand what each tool does, so write them clearly. Two lists at the bottom of the file control which functions are exposed through each interface:

- **`PYTHON_TOOLS`** — the agent calls these as regular Python functions from scripts it writes inside the container, receiving native Python objects back (numpy arrays, dicts, …). Best for functions the agent calls **many times in a sweep/loop** or whose results are large arrays it processes in code (here, `set_angle`/`measure_power`).
- **`MCP_TOOLS`** — the agent calls these as tools and reads the result as text in the conversation. Best for **one-off calls** whose result the agent reasons about directly (scalars, status strings, small structured dicts — here, `get_current_lab_temperature`).

A function can appear in both lists, and each list is optional. `experiment()` wraps your hardware class so it is opened lazily (see [Stateful experiments](#stateful-experiments)), and `GRACEFUL_EXPERIMENT_SHUTDOWN` runs on exit to leave instruments in a safe state.

> Heads up: the tools file is imported several times (once per MCP subprocess, plus in the parent process for client generation), so avoid expensive or stateful work at module top level.

### 2. Start a session

Just run:

```bash
agent start
```

The interactive wizard guides you through each setting. You can also pass everything explicitly:

```bash
agent start \
    --agent claude-code \
    --tools tools.py \
    --context ./context/ \
    --shared ./shared/
```

Here `--context` is a directory of experiment background (protocols, descriptions, prior data) mounted **read-only** for the agent, and `--shared` is a **read-write** directory for exchanging data (measurements, figures, analysis) between your instruments and the agent. Both are optional. Independent sessions can point at the **same** `--context`/`--shared` directories (e.g. to build on a shared dataset) or use **separate** ones to stay fully isolated — it's up to you.

The agent starts in your terminal. Ask it to use your tools — for example: *"Calibrate the setup: find the waveplate angles that maximize detected power, then report them."* It can call your tools, write Python scripts, and create files — all visible in the shared and workspace directories on your host.

> **Exiting drops you into the container shell.** When the agent exits (or you press `Ctrl+C`), you are **still inside the container's shell** — handy for inspecting files the agent created. Type `exit` there to leave the container; the session is then committed so you can resume it later.

### Stop and resume

Stopping (exiting the container) automatically saves the session: the container is committed to an image and persists at `~/.safe_lab_agents/sessions/<name>/`. Resume it later:

```bash
agent resume --name session-20260413-153042
```

> **Security — your credentials and saved sessions.** Stopping a session saves it as a local image so you can
> resume it. On the way out, the tool tries to strip the login credentials you gave the agent out of that saved
> image, and it doesn't keep them in the session folder either. So resuming asks you to sign in again.
>
> This clean-up is **best-effort, not a guarantee** — a stray copy of a secret could still end up inside a
> saved image or the session files. Treat both as sensitive: don't share saved session images, and rotate the
> key or token if you think one may have leaked.

### See a real run

The example ships with a **complete captured run** where the agent calibrated the optical bench from scratch. See [`example_setup/example_agent_run.md`](example_setup/example_agent_run.md) for the walkthrough, plus the self-contained [HTML data report](https://raw.githack.com/MaxNaeg/safe_lab_agents/main/example_setup/shared_calibration_example/auto_log/report_safe_lab_agents.html) and [conversation transcript](https://raw.githack.com/MaxNaeg/safe_lab_agents/main/example_setup/shared_calibration_example/conversation_safe_lab_agents.html) — the links open directly in your browser (the underlying `.html` files live in `example_setup/shared_calibration_example/` and also render offline).

## Defining tools (reference)

The tools file you write is the agent's entire interface to your hardware. A complete example is [`example_setup/tools.py`](example_setup/tools.py) (with its experiment class in [`example_setup/setup.py`](example_setup/setup.py)); the sections below cover the building blocks.

### Stateful experiments

When your tools share a stateful object — an instrument driver holding a serial/USB connection, a session, etc. — wrap it with `experiment()`:

```python
from safe_lab_agents import experiment
# import any library controlling your hardware
from pylablib.devices.example_provider import ElliptecMotor, ... 

class Setup:
    def __init__(self, port: str = "/dev/ttyUSB0"):
        self.motor = ElliptecMotor(port)
        ...  # open the hardware connection

    def get_position(self, component: str) -> float:
        """Measure the position of an optical component, in millimeters."""
        pos = self.motor.get_position()
        ...
        

    def close(self) -> None:
        self.motor.close()
        ...  # release the connection

exp = experiment(Setup, port="/dev/ttyUSB0")
```

`experiment()` constructs the object **lazily** — on first use, inside the process that runs the tools — so the hardware is opened once, not on every import of the file.

You can expose the experiment's methods as tools in two ways:

```python
# Register a method directly (no wrapper needed):
MCP_TOOLS  = [exp.get_position]

# Or wrap it when you want to transform the result, or change the documentation shown to the agent:
def get_position(component: str) -> float:
    """Measure the position of an optical component, in millimeters."""
    return float(exp.get_position(component))
MCP_TOOLS.append(get_position)
```

### Graceful shutdown hook

If your tools file defines a top-level callable named `GRACEFUL_EXPERIMENT_SHUTDOWN`, it is called automatically when the tool process stops. Use this to put hardware into a safe state (and to close a [stateful experiment](#stateful-experiments): `GRACEFUL_EXPERIMENT_SHUTDOWN = exp.close`):

```python
def _shutdown_instrument():
    """Set the power supply to zero and disable the output."""
    set_angle(0.0, "polarizer")
    # ... close connections, disable outputs, etc.

GRACEFUL_EXPERIMENT_SHUTDOWN = _shutdown_instrument
```

It is called automatically when you exit the container, so use it to release hardware connections and leave instruments in a safe state.


### Python tool client

When `PYTHON_TOOLS` is declared, a `tools_client.py` is auto-generated in the workspace (`/agent/workspace/tools_client.py`) at session start, so the agent — or any script inside Docker — can call tools as regular functions and receive native Python objects:

```python
import sys; sys.path.insert(0, "/agent/workspace")
from tools_client import set_angle, measure_power

set_angle(30.0, "polarizer")
power = measure_power()   # returns a native dict, not JSON text
```

The agent is told about the available Python tools (names, signatures, docstrings) via its system prompt.

> **Safety — argument types are only shallow-checked.** Incoming arguments are validated against your tool's type hints at the **first level only**: a hint of `list[int]` is checked to be a list, but *element* types are **not** inspected — `[1, "two"]` would pass. The agent driving these calls is untrusted, so a tool that requires specific element types, shapes, or bounds must validate them itself and raise a clear error. (For safety, tool **inputs** must be JSON-serializable values or numpy arrays; **return values** can be any Python object, since the host controls them.)

## Automatic logging & ELN export

The `--auto-log` flag records every tool call as a structured ELN entry — no changes to your tools needed. Records land in **`shared_dir/auto_log/`** (or `workspace/auto_log/` if no `--shared` is set); the CLI prints the exact path at startup.

```bash
agent start --tools tools.py --shared ./data/ --auto-log
```

Each call produces a JSON file with the function name, parameters, return values, and timestamps. Numpy arrays are extracted into a companion HDF5 file and replaced with a reference in the JSON:

```json
{
  "title": "measure_power",
  "duration_ms": 234,
  "parameters": {"channel": 1},
  "result": {
    "power": {"value": 2.5, "unit": "W"},
    "trace": {"_type": "ndarray", "file": "exp_….h5", "dataset": "/trace", "shape": [1024]}
  }
}
```


**Recommended format of tool results**

For best results, tools should return a `dict` when `--auto-log` is on — keys become named fields. Non-dict returns are still logged, just less structured.

We also recommend attaching a unit to any measurement value by wrapping it with `quantity(value, unit, term=None)`. Units are opt-in per value — anything you don't wrap stays a plain value.

```python
from safe_lab_agents import quantity, Quantity # for type hints for the agent

def measure_power(channel: int) -> dict[str, Quantity|str]:
    return {
        "power": quantity(2.5, "W"),      # scalar with a unit
        "trace": quantity(samples, "V"),  # numpy array with a unit
        "status": "ok",                   # plain value, no unit
    }
```

Units flow through the logs, HTML report, and .eln export, so the recorded data stays self-describing.



**Batches and analyses.** Tool calls (your experiments) are auto-logged automatically. In addition, the agent is automatically instructed to record its own analyses and reasoning. Inside the container, `auto_log_client.py` exposes helpers for this:

- `start_batch(label, description="")` / `stop_batch()` — group related calls (a sweep, an optimization loop, a multi-step protocol) into a single merged record (one JSON + one HDF5).
- `log_analysis(title, text, data=…, references=…, script=…, figures=…, kind=…)` — record the agent's own analysis, fits, figures, and reasoning. The agent is instructed to log not just successes but hypotheses, decisions, debugging detours, and failures, tagged by `kind` (`analysis`, `hypothesis`, `decision`, `debug`, `failed`, `observation`).

**On exit**, a `session_summary.json` and a standard **`<session>.eln`** archive are written automatically to the log folder.

**View the log** as a single self-contained HTML page (embedded figures, filter/search by kind, provenance links) — see the [example report](https://raw.githack.com/MaxNaeg/safe_lab_agents/main/example_setup/shared_calibration_example/auto_log/report_safe_lab_agents.html) (opens in your browser):

```bash
agent report path/to/auto_log --open
```

**Export to other ELNs** — the log folder can be packaged as a standard [`.eln`](https://github.com/TheELNConsortium/TheELNFileFormat) file (a ZIP wrapping an RO-Crate), importable into **eLabFTW, Kadi4Mat, PASTA, SampleDB, RSpace, and datalab**. This happens automatically on exit; re-export manually after adding analyses on `resume`:

```bash
agent export-eln path/to/auto_log -o session.eln
```

### Kadi4Mat integration
You can automatically push all the log enties to the lab notebook Kadi4Mat:
Install the extra (`pip install -e ".[kadi4mat]"`), configure once with `kadi-apy config`, then pass `--kadi4mat-project <name>` to push every logged record to a [Kadi4Mat](https://kadi4mat.iam.kit.edu) ELN (auto-enables `--auto-log`; rate-limited via `--kadi-max-per-minute`/`--kadi-max-per-session`).

## CLI Reference

### Global options

These apply to every subcommand and must come **before** the subcommand (e.g. `agent --log-level DEBUG start …`).

| Option | Description |
|--------|-------------|
| `--log-level` | Logging verbosity: `DEBUG`, `INFO`, `WARNING` (default), or `ERROR`. Also read from the `LOG_LEVEL` environment variable. Applies to the MCP server subprocess too, so tool-call / auto-log debug output is included. Logs go to stderr. |

### `agent start`

Start a new agent session. All options are optional — missing ones are prompted interactively.

| Option | Description |
|--------|-------------|
| `--agent`, `-a` | Agent type: `claude-code` or `openclaw` |
| `--tools`, `-t` | Path to Python file with MCP tool functions |
| `--context`, `-c` | Directory with experiment context (mounted read-only) |
| `--shared`, `-s` | Shared directory for data exchange (mounted read-write) |
| `--task` | Initial task for autonomous mode |
| `--task-file` | Path to a text/markdown file whose content is the initial task (mutually exclusive with `--task`) |
| `--name`, `-n` | Session name (auto-generated if omitted) |
| `--server` | Predefined MCP servers to enable (repeatable). See [More features](#more-features). |
| `--kadi4mat-project` | Kadi4Mat project name. Enables Kadi4Mat ELN push and auto-enables `--auto-log`. See [More features](#more-features). |
| `--kadi-max-per-minute` | Kadi4Mat: max records per minute (default 10) |
| `--kadi-max-per-session` | Kadi4Mat: max records per session (default 500, `0` = unlimited) |
| `--requirements`, `-r` | `requirements.txt` for extra Python packages in Docker |
| `--rebuild` | Force a full image rebuild (`--no-cache --pull`), ignoring the build cache. See [Python Packages in Docker](#python-packages-in-docker). |
| `--agent-args` | Agent-specific argument as `KEY=VALUE` (repeatable). See [Agent-Specific Arguments](#agent-specific-arguments). |
| `--port` | MCP server port (0 = auto) |
| `--container` | Container runtime: `docker` or `podman` (prompted if omitted; Podman auto-initializes the machine if needed) |
| `--no-web` | Disable web tools (soft restriction — does not block network access). Claude Code: built-in web tools disabled, but Bash can still reach the network. OpenClaw: system-prompt instruction only. |
| `--egress-lockdown/--no-egress-lockdown` | In-container egress firewall (default: **on**): the host is reachable only on the MCP port and private/LAN ranges are blocked, while the public internet (the agent's model API) stays open. If the rules cannot be applied the container refuses to start (fail-closed) — disable only if your runtime cannot support in-container iptables. |
| `--mem-limit` | Container memory limit, e.g. `8g` or `512m`. Default: half the RAM visible to the container runtime (min 2 GB). Swap is disabled alongside, so the limit is a hard ceiling (the container is OOM-killed instead of swap-thrashing the host). |
| `--cpu-limit` | Container CPU limit in cores, e.g. `2` or `2.5`. Default: all but one of the runtime's cores, so the host-side MCP tool server stays responsive. |
| `--update-tools` | Expose a `reload_tools` MCP tool the agent can call to reload your tools file without restarting the container |
| `--auto-log` | Automatically log every tool call as a local ELN record (JSON + HDF5). See [Automatic logging & ELN export](#automatic-logging--eln-export). |
| `--config` | Path to a YAML config file supplying defaults for the options above. See [Config file](#config-file). |
| `--no-config` | Do not auto-discover `safe-lab-agents.config.yaml` in the current directory. |

### Config file

Instead of retyping the same flags every run, store defaults in a YAML file. Keys are exactly the flag names with the leading `--` stripped (kept hyphenated). A flag passed on the command line always overrides the config file (a warning is printed when it does); a value from the file means the interactive wizard won't prompt for it.

By default, `start` auto-discovers `safe-lab-agents.config.yaml` in the current directory. Use `--config <path>` to point at a specific file, or `--no-config` to ignore auto-discovery.

```yaml
# safe-lab-agents.config.yaml
agent: claude-code
tools: ./tools.py            # paths resolve relative to THIS file's directory
context: ./context
shared: ./shared
auto-log: true
agent-args:
  model: opus
  effort: max
```

With that file in place, `agent start` runs with no flags, and e.g. `agent start --agent openclaw` overrides just the agent. Path-valued keys (`tools`, `context`, `shared`, `requirements`, `task-file`) resolve relative to the config file's location, so the file is portable.

### `agent resume`

Resume a previously stopped session. Resume is always interactive — a session that originally ran autonomously (`--task`) is continued interactively so you can drive it by hand. The container runtime is autodetected from the saved session metadata.

| Option | Description |
|--------|-------------|
| `--name`, `-n` | Session to resume (prompted if omitted) |
| `--agent-args` | Override agent-specific args for this resume (repeatable, same syntax as `start`) |

### `agent history`

View the conversation history of a session. Without flags it prints to the terminal with rich formatting; `--html`/`--open` instead render a self-contained HTML viewer (role-colored cards, filter/search, collapsible tool calls, inlined agent-read images). Works for both `claude-code` and `openclaw` sessions.

| Option | Description |
|--------|-------------|
| `--name`, `-n` | Session name (prompted if omitted) |
| `--last`, `-l` | Show only the last N entries |
| `--html`, `-o` | Write a self-contained HTML viewer instead of printing (default: `<session>/conversation_safe_lab_agents.html`) |
| `--open` | Open the HTML conversation viewer in the default browser |

### `agent list`

List all saved sessions as a table, including a `Container` column showing the runtime each session used. Takes no options.

### `agent report`

Build a self-contained HTML report from an auto-log folder.

| Option | Description |
|--------|-------------|
| `LOG_DIR` | Path to an `auto_log/` folder (positional, required) |
| `--output`, `-o` | Output HTML path (default: `<log_dir>/report_safe_lab_agents.html`) |
| `--open` | Open the report in the default browser when done |

### `agent export-eln`

Export an auto-log folder as a standard `.eln` (RO-Crate) file for import into other ELNs.

| Option | Description |
|--------|-------------|
| `LOG_DIR` | Path to an `auto_log/` folder (positional, required) |
| `--output`, `-o` | Output `.eln` path (default: `<log_dir>/<session>.eln`) |
| `--name` | Human name for the session (root `Dataset`) |
| `--author` | Optional human author name to attribute records to |
| `--affiliation` | Optional organisation for the author |

## Agent-Specific Arguments

Agent backends declare their own accepted arguments. Pass them with `--agent-args KEY=VALUE` (or just `KEY` for boolean flags). The flag can be repeated:

```bash
agent start --tools tools.py --agent-args effort=high --agent-args dangerously-skip-permissions
```

If a required argument is missing the CLI will prompt for it automatically.

### Claude Code

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `model` | string | — | Claude model alias or full ID (e.g. `sonnet`, `opus`, `claude-sonnet-4-6`) |
| `effort` | string | `low` | Effort level: `low`, `medium`, `high`, `xhigh`, `max` |
| `copy-host-credentials` | bool | `false` | Copy Claude login credentials from the host into the container. Off by default — without it, log in inside the container or pass `oauth-token` |
| `oauth-token` | string | — | Claude OAuth token (`sk-ant-oat…`) to authenticate with directly. Never stored in session metadata. See [Authentication](#claude-code-authentication) |
| `dangerously-skip-permissions` | bool | `false` | Pass `--dangerously-skip-permissions` to Claude Code (interactive mode) |

None of these are required — Claude Code works without any `--agent-args`.

#### Claude Code authentication

Claude Code uses a Claude **subscription** (Pro/Max) — no API key is needed. The CLI resolves credentials in this order:

1. **`oauth-token` agent arg** — if you pass `--agent-args oauth-token=sk-ant-oat…`, that token is injected directly (as `CLAUDE_CODE_OAUTH_TOKEN`) and the steps below are skipped. The token is marked secret and is **never written to session metadata**, so it must be re-supplied when you `resume` a session that relies on it. Generate one on any logged-in machine with `claude setup-token`.
2. **Host credentials** — pass `copy-host-credentials` and, if the host running `safe-lab-agents` is logged into Claude Code, its credentials are copied into the container. This is **off by default**; without it the CLI falls through to the in-container login below.
3. **In-container login** — if the host is *not* logged in:
   - **Interactive mode:** just log in inside the container session as usual.
   - **Autonomous mode:** the CLI launches a one-time login first — it runs `claude setup-token` inside a throwaway container, prints a sign-in URL, you open it and paste the code back, and the resulting token is captured and used for the run. After that, the autonomous task starts automatically.

> **Resuming asks you to sign in again.** Your login isn't kept in the saved session (see the note under [Stop and resume](#stop-and-resume)), so resuming a Claude Code session logs in again — re-run the in-container login, or pass `--agent-args oauth-token=…`. If you used `copy-host-credentials`, it re-copies from your machine automatically instead. Your conversation is preserved either way.

> **Security:** passing a token on the command line leaves it in your shell history and process list. Prefer `oauth-token` for short-lived/CI use, or rely on host credentials / the in-container login otherwise.

### OpenClaw

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `api-key` | string | — | API key for the LLM provider (required, secret). Not saved with the session — you re-enter it on `resume` (prompted, or `--agent-args api-key=…`) |
| `provider` | string | — | LLM provider: `anthropic`, `openai`, `google`, `openrouter` (required) |
| `model` | string | — | Model name, e.g. `gpt-4o`, `claude-sonnet-4-6` (required) |

All three are required — if omitted, the CLI prompts for them. Because the key is not persisted, `agent resume`
re-prompts for it (pass `--agent-args api-key=…` to supply it non-interactively).

```bash
agent start --tools tools.py \
    --agent-args provider=anthropic --agent-args model=claude-sonnet-4-6 --agent-args api-key=sk-…
```

## Autonomous Mode

Give the agent a task and let it run to completion without interaction:

```bash
agent start \
    --agent claude-code \
    --tools tools.py \
    --shared ./data/ \
    --task "Run a polarizer sweep from 0 to 180° in 10° steps, measuring power at each angle. Save all results to a CSV file and create a summary plot."
```

The agent's output streams to your terminal. When the task finishes (or you press `Ctrl+C`), the session is saved.

## Python Packages in Docker

The Docker container comes with Python 3 and common scientific packages pre-installed (`numpy`, `pandas`, `matplotlib`, `scipy`, `h5py`). To install more, create a `requirements.txt` and pass it with `--requirements`:

```bash
agent start --tools tools.py --requirements my_requirements.txt
```

The packages are installed at image build time and cached, keyed on the Dockerfile, entrypoint, and requirements file — so a rebuild triggers automatically when any of those change.

> **Note — the cache does not see upstream updates.** The base image and the agent/Python toolchain are installed unpinned, so a "latest" tag that moves upstream will **not** trigger a rebuild — you can keep running a stale toolchain indefinitely. Pass `--rebuild` to force a full rebuild that ignores the cache and re-pulls the base image (`--no-cache --pull`).

## More features

Smaller or optional capabilities, one line each:

- **Agent workspace** — the agent's working directory (`/agent/workspace` inside the container — scripts, analysis, output files) is bind-mounted to `~/.safe_lab_agents/sessions/<name>/workspace/` on your host, so everything it creates is available there.
- **`--update-tools`** — exposes a `reload_tools` MCP tool the agent can call to pick up edits to your tools file without restarting the container (handy while developing tools).
- **`--no-web`** — soft-disable the agent's web tools (a lab agent driving hardware usually shouldn't browse); does not block network access.
- **Egress lockdown** (default on) — before the agent starts, the container installs an internal firewall so the host is reachable *only* on the MCP tool-server port and private/LAN addresses are blocked; the public internet stays open for the agent's model API. Note: LAN machines with public IP addresses can't be told apart from the internet and remain reachable. Opt out with `--no-egress-lockdown` if your runtime can't support in-container iptables (the container fails closed otherwise).
- **`--port`** — pin the host-side MCP server port (default auto-selects a free one).
- **`@results_to_shared`** — decorator that copies selected return values of an MCP tool into the shared directory and hands the agent a confirmation string (a niche helper — `PYTHON_TOOLS` already return native objects directly). `from safe_lab_agents import results_to_shared`.
- **`@no_autolog`** — decorator to exclude a specific tool from auto-logging. `from safe_lab_agents import no_autolog`.
- **Predefined MCP servers** (`--server`) — enable a built-in bundle of tools by name; for example, `--server lab-notebook` adds a simple Markdown-based lab-notebook server.
- **Podman support** — Docker and Podman are equal choices per session (`--container`); the Podman machine (macOS/Windows) or socket (Linux) is started automatically, and the runtime is autodetected on `resume`.
- **Docker auto-start** — Docker Desktop is launched automatically on macOS/Windows (and the daemon started via `systemctl` on Linux) if it isn't already running.

## Troubleshooting

### Podman on Windows

Podman on Windows must be installed with the **WSL backend** (the default for `podman machine init`). The agent container reaches the host-side tool server through the WSL virtual network, and that path is what Safe Lab Agents resolves automatically. The Hyper-V backend (`--provider hyperv`) uses a different network layout and is **not supported** — the container will not be able to reach the tool server.

On the first `--container podman` run, if the required firewall rule is missing, the CLI prints a one-time command to add it. The agent container reaches the host's tool server over the WSL virtual adapter, which the Windows firewall blocks by default. Run the printed command once in an **Administrator PowerShell** — for example:

```powershell
New-NetFirewallRule -DisplayName 'safe-lab-agents-mcp' -Direction Inbound -Action Allow -Protocol TCP -InterfaceAlias 'vEthernet (WSL)'
```

The rule is scoped to the WSL adapter only, so the tool server is **not** exposed to the rest of your network — and even reachable clients must present the session's mandatory MCP bearer token. Until the rule is added, tool calls from the agent will time out.

### Podman on Linux

You may need to additionally install:

```bash
sudo apt install uidmap         # user-namespace ID mapping, required for rootless Podman
sudo apt install podman-docker  # provides a `docker` command that transparently calls Podman
```

and set:

```bash
# Allow unprivileged user namespaces (some distros restrict them via AppArmor), which
# rootless containers need:
sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0
```

**Build fails with "Release file is not valid yet"**

If the build fails with an error like `E: Release file ... is not valid yet (invalid for another Xh Ymin Zs).`, the BuildKit container's clock is out of sync with the host. This can happen after the host clock is corrected (e.g. via NTP sync) while Podman's internal state still runs with the old time.

```bash
sudo timedatectl set-ntp true   # sync the host clock
podman system reset             # restart Podman's internal state
```

> **Warning:** `podman system reset` removes **all** Podman containers, images, volumes, and networks on your machine — not just those created by Safe Lab Agents. Make sure you have no other Podman workloads you need to preserve before running this command.

After the reset, retry your `agent start --container podman ...` command. The Docker image will be rebuilt from scratch on the first run.
