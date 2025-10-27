# mini-bang Benchmark

mini-bang is an experimental benchmark for evaluating autonomous research agents on multi-scale ``mini-bang`` simulators. The repository currently exposes the RAF (Reflexively Autocatalytic and Food-generated) family of simulators together with example agent entry points.

## Repository Layout

```
src/mini_bang/
  agent_connector.py      # High level harness for running agents against tasks
  agents/                 # Agent implementations (LangChain, etc.)
  mcp/                    # MCP transport (server/client/config)
  simulators/             # Simulator definitions discovered at runtime
  tasks/                  # Task definitions, configs, loaders
```


![SVG Image](figures/Benchmark.svg)

Visual diagram explaining relations between benchmark components

## Available Tasks

| ID | Level | Name | Objective | Baseline Agent |
| --- | --- | --- | --- | --- |
| `raf/timing-v1` | 2 | Anomaly Detection | Estimate first-appearance distributions for molecules at a fixed saturation | `LangChainRAFTimingAgent` |
| `raf/signature-v1` | 4 | Signature Recognition | Decide whether simulators are RAF across saturation levels | `LangChainRAFSignatureAgent` |
| `raf/mechanism-prob-v1` | 5 | Mechanism Probability | Predict probability of RAF formation as a function of saturation | `LangChainRAFMechanismProbabilityAgent` |
| `raf/mechanism-seq-v1` | 5 | Mechanism Sequence | Model the distribution of the final trajectory in RAF sequences | `LangChainRAFMechanismSequenceAgent` |
| `raf/predictive-v1` | 6 | Predictive Generalisation | Forecast RAF emergence probabilities in monotonic sequences | `LangChainRAFPredictiveAgent` |

Agent-facing task APIs expose a single entry point for on-demand simulation:

```python
task.api.generate_samples(
    saturation=18,
    runs=2,
    snapshot_times=[0.25, 0.5, 0.75, 1.0],
    extras=["is_raf", "first_hits"],
    macro_params={"seed": 123},
)
```

Each task enforces a per-task call limit (see the `api.max_generate_calls` field in the task config). Training “snapshots” are not exposed directly; agents must gather the data they need using `generate_samples` within the limit.

## Running Baseline Agents

### RAF Timing (Level 2)

The `LangChainRAFTimingAgent` gathers samples via `generate_samples` and outputs per-species first-hit distributions. Execute it through the connector. The connector uses the MCP client config at `src/mini_bang/mcp/mcp.config.json` to connect (stdio auto‑spawn by default, or HTTP if configured).

```python
from mini_bang.agent_connector import AgentConnector
from mini_bang.agents import LangChainRAFTimingAgent

connector = AgentConnector()
agent = LangChainRAFTimingAgent()
result = connector.run(agent, task_id="raf/timing-v1")

print(result.validation.details)
print(result.validation.metrics)
```

`result.submission.answer` contains the raw `{"distributions": ...}` payload produced by the agent.

### RAF Signature Recognition (Level 4)

```python
from mini_bang.agent_connector import AgentConnector
from mini_bang.agents import LangChainRAFSignatureAgent

connector = AgentConnector()
agent = LangChainRAFSignatureAgent()
result = connector.run(agent, task_id="raf/signature-v1")
print(result.validation.details)
```

### Mechanism Probability (Level 5)

```python
from mini_bang.agent_connector import AgentConnector
from mini_bang.agents import LangChainRAFMechanismProbabilityAgent

connector = AgentConnector()
agent = LangChainRAFMechanismProbabilityAgent()
result = connector.run(agent, task_id="raf/mechanism-prob-v1")
print(result.validation.details)
```

### Mechanism Sequence (Level 5)

```python
from mini_bang.agent_connector import AgentConnector
from mini_bang.agents import LangChainRAFMechanismSequenceAgent

connector = AgentConnector()
agent = LangChainRAFMechanismSequenceAgent()
result = connector.run(agent, task_id="raf/mechanism-seq-v1")
print(result.validation.details)
```

### Predictive Generalisation (Level 6)

```python
from mini_bang.agent_connector import AgentConnector
from mini_bang.agents import LangChainRAFPredictiveAgent

connector = AgentConnector()
agent = LangChainRAFPredictiveAgent()
result = connector.run(agent, task_id="raf/predictive-v1")
print(result.validation.details)
```

## Creating a New Agent

Agents must satisfy the `AgentProtocol` by implementing `solve(self, task: TaskEnvironment)`. They can be plain Python classes or integrate frameworks such as LangChain. The connector wraps non-`TaskSubmission` return values automatically.

Template:

```python
from mini_bang.framework import TaskSubmission
from mini_bang.agents.base import AgentBase

class MyAgent(AgentBase):
    def solve(self, task):
        # interact with task.api
        answer = {...}
        return TaskSubmission(answer=answer)
```

## Adding a New Task

1. **Create task package**: add `src/mini_bang/tasks/<my_task>/` containing:
   - `config.json` — description, metadata, dataset spec, validation thresholds.
   - `task_api.py` — optional facade exposing task-specific helper methods.
   - `task.py` — subclass of `ConfiguredSimulationTask` or `RemoteSimulationTask`.

2. **Register task**: import and register the task class in `src/mini_bang/tasks/loader.py` so the connector can discover it.

3. **(Optional)** Implement sample agents under `src/mini_bang/agents/` for quick experimentation.

The RAF timing task is an end-to-end example (`src/mini_bang/tasks/raf_timing/`).

## Adding a New Simulator

1. **Simulator package**: create `src/mini_bang/simulators/<sim_id>/` with:
   - `config.json` — contains `id`, `description`, `factory` (dotted path), optional `response_builder`, metadata, and default parameters under `defaults`.
   - `factory.py` — exposes a factory function returning a `MacroSimulatorBase` instance.
   - `response.py` (optional) — returns a callable that formats simulator output (`extras` calculations, derived statistics, etc.).
   - Micro/macro simulator implementations (`macro/`, `micro/`, `utils/`).

2. **Auto-discovery**: the registry scans every subpackage of `mini_bang.simulators` for a `config.json`, so no additional wiring is needed once the files exist.

3. **Extras**: declare advanced response options (e.g., `first_hits`, `is_raf`) within the response builder and reference them from tasks via the `extras` list in API requests.

## MCP Simulation Server

The benchmark exposes a Machine Control Protocol (MCP) server implemented with FastMCP. The client uses the canonical MCP config at `src/mini_bang/mcp/mcp.config.json`.

- Local (stdio, auto‑spawn): use the provided stdio config
  - File: `src/mini_bang/mcp/mcp.config.stdio.json`
  - To enable: copy over the active config
    - `cp src/mini_bang/mcp/mcp.config.stdio.json src/mini_bang/mcp/mcp.config.json`

- Remote (HTTP/streamable): run the server and point the config URL to it
  - Start server: `PYTHONPATH=src python -m mini_bang.mcp.server --transport http --host 127.0.0.1 --port 8000`
  - Example config (`src/mini_bang/mcp/mcp.config.json`):
    ```json
    {
      "mcpServers": {
        "mini-bang": {
          "url": "http://127.0.0.1:8000/mcp/",
          "transport": "http",
          "description": "Remote Mini‑Bang MCP server (HTTP)"
        }
      }
    }
    ```

All agents ultimately call the unified MCP tool `get_simulation` with payload fields: `simulator_id`, `saturation` (int or list), `runs`, optional `snapshot_times`, `extras`, and parameter dictionaries for `macro_params`, `micro_params`, and `sample_params`.

Server logs are written to `logs/mcp_server.log` by default (configurable with `--log-file` or `MINI_BANG_MCP_LOG`).

## Environment Setup

```bash
pip install "fastmcp>=2.13"
```
---
