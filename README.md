# mini-bang Benchmark

mini-bang is an experimental benchmark for evaluating autonomous research agents on multi-scale ``mini-bang`` simulators. The repository currently exposes the RAF (Reflexively Autocatalytic and Food-generated) family of simulators together with example agent entry points.

## Repository Layout

```
src/mini_bang/
  agent_connector.py      # High level harness for running agents against tasks
  agents/                 # Agent implementations (LangChain, etc.)
  api/server.py           # HTTP simulation service (macro/micro orchestration)
  runtime/server_manager.py# Lazy server bootstrapper
  simulators/             # Simulator definitions discovered at runtime
  tasks/                  # Task definitions, configs, loaders
sandbox/                  # Scratch space / legacy playground (not used by core libs)
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

Tasks expose labelled training data via `get_training_data()`. Tasks that allow additional exploration also provide `simulate(...)`, e.g.:

```python
fresh = task.api.simulate(
    saturation=18,
    runs=2,
    snapshot_times=[0.25, 0.5, 0.75, 1.0],
    extras=["is_raf", "first_hits"],
    macro_params={"seed": 123}
)
```

## Running Baseline Agents

### RAF Timing (Level 2)

The `LangChainRAFTimingAgent` consumes the RAF timing dataset and outputs per-species first-hit distributions. You can execute it through the connector:

```python
from mini_bang.agent_connector import AgentConnector
from mini_bang.agents import LangChainRAFTimingAgent

connector = AgentConnector()
agent = LangChainRAFTimingAgent()
result = connector.run(agent, task_id="raf/timing-v1")

print(result.validation.details)
print(result.validation.metrics)
```

The connector automatically starts the HTTP simulation server (if it is not already running), loads the RAF timing dataset, and validates the agent output. `result.submission.answer` contains the raw `{"distributions": ...}` payload produced by the agent.

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

## Launching the HTTP Simulation Server Manually

Although the connector manages the server automatically, you can start it explicitly:

```bash
python -m mini_bang.api.server
```

The server exposes:
- `GET /health`
- `GET /simulators`
- `POST /simulate/<sim_id>/generate`

Clients should provide `macro_params`, `micro_params`, `sample_params`, and optional `extras` to control the response payload.

Example request for the RAF simulator:

```bash
curl -s \
  -X POST http://127.0.0.1:8765/simulate/raf/generate \
  -H 'Content-Type: application/json' \
  -d '{
        "saturation": 18,
        "runs": 5,
        "macro_params": {"seed": 123},
        "micro_params": {"max_raf": false, "prune_catalysts": false},
        "sample_params": {"snapshot_times": [0.25, 0.5, 0.75, 1.0]},
        "extras": ["is_raf", "first_hits"]
      }'
```

The response contains sampled trajectories along with any requested extras (e.g., `is_raf`, `first_hits`).

## Environment Setup

```bash
pip install -r requirements.txt
```

If you rely on LangChain agents, configure your preferred LLM environment variables (e.g., `OPENAI_API_KEY`) before invocation.

---
