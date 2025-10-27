from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from langchain.schema.runnable import RunnableLambda

from mini_bang.agents.base import AgentBase
from mini_bang.framework import TaskEnvironment, TaskSubmission
from mini_bang.tasks.raf_mechanism_prob.probability_api import RAFMechanismProbabilityAPI


@dataclass
class _ProbContext:
    api: RAFMechanismProbabilityAPI
    metadata: Dict[str, Any]


class LangChainRAFMechanismProbabilityAgent(AgentBase):
    """Baseline agent that estimates RAF probability via empirical frequencies."""

    def __init__(self) -> None:
        self._pipeline = RunnableLambda(self._compute_probabilities)

    def solve(self, task: TaskEnvironment) -> TaskSubmission:
        if not isinstance(task.api, RAFMechanismProbabilityAPI):
            raise TypeError("LangChainRAFMechanismProbabilityAgent expects RAFMechanismProbabilityAPI")
        answer = self._pipeline.invoke({"context": _ProbContext(api=task.api, metadata=task.metadata or {})})
        return TaskSubmission(answer=answer)

    def _compute_probabilities(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        context: _ProbContext = payload["context"]
        meta = context.metadata
        saturations = [int(t) for t in meta.get("saturations", [])]
        snapshot_times = [float(t) for t in meta.get("snapshot_times", [])]

        # Use full call budget across sats with unique seeds
        remaining = getattr(context.api, "remaining_calls", lambda: 10**9)()
        if remaining <= 0:
            return {"probabilities": {str(s): 0.5 for s in saturations}}

        positives: Dict[str, int] = {str(s): 0 for s in saturations}
        counts: Dict[str, int] = {str(s): 0 for s in saturations}
        seed_base = 40_000
        calls = 0
        idx = 0
        while calls < remaining and saturations:
            sat = saturations[idx % len(saturations)]
            key = str(sat)
            seed = seed_base + calls
            try:
                resp = context.api.generate_samples(
                    saturation=sat,
                    runs=1,
                    snapshot_times=snapshot_times,
                    extras=["is_raf"],
                    macro_params={"seed": seed},
                )
            except Exception:
                break
            if bool(resp.get("is_raf", False)):
                positives[key] += 1
            counts[key] += 1
            calls += 1
            idx += 1

        probabilities: Dict[str, float] = {}
        for sat in saturations:
            key = str(sat)
            c = counts.get(key, 0)
            probabilities[key] = (positives.get(key, 0) / c) if c else 0.5
        return {"probabilities": probabilities}
