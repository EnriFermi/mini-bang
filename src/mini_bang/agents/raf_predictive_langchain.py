from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from langchain.schema.runnable import RunnableLambda

from mini_bang.agents.base import AgentBase
from mini_bang.framework import TaskEnvironment, TaskSubmission
from mini_bang.tasks.raf_predictive.predictive_api import RAFPredictiveAPI


@dataclass
class _PredictiveContext:
    api: RAFPredictiveAPI
    metadata: Dict[str, Any]


class LangChainRAFPredictiveAgent(AgentBase):
    """Baseline predictive agent using empirical RAF frequencies."""

    def __init__(self) -> None:
        self._pipeline = RunnableLambda(self._summarise) | RunnableLambda(self._emit)

    def solve(self, task: TaskEnvironment) -> TaskSubmission:
        if not isinstance(task.api, RAFPredictiveAPI):
            raise TypeError("LangChainRAFPredictiveAgent expects a RAFPredictiveAPI")
        answer = self._pipeline.invoke({"context": _PredictiveContext(api=task.api, metadata=task.metadata or {})})
        return TaskSubmission(answer=answer)

    def _summarise(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        context: _PredictiveContext = payload["context"]
        meta = context.metadata
        sequences = meta.get("sequences", [])
        seeds = [int(s) for s in meta.get("test_seeds", [])]
        stats: Dict[str, Dict[str, float]] = {}
        for seq in sequences:
            seq_id = seq.get("id") or "seq"
            sats = [int(t) for t in seq.get("saturations", [])]
            per_sat: Dict[str, list[bool]] = {str(t): [] for t in sats}
            for seed in seeds:
                resp = context.api.generate_samples(
                    saturation=sats,
                    runs=1,
                    snapshot_times=None,
                    extras=["is_raf"],
                    macro_params={"seed": seed},
                )
                for entry in resp.get("sequence", []):
                    sat_key = str(entry.get("saturation"))
                    per_sat.setdefault(sat_key, []).append(bool(entry.get("is_raf", False)))
            stats[seq_id] = {
                sat_key: (sum(1 for v in values if v) / len(values)) if values else 0.5
                for sat_key, values in per_sat.items()
            }
        return {"probabilities": stats}

    def _emit(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return {"probabilities": payload["probabilities"]}
