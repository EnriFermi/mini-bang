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


class LangChainRAFPredictiveAgent(AgentBase):
    """Baseline predictive agent using empirical RAF frequencies."""

    def __init__(self) -> None:
        self._pipeline = RunnableLambda(self._summarise) | RunnableLambda(self._emit)

    def solve(self, task: TaskEnvironment) -> TaskSubmission:
        if not isinstance(task.api, RAFPredictiveAPI):
            raise TypeError("LangChainRAFPredictiveAgent expects a RAFPredictiveAPI")
        answer = self._pipeline.invoke({"context": _PredictiveContext(api=task.api)})
        return TaskSubmission(answer=answer)

    def _summarise(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        context: _PredictiveContext = payload["context"]
        data = context.api.get_training_data()
        stats: Dict[str, Dict[str, float]] = {}
        for seq_id, info in data["sequences"].items():
            seq_stats: Dict[str, float] = {}
            saturations = info.get("saturations", [])
            for sat in saturations:
                sat_key = str(sat)
                values = [record["is_raf"].get(sat_key, False) for record in info.get("train", [])]
                if not values:
                    seq_stats[sat_key] = 0.5
                else:
                    seq_stats[sat_key] = sum(1.0 for v in values if v) / len(values)
            stats[seq_id] = seq_stats
        return {"probabilities": stats}

    def _emit(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return {"probabilities": payload["probabilities"]}
