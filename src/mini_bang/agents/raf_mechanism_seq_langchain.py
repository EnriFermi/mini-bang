from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from langchain.schema.runnable import RunnableLambda

from mini_bang.agents.base import AgentBase
from mini_bang.framework import TaskEnvironment, TaskSubmission
from mini_bang.tasks.raf_mechanism_seq.sequence_api import RAFMechanismSequenceAPI


@dataclass
class _SeqContext:
    api: RAFMechanismSequenceAPI


class LangChainRAFMechanismSequenceAgent(AgentBase):
    """Baseline agent that predicts Poisson means from training averages."""

    def __init__(self) -> None:
        self._pipeline = RunnableLambda(self._estimate_means)

    def solve(self, task: TaskEnvironment) -> TaskSubmission:
        if not isinstance(task.api, RAFMechanismSequenceAPI):
            raise TypeError("LangChainRAFMechanismSequenceAgent expects RAFMechanismSequenceAPI")
        answer = self._pipeline.invoke({"context": _SeqContext(api=task.api)})
        return TaskSubmission(answer=answer)

    def _estimate_means(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        context: _SeqContext = payload["context"]
        data = context.api.get_training_data()
        predictions: Dict[str, Dict[str, float]] = {}

        for seq_id, info in data["sequences"].items():
            final_means: Dict[str, float] = {}
            counts: Dict[str, int] = {}
            for item in info.get("train", []):
                steps = item.get("steps", [])
                if not steps:
                    continue
                final_step = steps[-1]
                for traj in final_step.get("trajectories", []):
                    for species, series in traj.items():
                        species_key = str(species)
                        final_means.setdefault(species_key, 0.0)
                        counts.setdefault(species_key, 0)
                        final_means[species_key] += float(series[-1])
                        counts[species_key] += 1
            for species, total in list(final_means.items()):
                denom = counts.get(species, 1)
                final_means[species] = total / denom if denom else 0.0
            predictions[seq_id] = final_means

        return {"predicted_means": predictions}
