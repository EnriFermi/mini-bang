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


class LangChainRAFMechanismProbabilityAgent(AgentBase):
    """Baseline agent that estimates RAF probability via empirical frequencies."""

    def __init__(self) -> None:
        self._pipeline = RunnableLambda(self._compute_probabilities)

    def solve(self, task: TaskEnvironment) -> TaskSubmission:
        if not isinstance(task.api, RAFMechanismProbabilityAPI):
            raise TypeError("LangChainRAFMechanismProbabilityAgent expects RAFMechanismProbabilityAPI")
        answer = self._pipeline.invoke({"context": _ProbContext(api=task.api)})
        return TaskSubmission(answer=answer)

    def _compute_probabilities(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        context: _ProbContext = payload["context"]
        data = context.api.get_training_data()
        probabilities: Dict[str, float] = {}
        for sat, samples in data["samples"].items():
            if not samples:
                probabilities[sat] = 0.5
                continue
            count = sum(1.0 for item in samples if item.get("is_raf"))
            probabilities[sat] = count / len(samples)
        return {"probabilities": probabilities}
