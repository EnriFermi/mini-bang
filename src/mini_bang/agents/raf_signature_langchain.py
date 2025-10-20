from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from langchain.schema.runnable import RunnableLambda

from mini_bang.agents.base import AgentBase
from mini_bang.framework import TaskEnvironment, TaskSubmission
from mini_bang.tasks.raf_signature.signature_api import RAFSignatureAPI


@dataclass
class _SignatureContext:
    api: RAFSignatureAPI
    metadata: Dict[str, Any]


class LangChainRAFSignatureAgent(AgentBase):
    """Baseline agent for RAF signature recognition using simple frequency estimates."""

    def __init__(self) -> None:
        self._pipeline = RunnableLambda(self._gather_stats) | RunnableLambda(self._emit_predictions)

    def solve(self, task: TaskEnvironment) -> TaskSubmission:
        if not isinstance(task.api, RAFSignatureAPI):
            raise TypeError("LangChainRAFSignatureAgent expects a RAFSignatureAPI")
        context = _SignatureContext(api=task.api, metadata=task.metadata or {})
        answer = self._pipeline.invoke({"context": context})
        return TaskSubmission(answer=answer)

    def _gather_stats(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        context: _SignatureContext = payload["context"]
        data = context.api.get_training_data()
        stats: Dict[str, float] = {}
        available_seeds: Dict[str, set[str]] = {}
        for sat, samples in data["simulators"].items():
            if not samples:
                stats[sat] = 0.5
                available_seeds[sat] = set()
                continue
            ratio = sum(1.0 for item in samples if item.get("is_raf")) / len(samples)
            stats[sat] = ratio
            available_seeds[sat] = {str(item.get("macro_seed")) for item in samples}
        meta = dict(context.metadata)
        return {
            "saturations": data["simulators"].keys(),
            "stats": stats,
            "metadata": meta,
            "available_seeds": available_seeds,
        }

    def _emit_predictions(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        stats: Dict[str, float] = payload["stats"]
        metadata: Dict[str, Any] = payload["metadata"]
        available_seeds: Dict[str, set[str]] = payload["available_seeds"]
        test_seeds = [str(s) for s in metadata.get("test_seeds", [])]

        predictions: Dict[str, Dict[str, Dict[str, float]]] = {}
        for sat, prob in stats.items():
            seed_map: Dict[str, Dict[str, float]] = {}
            targets = set(test_seeds)
            targets.update(available_seeds.get(sat, set()))
            if not targets:
                targets = {"default"}
            for seed in targets:
                seed_map[seed] = {"probability": float(prob)}
            predictions[sat] = seed_map

        return {"predictions": predictions}
