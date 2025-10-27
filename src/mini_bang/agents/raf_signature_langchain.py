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
        meta = context.metadata
        saturations = [int(t) for t in meta.get("saturations", [])]
        snapshot_times = [float(t) for t in meta.get("snapshot_times", [])]
        seeds = [int(s) for s in meta.get("test_seeds", [])]
        stats: Dict[str, float] = {}
        available_seeds: Dict[str, set[str]] = {}
        for sat in saturations:
            sat_key = str(sat)
            outcomes = []
            for seed in seeds:
                resp = context.api.generate_samples(
                    saturation=sat,
                    runs=1,
                    snapshot_times=snapshot_times,
                    extras=["is_raf"],
                    macro_params={"seed": seed},
                )
                outcomes.append(bool(resp.get("is_raf", False)))
            stats[sat_key] = (sum(1 for v in outcomes if v) / len(outcomes)) if outcomes else 0.5
            available_seeds[sat_key] = {str(s) for s in seeds}
        return {
            "saturations": [str(s) for s in saturations],
            "stats": stats,
            "metadata": dict(meta),
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
