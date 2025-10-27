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
    metadata: Dict[str, Any]


class LangChainRAFMechanismSequenceAgent(AgentBase):
    """Baseline agent that predicts Poisson means from training averages."""

    def __init__(self) -> None:
        self._pipeline = RunnableLambda(self._estimate_means)

    def solve(self, task: TaskEnvironment) -> TaskSubmission:
        if not isinstance(task.api, RAFMechanismSequenceAPI):
            raise TypeError("LangChainRAFMechanismSequenceAgent expects RAFMechanismSequenceAPI")
        answer = self._pipeline.invoke({"context": _SeqContext(api=task.api, metadata=task.metadata or {})})
        return TaskSubmission(answer=answer)

    def _estimate_means(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        context: _SeqContext = payload["context"]
        meta = context.metadata
        seqs = meta.get("sequences", [])
        snapshot_times = [float(t) for t in meta.get("snapshot_times", [])]
        # Use a single seed to estimate per-sequence means as an example
        seed = int(meta.get("example_seed", 6001))
        predictions: Dict[str, Dict[str, float]] = {}
        for seq in seqs:
            seq_id = seq.get("id") or "seq"
            sats = [int(t) for t in seq.get("saturations", [])]
            resp = context.api.generate_samples(
                saturation=sats,
                runs=1,
                snapshot_times=snapshot_times,
                extras=[],
                macro_params={"seed": seed},
            )
            final_means: Dict[str, float] = {}
            counts: Dict[str, int] = {}
            sequence = resp.get("sequence", [])
            last = sequence[-1] if sequence else {}
            for traj in last.get("trajectories", []):
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
