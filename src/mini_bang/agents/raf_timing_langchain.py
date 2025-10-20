from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable

from langchain.schema.runnable import RunnableLambda

from mini_bang.framework import TaskEnvironment, TaskSubmission
from mini_bang.tasks.raf_timing.timing_api import RAFTimingAPI
from mini_bang.agents.base import AgentBase


@dataclass
class _TimingContext:
    api: RAFTimingAPI


class LangChainRAFTimingAgent(AgentBase):
    """Estimate per-species first-hit distributions using LangChain runnables."""

    def __init__(self, smoothing: float = 1e-6) -> None:
        self._epsilon = smoothing
        self._pipeline = (
            RunnableLambda(self._fetch_data)
            | RunnableLambda(self._compute_distributions)
        )

    def solve(self, task: TaskEnvironment) -> TaskSubmission:
        if not isinstance(task.api, RAFTimingAPI):
            raise TypeError("LangChainRAFTimingAgent expects a RAFTimingAPI")
        context = _TimingContext(api=task.api)
        answer = self._pipeline.invoke({"context": context})
        return TaskSubmission(answer=answer)

    def _fetch_data(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        context: _TimingContext = payload["context"]
        training = context.api.get_training_data()
        spec = context.api.get_spec()
        snapshot_times = [float(t) for t in training.get("snapshot_times", [])]
        return {
            "snapshot_times": snapshot_times,
            "simulators": training.get("simulators", {}),
            "macro_seeds": spec.get("macro_seeds", list(training.get("simulators", {}).keys())),
        }

    def _compute_distributions(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        snapshot_times = payload["snapshot_times"]
        categories = [str(t) for t in snapshot_times] + ["none"]
        simulators: Dict[str, Any] = payload["simulators"]
        result: Dict[str, Any] = {}

        for seed, meta in simulators.items():
            train_block = meta.get("train", {})
            trajectories = train_block.get("trajectories", [])
            first_hits = train_block.get("first_hits") or self._derive_first_hits(trajectories, snapshot_times)
            metadata_species = meta.get("species", [])
            species = self._collect_species(trajectories, first_hits, metadata_species)
            seed_dist: Dict[str, Dict[str, float]] = {}

            for specie in species:
                counts = {cat: 0.0 for cat in categories}
                total = 0
                for hit in first_hits:
                    category = self._categorise_hit(hit, specie, categories)
                    counts[category] += 1.0
                    total += 1
                if total == 0:
                    uniform = 1.0 / len(categories)
                    probs = {cat: uniform for cat in categories}
                else:
                    probs = {cat: (counts[cat] / total) for cat in categories}
                # Apply light smoothing to avoid exact zeros
                probs = {
                    cat: (value + self._epsilon) for cat, value in probs.items()
                }
                normaliser = sum(probs.values())
                if normaliser <= 0:
                    normaliser = float(len(categories)) * self._epsilon
                probs = {cat: value / normaliser for cat, value in probs.items()}
                seed_dist[specie] = probs
            result[seed] = seed_dist

        return {"distributions": result}

    def _collect_species(
        self,
        trajectories: Iterable[Dict[str, Any]],
        first_hits: Iterable[Dict[str, Any]],
        metadata_species: Iterable[Any],
    ) -> Iterable[str]:
        seen: set[str] = set()
        for traj in trajectories:
            seen.update(str(k) for k in traj.keys())
        for hit in first_hits:
            seen.update(str(k) for k in hit.keys())
        for value in metadata_species or []:
            seen.add(str(value))

        def _key(name: str) -> tuple[int, str]:
            try:
                return (int(name), name)
            except ValueError:
                return (0, name)

        return sorted(seen, key=_key) if seen else []

    def _derive_first_hits(self, trajectories: Iterable[Dict[str, Any]], snapshot_times: Iterable[float]) -> list[dict[str, Any]]:
        hits: list[dict[str, Any]] = []
        timeline = list(snapshot_times)
        for traj in trajectories:
            record: dict[str, Any] = {}
            for species, series in traj.items():
                first_idx = next((idx for idx, value in enumerate(series) if value > 0), None)
                if first_idx is None:
                    record[species] = None
                elif timeline and first_idx < len(timeline):
                    record[species] = timeline[first_idx]
                elif timeline:
                    record[species] = timeline[-1]
                else:
                    record[species] = first_idx
            hits.append(record)
        return hits

    def _categorise_hit(self, hit: Dict[str, Any], species: str, categories: Iterable[str]) -> str:
        value = hit.get(species)
        if value is None:
            return "none"
        category = str(value)
        return category if category in categories else "none"
