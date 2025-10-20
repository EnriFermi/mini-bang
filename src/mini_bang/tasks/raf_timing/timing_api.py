from __future__ import annotations

from typing import Any, Dict, List

from mini_bang.framework.task import AgentAPI


class RAFTimingAPI(AgentAPI):
    """Expose RAF timing training data to agents."""

    def __init__(self, *, description: str, dataset: dict[str, Any]):
        self._description = description
        self._dataset = dataset

    def instructions(self) -> str:
        return self._description

    def get_training_data(self) -> dict[str, Any]:
        """Return the pre-generated trajectories for each training macro seed."""
        return {
            "snapshot_times": list(self._dataset["snapshot_times"]),
            "saturation": self._dataset["saturation"],
            "simulators": {
                seed: {
                    "macro_seed": meta["macro_seed"],
                    "trajectories": [self._clone_traj(traj) for traj in meta["train"]["trajectories"]],
                    "first_hits": [dict(hit) for hit in meta["train"]["first_hits"]],
                    "species": list(meta.get("species", [])),
                }
                for seed, meta in self._dataset["simulators"].items()
            },
        }

    def get_spec(self) -> dict[str, Any]:
        return {
            "snapshot_times": list(self._dataset["snapshot_times"]),
            "saturation": self._dataset["saturation"],
            "macro_seeds": list(self._dataset["simulators"].keys()),
        }

    @staticmethod
    def _clone_traj(traj: Dict[str, List[int]]) -> Dict[str, List[int]]:
        return {species: list(counts) for species, counts in traj.items()}
