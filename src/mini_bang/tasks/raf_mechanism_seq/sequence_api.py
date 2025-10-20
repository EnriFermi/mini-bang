from __future__ import annotations

from typing import Any, Sequence

from mini_bang.framework.task import AgentAPI
from mini_bang.tasks.raf_common.api_client import RAFSimulationClient


class RAFMechanismSequenceAPI(AgentAPI):
    """Agent API for the RAF mechanism sequence task."""

    def __init__(
        self,
        *,
        description: str,
        dataset: dict[str, Any],
        client: RAFSimulationClient,
    ) -> None:
        self._description = description
        self._dataset = dataset
        self._client = client

    def instructions(self) -> str:
        return self._description

    def get_training_data(self) -> dict[str, Any]:
        """Provide the full step-by-step trajectories for each training sequence."""
        return {
            "sequences": {
                seq_id: {
                    "saturations": list(info["saturations"]),
                    "train": [
                        {
                            "macro_seed": item["macro_seed"],
                            "steps": [
                                {
                                    "saturation": step["saturation"],
                                    "trajectories": [self._clone_traj(traj) for traj in step["trajectories"]],
                                }
                                for step in item["steps"]
                            ],
                        }
                        for item in info["train"]
                    ],
                }
                for seq_id, info in self._dataset["sequences"].items()
            },
            "snapshot_times": list(self._dataset["snapshot_times"]),
        }

    def simulate(
        self,
        *,
        saturation: int | Sequence[int],
        runs: int,
        snapshot_times: Sequence[float] | None = None,
        extras: Sequence[str] | None = None,
        macro_params: dict[str, Any] | None = None,
        micro_params: dict[str, Any] | None = None,
        sample_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Proxy to the simulation server; supports passing a saturation list for sequences."""
        return self._client.generate_samples(
            saturation=saturation,
            runs=runs,
            snapshot_times=snapshot_times,
            extras=extras,
            macro_params=macro_params,
            micro_params=micro_params,
            sample_params=sample_params,
        )

    @staticmethod
    def _clone_traj(traj: dict[str, list[int]]) -> dict[str, list[int]]:
        return {species: list(counts) for species, counts in traj.items()}
