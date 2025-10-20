from __future__ import annotations

from typing import Any, Sequence

from mini_bang.framework.task import AgentAPI
from mini_bang.tasks.raf_common.api_client import RAFSimulationClient


class RAFSignatureAPI(AgentAPI):
    """Agent-facing API for the RAF Signature Recognition task."""

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
        """Return labelled training simulations grouped by saturation."""
        return {
            "saturations": list(self._dataset["saturations"]),
            "snapshot_times": list(self._dataset["snapshot_times"]),
            "simulators": {
                sat: [
                    {
                        "macro_seed": item["macro_seed"],
                        "is_raf": bool(item["is_raf"]),
                        "trajectories": [self._clone_traj(traj) for traj in item["trajectories"]],
                    }
                    for item in simulators
                ]
                for sat, simulators in self._dataset["simulators"].items()
            },
        }

    def simulate(
        self,
        *,
        saturation: int,
        runs: int,
        snapshot_times: Sequence[float] | None = None,
        extras: Sequence[str] | None = None,
        macro_params: dict[str, Any] | None = None,
        micro_params: dict[str, Any] | None = None,
        sample_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Request fresh trajectories from the simulator."""
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
