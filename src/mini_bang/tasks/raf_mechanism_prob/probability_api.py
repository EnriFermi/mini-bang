from __future__ import annotations

from typing import Any, Sequence

from mini_bang.framework.task import AgentAPI
from mini_bang.tasks.raf_common.api_client import RAFSimulationClient


class RAFMechanismProbabilityAPI(AgentAPI):
    """Agent interface for the RAF mechanism probability task."""

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
        """Return the labelled boolean outcomes for each saturation."""
        return {
            "saturations": list(self._dataset["saturations"]),
            "samples": {
                sat: [
                    {
                        "macro_seed": item["macro_seed"],
                        "is_raf": bool(item["is_raf"]),
                    }
                    for item in samples
                ]
                for sat, samples in self._dataset["samples"].items()
            },
        }

    def simulate(
        self,
        *,
        saturation: int,
        runs: int = 1,
        snapshot_times: Sequence[float] | None = None,
        extras: Sequence[str] | None = None,
        macro_params: dict[str, Any] | None = None,
        micro_params: dict[str, Any] | None = None,
        sample_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._client.generate_samples(
            saturation=saturation,
            runs=runs,
            snapshot_times=snapshot_times,
            extras=extras,
            macro_params=macro_params,
            micro_params=micro_params,
            sample_params=sample_params,
        )
