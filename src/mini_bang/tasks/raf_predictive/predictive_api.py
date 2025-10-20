from __future__ import annotations

from typing import Any, Sequence

from mini_bang.framework.task import AgentAPI
from mini_bang.tasks.raf_common.api_client import RAFSimulationClient


class RAFPredictiveAPI(AgentAPI):
    """API giving access to monotonic RAF sequences for predictive generalisation."""

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
        """Return monotonic sequences with RAF labels for each saturation."""
        return {
            "sequences": {
                seq_id: {
                    "saturations": info["saturations"],
                    "train": [
                        {
                            "macro_seed": item["macro_seed"],
                            "is_raf": dict(item["is_raf"]),
                        }
                        for item in info["train"]
                    ],
                }
                for seq_id, info in self._dataset["sequences"].items()
            }
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
        """Forward simulate(...) calls to the RAF HTTP server."""
        return self._client.generate_samples(
            saturation=saturation,
            runs=runs,
            snapshot_times=snapshot_times,
            extras=extras,
            macro_params=macro_params,
            micro_params=micro_params,
            sample_params=sample_params,
        )
