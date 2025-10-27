from __future__ import annotations

from typing import Any, Dict, Sequence

from mini_bang.framework.task import AgentAPI
from mini_bang.tasks.raf_common.api_client import RAFSimulationAPI


class RAFTimingAPI(AgentAPI):
    """Expose RAF timing training data to agents."""

    def __init__(self, *, description: str, client: RAFSimulationAPI, max_generate_calls: int | None = None):
        self._description = description
        self._client = client
        self._max_calls = int(max_generate_calls) if max_generate_calls is not None else 50
        self._used_calls = 0

    def instructions(self) -> str:
        return self._description

    # Expose remaining call budget so agents can plan sampling.
    def remaining_calls(self) -> int:
        return max(0, self._max_calls - self._used_calls)

    def _consume(self) -> None:
        if self._used_calls >= self._max_calls:
            raise RuntimeError("generate_samples call limit exceeded for this task")
        self._used_calls += 1

    # Expose on-demand simulation with a per-task call limit
    def generate_samples(
        self,
        saturation: int | Sequence[int],
        runs: int,
        snapshot_times: Sequence[float] | None = None,
        *,
        max_raf: bool = False,
        prune_catalysts: bool = False,
        seed: int | None = None,
        extras: Sequence[str] | None = None,
        macro_params: dict[str, Any] | None = None,
        micro_params: dict[str, Any] | None = None,
        sample_params: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
        self._consume()
        return self._client.generate_samples(
            saturation=saturation,
            runs=runs,
            snapshot_times=snapshot_times,
            max_raf=max_raf,
            prune_catalysts=prune_catalysts,
            seed=seed,
            extras=extras,
            macro_params=macro_params,
            micro_params=micro_params,
            sample_params=sample_params,
        )
