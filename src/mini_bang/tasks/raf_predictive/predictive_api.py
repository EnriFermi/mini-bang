from __future__ import annotations

from typing import Any, Sequence

from mini_bang.framework.task import AgentAPI
from mini_bang.tasks.raf_common.api_client import RAFSimulationAPI


class RAFPredictiveAPI(AgentAPI):
    """API giving access to monotonic RAF sequences for predictive generalisation."""

    def __init__(
        self,
        *,
        description: str,
        dataset: dict[str, Any],
        client: RAFSimulationAPI,
        max_generate_calls: int | None = None,
    ) -> None:
        self._description = description
        self._dataset = dataset
        self._client = client
        self._max_calls = int(max_generate_calls) if max_generate_calls is not None else 50
        self._used_calls = 0

    def instructions(self) -> str:
        return self._description

    # No training data access: agents should use generate_samples

    def _consume(self) -> None:
        if self._used_calls >= self._max_calls:
            raise RuntimeError("generate_samples call limit exceeded for this task")
        self._used_calls += 1

    def generate_samples(
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
        self._consume()
        return self._client.generate_samples(
            saturation=saturation,
            runs=runs,
            snapshot_times=snapshot_times,
            extras=extras,
            macro_params=macro_params,
            micro_params=micro_params,
            sample_params=sample_params,
        )

    # simulate() removed: use generate_samples()
