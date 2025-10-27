from __future__ import annotations

from typing import Any, Dict, Sequence

from mini_bang.framework.task import AgentAPI
from mini_bang.mcp.client import MCPClient

_DEFAULT_INSTRUCTIONS = (
    "Use generate_samples(T, runs, snapshot_times=None, *, max_raf=False, "
    "prune_catalysts=False, seed=None) to obtain trajectories for RAF networks. "
    "T must exceed the macro seed size M0."
)


class RAFSimulationAPI(AgentAPI):
    """Simulation API backed by the MCP server."""

    def __init__(
        self,
        *,
        simulator_id: str,
        instructions: str | None = None,
        client: MCPClient | None = None,
    ) -> None:
        self._simulator_id = simulator_id
        self._instructions = instructions or _DEFAULT_INSTRUCTIONS
        self._client = client or MCPClient()

    def instructions(self) -> str:
        return self._instructions

    def describe(self) -> Dict[str, Any]:
        return self._client.describe()

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
        payload: dict[str, Any] = {
            "saturation": saturation,
            "runs": runs,
        }
        macro_payload: dict[str, Any] = dict(macro_params or {})
        micro_payload: dict[str, Any] = dict(micro_params or {})
        sample_payload: dict[str, Any] = dict(sample_params or {})

        if seed is not None:
            macro_payload.setdefault("seed", seed)
        if max_raf is not None:
            micro_payload.setdefault("max_raf", bool(max_raf))
        if prune_catalysts is not None:
            micro_payload.setdefault("prune_catalysts", bool(prune_catalysts))
        if snapshot_times is not None:
            sample_payload.setdefault("snapshot_times", list(snapshot_times))

        if macro_payload:
            payload["macro_params"] = macro_payload
        if micro_payload:
            payload["micro_params"] = micro_payload
        if sample_payload:
            payload["sample_params"] = sample_payload
        if extras:
            payload["extras"] = list(extras)

        return self._client.call_get_simulation(self._simulator_id, payload)
