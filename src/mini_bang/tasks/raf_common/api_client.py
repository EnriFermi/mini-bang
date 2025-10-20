from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Sequence

from mini_bang.framework.task import AgentAPI

_DEFAULT_INSTRUCTIONS = (
    "Use generate_samples(T, runs, snapshot_times=None, *, max_raf=False, "
    "prune_catalysts=False, seed=None) to obtain trajectories for RAF networks. "
    "T must exceed the macro seed size M0."
)


class RAFSimulationClient(AgentAPI):
    """HTTP client for RAF simulators exposed by the simulation server."""

    def __init__(self, base_url: str, simulator_id: str, instructions: str | None = None):
        self._base_url = base_url.rstrip("/")
        self._simulator_id = simulator_id
        self._instructions = instructions or _DEFAULT_INSTRUCTIONS

    def instructions(self) -> str:
        return self._instructions

    def generate_samples(
        self,
        saturation: int,
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
            macro_payload["seed"] = seed
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

        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self._base_url}/simulate/{self._simulator_id}/generate",
            data=data,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=3600) as response:
                if response.status != 200:
                    raise RuntimeError(f"Server returned status {response.status}")
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Server error {exc.code}: {body}") from exc
