from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:  # pragma: no cover
    from mini_bang.framework.simulation import MicroSession


def raf_response_builder(
    *,
    simulator_id: str,
    saturation: int | Iterable[int],
    runs: int,
    macro_params: dict[str, Any],
    micro_params: dict[str, Any],
    sample_params: dict[str, Any],
    micro_session: "MicroSession" | list["MicroSession"],
    trajectories: list[Any],
    extras: list[str],
) -> dict[str, Any]:
    if isinstance(saturation, Iterable) and not isinstance(saturation, (str, bytes)):
        return _build_sequence_response(
            simulator_id=simulator_id,
            saturations=list(saturation),
            runs=runs,
            macro_params=macro_params,
            micro_params=micro_params,
            sample_params=sample_params,
            micro_sessions=list(micro_session) if isinstance(micro_session, list) else [micro_session],
            sequences=trajectories,
            extras=extras,
        )

    session = micro_session[0] if isinstance(micro_session, list) else micro_session
    response = {
        "simulator_id": simulator_id,
        "saturation": saturation,
        "runs": runs,
        "trajectories": trajectories,
        "metadata": session.metadata(),
        "macro_params": macro_params,
        "micro_params": micro_params,
        "sample_params": sample_params,
    }
    snapshot_times = sample_params.get("snapshot_times") if isinstance(sample_params, dict) else None
    if "first_hits" in extras:
        hits: list[dict[str, Any]] = []
        for traj in trajectories:
            hits.append(_first_hit_map(traj, snapshot_times))
        response["first_hits"] = hits
        if snapshot_times is not None:
            response.setdefault("snapshot_times", list(snapshot_times))
    if "is_raf" in extras:
        raw_micro = session.raw()
        if hasattr(raw_micro, "crn"):
            crn = raw_micro.crn
            is_raf_fn = getattr(crn, "is_raf", None)
            if callable(is_raf_fn):
                response["is_raf"] = bool(is_raf_fn())
    return response


def _build_sequence_response(
    *,
    simulator_id: str,
    saturations: list[int],
    runs: int,
    macro_params: dict[str, Any],
    micro_params: dict[str, Any],
    sample_params: dict[str, Any],
    micro_sessions: list[Any],
    sequences: list[Any],
    extras: list[str],
) -> dict[str, Any]:
    snapshot_times = sample_params.get("snapshot_times") if isinstance(sample_params, dict) else None
    entries: list[dict[str, Any]] = []
    for sat, session, samples in zip(saturations, micro_sessions, sequences):
        entry = {
            "saturation": sat,
            "trajectories": samples,
            "metadata": session.metadata(),
        }
        if "is_raf" in extras:
            raw = session.raw()
            if hasattr(raw, "crn"):
                func = getattr(raw.crn, "is_raf", None)
                if callable(func):
                    entry["is_raf"] = bool(func())
        if "first_hits" in extras:
            entry["first_hits"] = [
                _first_hit_map(traj, snapshot_times)
                for traj in samples
            ]
        entries.append(entry)

    response = {
        "simulator_id": simulator_id,
        "sequence": entries,
        "sequence_saturations": saturations,
        "runs": runs,
        "macro_params": macro_params,
        "micro_params": micro_params,
        "sample_params": sample_params,
    }
    if snapshot_times is not None:
        response.setdefault("snapshot_times", list(snapshot_times))
    return response


def _first_hit_map(traj: dict[str, list[int]], snapshot_times: list[float] | None) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for species, series in traj.items():
        first_idx = next((idx for idx, value in enumerate(series) if value > 0), None)
        if first_idx is None:
            result[species] = None
        elif snapshot_times and first_idx < len(snapshot_times):
            result[species] = snapshot_times[first_idx]
        elif snapshot_times:
            result[species] = snapshot_times[-1]
        else:
            result[species] = first_idx
    return result
