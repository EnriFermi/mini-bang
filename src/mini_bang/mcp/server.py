from __future__ import annotations

import argparse
import json
import logging
import os
import threading
from typing import Any, Dict, Iterable, Optional, Sequence

from fastmcp import FastMCP

from mini_bang.framework.simulation import SimulationEngine
from mini_bang.mcp.config import MCPConfig
from mini_bang.simulators.registry import get_simulator_entry

LOGGER = logging.getLogger("mini_bang.mcp")

mcp = FastMCP(
    name="mini-bang",
    instructions="""
        Unified simulation endpoint. Call get_simulation(simulator_id=..., saturation=..., runs=...)
        with optional snapshot_times/extras/macro_params/micro_params/sample_params.
    """,
)

_ENGINE = SimulationEngine()


def _ensure_macro(simulator_id: str):
    if _ENGINE.has_macro(simulator_id):
        return get_simulator_entry(simulator_id)
    entry = get_simulator_entry(simulator_id)
    _ENGINE.register_macro(simulator_id, entry.factory())
    return entry


def _invoke_simulation(
    *,
    simulator_id: str,
    saturation: Any,
    runs: int,
    extras: list[str],
    macro_params: dict[str, Any],
    micro_params: dict[str, Any],
    sample_params: dict[str, Any],
) -> dict[str, Any]:
    entry = _ensure_macro(simulator_id)
    macro = _ENGINE.spawn_macro(simulator_id, **macro_params)
    micro = macro.create_micro(saturation, **micro_params)

    saturation_is_sequence = isinstance(saturation, Iterable) and not isinstance(
        saturation, (str, bytes)
    )

    if saturation_is_sequence:
        micro_list = micro if isinstance(micro, list) else [micro]
        trajectories = [
            [session.sample(**sample_params) for _ in range(runs)]
            for session in micro_list
        ]
    else:
        session = micro[0] if isinstance(micro, list) else micro
        trajectories = [session.sample(**sample_params) for _ in range(runs)]

    response = entry.response_builder()(
        simulator_id=simulator_id,
        saturation=saturation,
        runs=runs,
        macro_params=macro_params,
        micro_params=micro_params,
        sample_params=sample_params,
        micro_session=micro,
        trajectories=trajectories,
        extras=extras,
    )
    return response


@mcp.tool(name="get_simulation")
def get_simulation(
    simulator_id: str,
    saturation: Any,
    runs: int = 1,
    snapshot_times: Optional[Sequence[float]] = None,
    max_raf: Optional[bool] = None,
    prune_catalysts: Optional[bool] = None,
    seed: Optional[int] = None,
    extras: Optional[Iterable[str]] = None,
    macro_params: Optional[Dict[str, Any]] = None,
    micro_params: Optional[Dict[str, Any]] = None,
    sample_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload_macro: Dict[str, Any] = dict(macro_params or {})
    payload_micro: Dict[str, Any] = dict(micro_params or {})
    payload_sample: Dict[str, Any] = dict(sample_params or {})
    LOGGER.info(f'Recieved payload simulator_id: {simulator_id}')
    if seed is not None:
        payload_macro.setdefault("seed", seed)
    if max_raf is not None:
        payload_micro.setdefault("max_raf", bool(max_raf))
    if prune_catalysts is not None:
        payload_micro.setdefault("prune_catalysts", bool(prune_catalysts))
    if snapshot_times is not None:
        payload_sample.setdefault("snapshot_times", list(snapshot_times))

    return _invoke_simulation(
        simulator_id=simulator_id,
        saturation=saturation,
        runs=runs,
        extras=list(extras or []),
        macro_params=payload_macro,
        micro_params=payload_micro,
        sample_params=payload_sample,
    )


"""
Note on HTTP runtime:
FastMCP provides a built-in HTTP/SSE transport. We rely on that instead of a
custom HTTP gateway. Use --transport http (alias for SSE in this version) to
serve over the network.
"""


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="mini-bang MCP simulation server")
    parser.add_argument(
        "--transport",
        default="stdio",
        choices=["stdio", "http", "sse"],
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--log-file",
        default=None,
        help="Path to write server logs (defaults to ~/.mini_bang/logs/mcp_server.log)",
    )
    args = parser.parse_args(argv)

    # Configure logging to file (avoid stdout to keep stdio protocol clean)
    log_file = args.log_file or os.getenv("MINI_BANG_MCP_LOG")
    if not log_file:
        # Default to a repo-local logs directory to avoid permission issues
        from pathlib import Path

        repo_root = Path(__file__).resolve().parents[3]
        base_dir = repo_root / "logs"
        base_dir.mkdir(parents=True, exist_ok=True)
        log_file = str(base_dir / "mcp_server.log")
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        filename=log_file,
        filemode="a",
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    LOGGER.info("Writing server logs to %s", log_file)
    config = MCPConfig.load()
    LOGGER.info("MCP config: %s", config.to_dict())

    transport = args.transport

    if transport == "http":
        try:
            mcp.run("http", host=args.host, port=args.port)
        except (TypeError, ValueError, AttributeError):
            LOGGER.info("HTTP transport not supported in this fastmcp version; falling back to SSE")
            mcp.run("sse", host=args.host, port=args.port)
    elif transport == "sse":
        mcp.run("sse", host=args.host, port=args.port)
    elif transport == "stdio":
        mcp.run("stdio")
    else:
        raise RuntimeError(f"Unsupported transport '{args.transport}'")


if __name__ == "__main__":
    main()
