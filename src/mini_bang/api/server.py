from __future__ import annotations

import json
import logging
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

from mini_bang.framework.simulation import SimulationEngine
from mini_bang.simulators.registry import get_simulator_entry, list_simulator_entries


LOGGER = logging.getLogger("mini_bang.api")


class _SimulationRequestHandler(BaseHTTPRequestHandler):
    engine = SimulationEngine()

    @classmethod
    def _ensure_macro(cls, simulator_id: str):
        entry = get_simulator_entry(simulator_id)
        if cls.engine.has_macro(simulator_id):
            return entry

        factory = entry.factory()
        cls.engine.register_macro(simulator_id, factory)
        return entry

    def log_message(self, fmt: str, *args: Any) -> None:
        LOGGER.debug("%s - %s", self.address_string(), fmt % args)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self._write_json({"status": "ok"})
            return
        if parsed.path == "/simulators":
            simulators = [entry.summary() for entry in list_simulator_entries()]
            self._write_json({"simulators": simulators})
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Unknown endpoint")

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        segments = [segment for segment in parsed.path.split("/") if segment]
        if len(segments) == 3 and segments[0] == "simulate" and segments[2] == "generate":
            simulator_id = segments[1]
            self._handle_generate(simulator_id)
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Unknown endpoint")

    def _handle_generate(self, simulator_id: str) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            self.send_error(HTTPStatus.BAD_REQUEST, "Missing request body")
            return

        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            self.send_error(HTTPStatus.BAD_REQUEST, "Invalid JSON payload")
            return

        required_fields = {"saturation", "runs"}
        if not required_fields.issubset(payload):
            self.send_error(
                HTTPStatus.BAD_REQUEST,
                f"Missing fields: {sorted(required_fields - payload.keys())}",
            )
            return

        raw_saturation = payload["saturation"]
        if isinstance(raw_saturation, (list, tuple)):
            saturation_list = [int(x) for x in raw_saturation]
            if not saturation_list:
                self.send_error(HTTPStatus.BAD_REQUEST, "Saturation list must not be empty")
                return
            saturation = saturation_list
        else:
            saturation = int(raw_saturation)
        runs = int(payload["runs"])

        if runs <= 0:
            self.send_error(HTTPStatus.BAD_REQUEST, "runs must be positive")
            return

        macro_params = payload.get("macro_params") or {}
        micro_params = payload.get("micro_params") or {}
        sample_params = payload.get("sample_params") or {}
        extras = payload.get("extras") or []

        if (
            not isinstance(macro_params, dict)
            or not isinstance(micro_params, dict)
            or not isinstance(sample_params, dict)
            or not isinstance(extras, (list, tuple))
        ):
            self.send_error(HTTPStatus.BAD_REQUEST, "macro_params, micro_params, sample_params must be objects and extras must be a list")
            return

        try:
            entry = self._ensure_macro(simulator_id)
        except KeyError as exc:
            self.send_error(HTTPStatus.NOT_FOUND, str(exc))
            return

        macro = self.engine.spawn_macro(simulator_id, **macro_params)
        micro_handle = macro.create_micro(saturation, **micro_params)

        sequence_mode = isinstance(saturation, list)
        sample_kwargs = dict(sample_params)

        if sequence_mode:
            if not isinstance(micro_handle, list):
                micro_list = [micro_handle]
            else:
                micro_list = micro_handle
            if len(micro_list) != len(saturation):
                raise RuntimeError("Mismatch between requested saturations and returned micro simulators")
            trajectories = [
                [session.sample(**sample_kwargs) for _ in range(runs)]
                for session in micro_list
            ]
        else:
            if isinstance(micro_handle, list):
                if not micro_handle:
                    raise RuntimeError("Macro returned empty micro simulator list")
                micro_handle = micro_handle[0]
            trajectories = [micro_handle.sample(**sample_kwargs) for _ in range(runs)]

        try:
            response_builder = entry.response_builder()
            response = response_builder(
                simulator_id=simulator_id,
                saturation=saturation,
                runs=runs,
                macro_params=dict(macro_params),
                micro_params=dict(micro_params),
                sample_params=dict(sample_params),
                micro_session=micro_handle,
                trajectories=trajectories,
                extras=list(extras),
            )
            self._write_json(response)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Simulation error for %s", simulator_id)
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def _write_json(self, payload: dict[str, Any], *, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run_server(host: str = "127.0.0.1", port: int = 8765, log_level: int = logging.INFO) -> None:
    logging.basicConfig(level=log_level)
    server = ThreadingHTTPServer((host, port), _SimulationRequestHandler)
    LOGGER.info("Simulation API server listening on http://%s:%d", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def _parse_args() -> tuple[str, int, int]:
    import argparse

    parser = argparse.ArgumentParser(description="mini-bang simulation API server")
    parser.add_argument("--host", default="127.0.0.1", help="Interface to bind (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind (default 8765)")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Logging verbosity",
    )
    args = parser.parse_args()
    level = getattr(logging, args.log_level.upper(), logging.INFO)
    return args.host, args.port, level


if __name__ == "__main__":
    host, port, level = _parse_args()
    run_server(host=host, port=port, log_level=level)
