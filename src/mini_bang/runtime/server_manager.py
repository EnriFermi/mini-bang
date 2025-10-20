from __future__ import annotations

import logging
import socket
import threading
import time
import traceback
import urllib.request
from queue import SimpleQueue
from typing import Optional

_SERVER_THREAD: Optional[threading.Thread] = None
_SERVER_URL: Optional[str] = None
_SERVER_EXCEPTION_QUEUE: Optional[SimpleQueue[str | None]] = None


def _probe_port(host: str, port: int) -> bool:
    try:
        req = urllib.request.Request(f"http://{host}:{port}/health", method="GET")
        with urllib.request.urlopen(req, timeout=1) as resp:
            return resp.status == 200
    except Exception:
        return False


def ensure_server_running(host: str = "127.0.0.1") -> str:
    global _SERVER_THREAD, _SERVER_URL, _SERVER_EXCEPTION_QUEUE

    if _SERVER_THREAD and _SERVER_THREAD.is_alive() and _SERVER_URL:
        return _SERVER_URL

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        _, port = sock.getsockname()

    _SERVER_EXCEPTION_QUEUE = SimpleQueue()

    thread = threading.Thread(
        target=_server_worker,
        kwargs={
            "host": host,
            "port": port,
            "exc_queue": _SERVER_EXCEPTION_QUEUE,
        },
        name="mini-bang-simulation-server",
        daemon=True,
    )
    _SERVER_THREAD = thread
    thread.start()

    deadline = time.time() + 5
    while time.time() < deadline:
        if _SERVER_EXCEPTION_QUEUE and not _SERVER_EXCEPTION_QUEUE.empty():
            payload = _SERVER_EXCEPTION_QUEUE.get()
            if payload:
                raise RuntimeError(f"Simulation API server failed to start:\n{payload}")
        if not thread.is_alive():
            raise RuntimeError("Simulation API server terminated unexpectedly")
        if _probe_port(host, port):
            _SERVER_URL = f"http://{host}:{port}"
            _SERVER_EXCEPTION_QUEUE = None
            return _SERVER_URL
        time.sleep(0.1)

    _SERVER_EXCEPTION_QUEUE = None
    raise TimeoutError("Simulation API server did not become ready in time")


def _server_worker(host: str, port: int, exc_queue: SimpleQueue[str | None]) -> None:
    from mini_bang.api.server import run_server  # local import to avoid circulars

    try:
        run_server(host=host, port=port, log_level=logging.INFO)
    except Exception:  # pragma: no cover
        exc_queue.put(traceback.format_exc())
    else:
        exc_queue.put(None)
