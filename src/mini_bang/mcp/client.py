from __future__ import annotations

import asyncio
import json
from typing import Any, Optional

from fastmcp import Client
import mcp.types as mcp_types

from .config import MCPConfig


class MCPClient:
    """Thin synchronous wrapper around fastmcp.Client.

    Uses fastmcp's transport inference: stdio (auto-spawn), HTTP/SSE, or in-memory.
    The transport and parameters are loaded from an MCP config JSON file.
    """

    def __init__(self, config: Optional[MCPConfig] = None):
        self._config = config or MCPConfig.load()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[object] = None
        self._client: Optional[Client] = None
        self._started: bool = False
        self._lock = None  # lazy import threading to keep deps low

    def describe(self) -> dict[str, Any]:
        return self._config.describe()

    def call(self, method: str, params: dict[str, Any]) -> Any:
        self._ensure_started()
        return self._call_tool_sync(method, params)

    # Convenience wrapper for the core benchmark method.
    def call_get_simulation(self, simulator_id: str, params: dict[str, Any]) -> Any:
        payload = dict(params)
        payload["simulator_id"] = simulator_id
        return self.call("get_simulation", payload)

    def _call_tool_sync(self, name: str, arguments: dict[str, Any]) -> Any:
        # Submit coroutine to background event loop and wait
        coro = self._client_call_tool(name, arguments)
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)  # type: ignore[arg-type]
        return fut.result()

    async def _client_call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        assert self._client is not None
        result = await self._client.call_tool(name, arguments)
        is_error = getattr(result, "is_error", False)
        if is_error:
            text = None
            content = getattr(result, "content", None)
            if content:
                first = content[0]
                if hasattr(first, "text"):
                    text = first.text
            raise RuntimeError(text or "MCP tool error")

        structured = getattr(result, "structured_content", None)
        if structured is not None:
            return structured

        content = getattr(result, "content", None)
        if content:
            first = content[0]
            if hasattr(first, "text") and isinstance(first.text, str):
                try:
                    return json.loads(first.text)
                except json.JSONDecodeError:
                    return first.text
        return None

    def _ensure_started(self) -> None:
        if self._started:
            return
        # lazy import to avoid global threading dependency at import time
        import threading

        if self._lock is None:
            self._lock = threading.Lock()
        with self._lock:
            if self._started:
                return
            loop = asyncio.new_event_loop()
            self._loop = loop

            def _run_loop() -> None:
                asyncio.set_event_loop(loop)
                loop.run_forever()

            thread = threading.Thread(target=_run_loop, name="MCPClientLoop", daemon=True)
            thread.start()
            self._thread = thread

            # Create and enter client context in the loop
            self._client = Client(self._config.fastmcp)
            # Call __aenter__ to connect
            enter_fut = asyncio.run_coroutine_threadsafe(self._client.__aenter__(), loop)
            enter_fut.result()
            self._started = True

            # Ensure clean shutdown at exit
            try:
                import atexit

                atexit.register(self.close)
            except Exception:
                pass

    def close(self) -> None:
        if not self._started:
            return
        assert self._loop is not None
        try:
            if self._client is not None:
                fut = asyncio.run_coroutine_threadsafe(
                    self._client.__aexit__(None, None, None), self._loop
                )
                fut.result()
        finally:
            # stop loop and join thread
            self._loop.call_soon_threadsafe(self._loop.stop)
            if self._thread is not None:
                try:
                    # type: ignore[attr-defined]
                    self._thread.join(timeout=1.0)  # type: ignore[call-arg]
                except Exception:
                    pass
            self._started = False


__all__ = ["MCPClient", "MCPConfig"]
