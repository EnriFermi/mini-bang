from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastmcp.mcp_config import MCPConfig as FastMCPCfg
from fastmcp.mcp_config import StdioMCPServer


@dataclass
class MCPConfig:
    """Minimal loader for canonical fastmcp MCP config in this package.

    Assumes a single config file at src/mini_bang/mcp/mcp.config.json and returns
    a FastMCP-compatible MCPConfig object for use with fastmcp.Client.
    """

    path: Path
    fastmcp: FastMCPCfg

    @classmethod
    def load(cls) -> "MCPConfig":
        path = Path(__file__).with_name("mcp.config.json")
        fast_cfg = FastMCPCfg.from_file(path)

        # Ensure subprocess can import this repo (dev convenience)
        try:
            for server in fast_cfg.mcpServers.values():  # type: ignore[attr-defined]
                if isinstance(server, StdioMCPServer):
                    env = dict(server.env or {})
                    if "PYTHONPATH" not in env:
                        src_dir = Path(__file__).resolve().parents[3]  # .../repo/src
                        env["PYTHONPATH"] = os.pathsep.join(
                            [str(src_dir), os.environ.get("PYTHONPATH", "")]
                        ).rstrip(os.pathsep)
                        server.env = env
        except Exception:
            pass

        return cls(path=path, fastmcp=fast_cfg)

    def to_dict(self) -> dict[str, Any]:
        return self.fastmcp.to_dict()

    def describe(self) -> dict[str, Any]:
        out = self.fastmcp.to_dict()
        out["_configPath"] = str(self.path)
        return out
