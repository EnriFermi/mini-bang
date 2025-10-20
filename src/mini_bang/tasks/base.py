from __future__ import annotations

import copy
from typing import Any

from mini_bang.framework.task import SimulationTask, TaskEnvironment
from mini_bang.runtime import ensure_server_running
from mini_bang.tasks.config_loader import load_task_config


class ConfiguredSimulationTask(SimulationTask):
    """
    Base class for tasks that load their configuration from a data file.
    Subclasses must set `config_package`.
    """

    config_package: str
    config_filename: str = "config.json"

    def __init__(self) -> None:
        if not getattr(self, "config_package", None):
            raise ValueError("ConfiguredSimulationTask subclasses must set config_package")
        self._config: dict[str, Any] = load_task_config(self.config_package, self.config_filename)

    def config_copy(self) -> dict[str, Any]:
        """Return a deep copy of the task configuration."""
        return copy.deepcopy(self._config)

    def config_value(self, key: str, default: Any = None) -> Any:
        """Return a deep copy of a config value."""
        return copy.deepcopy(self._config.get(key, default))

    def build(self) -> TaskEnvironment:
        return self._build_environment()

    def _build_environment(self) -> TaskEnvironment:
        raise NotImplementedError


class RemoteSimulationTask(ConfiguredSimulationTask):
    """
    Task base for simulations that require the shared HTTP server.
    """

    def build(self) -> TaskEnvironment:
        server_url = ensure_server_running()
        return self._build_remote_environment(server_url)

    def _build_environment(self) -> TaskEnvironment:  # pragma: no cover - not used
        raise NotImplementedError("RemoteSimulationTask uses _build_remote_environment instead")

    def _build_remote_environment(self, server_url: str) -> TaskEnvironment:
        raise NotImplementedError
