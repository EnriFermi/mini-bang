from __future__ import annotations

from dataclasses import dataclass, field
import importlib
import json
import pkgutil
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, Protocol

from mini_bang.simulators.base.macro.simulator import MacroSimulatorBase

if TYPE_CHECKING:  # pragma: no cover
    from mini_bang.framework.simulation import MicroSession


class ResponseBuilder(Protocol):
    def __call__(
        self,
        *,
        simulator_id: str,
        saturation: int,
        runs: int,
        macro_params: dict[str, Any],
        micro_params: dict[str, Any],
        sample_params: dict[str, Any],
        micro_session: "MicroSession",
        trajectories: list[Any],
        extras: list[str],
    ) -> dict[str, Any]:
        ...


def import_from_path(path: str) -> Callable[..., Any]:
    module_name, _, attr = path.partition(":")
    if not module_name or not attr:
        raise ValueError(f"Invalid import path '{path}'")
    module = importlib.import_module(module_name)
    return getattr(module, attr)


@dataclass(frozen=True)
class SimulatorSpec:
    simulator_id: str
    description: str
    factory_path: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    response_builder_path: str | None = None
    config: Dict[str, Any] = field(default_factory=dict)

    def factory(self) -> Callable[..., MacroSimulatorBase]:
        func = import_from_path(self.factory_path)

        def _builder(**overrides: Any) -> MacroSimulatorBase:
            params = dict(self.config.get("defaults", {}))
            for key, value in overrides.items():
                if value is not None:
                    params[key] = value
            return func(**params)

        return _builder

    def response_builder(self) -> ResponseBuilder:
        if not self.response_builder_path:
            return default_response_builder
        func = import_from_path(self.response_builder_path)
        return func

    def create_macro(self, **overrides: Any) -> MacroSimulatorBase:
        factory = self.factory()
        return factory(**overrides)


def default_response_builder(
    *,
    simulator_id: str,
    saturation: int,
    runs: int,
    macro_params: dict[str, Any],
    micro_params: dict[str, Any],
    sample_params: dict[str, Any],
    micro_session: "MicroSession",
    trajectories: list[Any],
    extras: list[str],
) -> dict[str, Any]:
    return {
        "simulator_id": simulator_id,
        "saturation": saturation,
        "runs": runs,
        "trajectories": trajectories,
        "metadata": micro_session.metadata(),
        "macro_params": macro_params,
        "micro_params": micro_params,
        "sample_params": sample_params,
    }


class SimulatorRegistry:
    def __init__(self) -> None:
        self._specs: Dict[str, SimulatorSpec] = {}
        self._load_builtin_specs()

    def _load_builtin_specs(self) -> None:
        from importlib import resources

        for package in discover_simulator_packages():
            try:
                cfg_resource = resources.files(package).joinpath("config.json")
            except FileNotFoundError:
                continue
            if not cfg_resource.is_file():
                continue
            with cfg_resource.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            spec = SimulatorSpec(
                simulator_id=data["id"],
                description=data.get("description", ""),
                factory_path=data["factory"],
                metadata=data.get("metadata", {}),
                response_builder_path=data.get("response_builder"),
                config={k: v for k, v in data.items() if k not in {"id", "description", "factory", "metadata", "response_builder"}},
            )
            self.register(spec)

    def register(self, spec: SimulatorSpec) -> None:
        if spec.simulator_id in self._specs:
            raise ValueError(f"Simulator '{spec.simulator_id}' already registered")
        self._specs[spec.simulator_id] = spec

    def get(self, simulator_id: str) -> SimulatorSpec:
        try:
            return self._specs[simulator_id]
        except KeyError as exc:
            raise KeyError(f"Unknown simulator '{simulator_id}'") from exc

    def all(self) -> Iterable[SimulatorSpec]:
        return tuple(self._specs.values())


def discover_simulator_packages() -> Iterable[str]:
    from importlib import import_module

    root_pkg = import_module("mini_bang.simulators")

    packages: list[str] = []
    for module_info in pkgutil.iter_modules(root_pkg.__path__, root_pkg.__name__ + "."):
        if module_info.ispkg:
            packages.append(module_info.name)
    return packages


_REGISTRY = SimulatorRegistry()


def get_simulator_entry(simulator_id: str) -> SimulatorSpec:
    return _REGISTRY.get(simulator_id)


def list_simulator_entries() -> Iterable[SimulatorSpec]:
    return _REGISTRY.all()


__all__ = [
    "SimulatorSpec",
    "SimulatorRegistry",
    "get_simulator_entry",
    "list_simulator_entries",
    "default_response_builder",
]
