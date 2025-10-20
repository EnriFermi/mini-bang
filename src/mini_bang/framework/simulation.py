from __future__ import annotations

from typing import Any, Callable, Dict

from pydantic import BaseModel

from mini_bang.simulators.base.macro.simulator import MacroSimulatorBase
from mini_bang.simulators.base.micro.simulator import MicroSimulatorBase


class SimulationEngine:
    """
    Registry-backed helper that instantiates macro simulators and exposes them
    via lightweight session handles. Used exclusively inside the API server.
    """

    def __init__(self) -> None:
        self._registry: Dict[str, Callable[..., MacroSimulatorBase]] = {}

    def register_macro(self, name: str, factory: Callable[..., MacroSimulatorBase]) -> None:
        if name in self._registry:
            raise ValueError(f"Macro simulator '{name}' already registered")
        self._registry[name] = factory

    def has_macro(self, name: str) -> bool:
        return name in self._registry

    def spawn_macro(self, name: str, **kwargs: Any) -> "MacroSession":
        if name not in self._registry:
            raise KeyError(f"Macro simulator '{name}' is not registered")
        macro = self._registry[name](**kwargs)
        if not isinstance(macro, MacroSimulatorBase):
            raise TypeError("Factory must return a MacroSimulatorBase instance")
        return MacroSession(macro)


class MacroSession:
    __slots__ = ("_macro",)

    def __init__(self, macro: MacroSimulatorBase) -> None:
        self._macro = macro

    def describe_saturation(self) -> type[BaseModel]:
        return self._macro.get_saturation_description()

    def create_micro(self, saturation: Any, **kwargs: Any):
        micro = self._macro.get_micro_simulator(saturation, **kwargs)
        if isinstance(micro, MicroSimulatorBase):
            return MicroSession(micro)
        if isinstance(micro, (list, tuple)):
            sessions: list[MicroSession] = []
            for m in micro:
                if not isinstance(m, MicroSimulatorBase):
                    raise TypeError("Macro simulator returned unsupported micro simulator type")
                sessions.append(MicroSession(m))
            return sessions
        raise TypeError("Macro simulator returned unsupported micro simulator type")


class MicroSession:
    __slots__ = ("_micro",)

    def __init__(self, micro: MicroSimulatorBase) -> None:
        if not isinstance(micro, MicroSimulatorBase):
            raise TypeError("MicroSession expects a MicroSimulatorBase instance")
        self._micro = micro

    def sample(self, **kwargs: Any) -> Any:
        return self._micro.sample(**kwargs)

    def metadata(self) -> dict[str, Any]:
        if hasattr(self._micro, "crn"):
            species = getattr(self._micro.crn, "species", None)
            if species is not None:
                return {"species": list(species)}
        return {}

    def raw(self) -> MicroSimulatorBase:
        """Escape hatch for trusted validation code inside the API server."""
        return self._micro
