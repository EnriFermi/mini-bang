from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class MacroSimulatorBase(ABC):
    def __init__(self, complexity: float, **kwargs):
        if not 0.0 <= complexity <= 1.0:
            raise ValueError("Complexity must be between 0.0 and 1.0")
        self._complexity = complexity

    @property
    def complexity(self) -> float:
        return self._complexity

    @abstractmethod
    def get_micro_simulator(self, T: Any, **kwargs):
        """Create and return micro simulator instance based on parameter T."""
        raise NotImplementedError

    @abstractmethod
    def get_saturation_description(self) -> type[BaseModel]:
        """
        Return pydantic model describing parameter T with its constraints.
        Example:
            return create_model('ParameterT',
                value=(int, Field(ge=0, le=100)),
                description=(str, Field(default="Number of molecules")))
        """
        raise NotImplementedError
