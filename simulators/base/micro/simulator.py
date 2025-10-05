from abc import ABC, abstractmethod
from typing import Dict, Any


class MicroSimulatorBase(ABC):
    """Base class for microscopic simulators."""

    @abstractmethod
    def sample(self, **kwargs) -> str:
        """
        Sample the system state

        Args:
            **kwargs: Additional simulator-specific parameters
        """
        pass
