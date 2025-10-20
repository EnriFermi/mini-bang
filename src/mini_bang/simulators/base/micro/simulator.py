from abc import ABC, abstractmethod


class MicroSimulatorBase(ABC):
    """Base class for microscopic simulators."""

    @abstractmethod
    def sample(self, **kwargs):
        """
        Sample the system state

        Args:
            **kwargs: Additional simulator-specific parameters
        """
        raise NotImplementedError
