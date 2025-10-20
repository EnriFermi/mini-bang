from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from mini_bang.framework import TaskEnvironment, TaskSubmission


class AgentBase(ABC):
    """Convenience base class for agents."""

    @abstractmethod
    def solve(self, task: TaskEnvironment) -> TaskSubmission | Any:
        ...
