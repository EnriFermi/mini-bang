from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


class AgentAPI(Protocol):
    """Minimal protocol for agent-facing APIs."""

    def instructions(self) -> str:
        ...


@dataclass
class TaskEnvironment:
    """Bundle handed to an agent: natural language instructions plus API handle."""

    description: str
    api: AgentAPI
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskSubmission:
    """Structured payload returned by an agent."""

    answer: Any
    artifacts: dict[str, Any] | None = None


@dataclass
class ValidationResult:
    success: bool
    details: str | None = None
    metrics: dict[str, Any] | None = None


class SimulationTask:
    """Base class for benchmark tasks."""

    task_id: str

    def build(self) -> TaskEnvironment:
        raise NotImplementedError

    def validate(self, submission: TaskSubmission) -> ValidationResult:
        raise NotImplementedError
