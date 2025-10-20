from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

from mini_bang.framework import TaskEnvironment, TaskSubmission, ValidationResult
from mini_bang.tasks.loader import TaskLoader


class AgentProtocol(Protocol):
    """Contract for agent implementations."""

    def solve(self, task: TaskEnvironment) -> TaskSubmission | Any:
        ...


@dataclass
class AgentRunResult:
    task_id: str
    environment: TaskEnvironment
    submission: TaskSubmission
    validation: ValidationResult


class AgentConnector:
    """
    High-level entry point used by benchmark authors and agent developers.
    """

    def __init__(self, loader: TaskLoader | None = None) -> None:
        self._tasks = loader or TaskLoader()

    def list_tasks(self) -> list[str]:
        return [task.task_id for task in self._tasks.all()]

    def prepare(self, task_id: str) -> TaskEnvironment:
        task = self._tasks.get(task_id)
        return task.build()

    def run(self, agent: AgentProtocol | Callable[[TaskEnvironment], Any], task_id: str) -> AgentRunResult:
        task = self._tasks.get(task_id)
        environment = task.build()
        submission = self._invoke_agent(agent, environment)
        if not isinstance(submission, TaskSubmission):
            submission = TaskSubmission(answer=submission)
        validation = task.validate(submission)
        return AgentRunResult(
            task_id=task.task_id,
            environment=environment,
            submission=submission,
            validation=validation,
        )

    def _invoke_agent(
        self,
        agent: AgentProtocol | Callable[[TaskEnvironment], Any],
        environment: TaskEnvironment,
    ) -> TaskSubmission | Any:
        if hasattr(agent, "solve") and callable(getattr(agent, "solve")):
            return agent.solve(environment)
        if callable(agent):
            return agent(environment)
        raise TypeError("Agent must be callable or provide a 'solve' method")
