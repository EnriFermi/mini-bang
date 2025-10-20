from __future__ import annotations

from typing import Dict, Iterable

from mini_bang.framework.task import SimulationTask
from mini_bang.tasks.raf_timing.timing_task import RAFLevel2TimingTask
from mini_bang.tasks.raf_signature.signature_task import RAFLevel4SignatureTask
from mini_bang.tasks.raf_mechanism_prob.probability_task import RAFLevel5MechanismProbabilityTask
from mini_bang.tasks.raf_mechanism_seq.sequence_task import RAFLevel5MechanismSequenceTask
from mini_bang.tasks.raf_predictive.predictive_task import RAFLevel6PredictiveTask


class TaskLoader:
    """Lightweight registry for benchmark tasks."""

    def __init__(self) -> None:
        self._tasks: Dict[str, SimulationTask] = {}
        self._auto_register()

    def _auto_register(self) -> None:
        self.register(RAFLevel2TimingTask())
        self.register(RAFLevel4SignatureTask())
        self.register(RAFLevel5MechanismProbabilityTask())
        self.register(RAFLevel5MechanismSequenceTask())
        self.register(RAFLevel6PredictiveTask())

    def register(self, task: SimulationTask) -> None:
        if task.task_id in self._tasks:
            raise ValueError(f"Task '{task.task_id}' already registered")
        self._tasks[task.task_id] = task

    def get(self, task_id: str) -> SimulationTask:
        if task_id not in self._tasks:
            raise KeyError(f"Unknown task '{task_id}'")
        return self._tasks[task_id]

    def all(self) -> Iterable[SimulationTask]:
        return self._tasks.values()
