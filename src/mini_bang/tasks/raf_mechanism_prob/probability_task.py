from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List

from mini_bang.framework.task import TaskEnvironment, TaskSubmission, ValidationResult
from mini_bang.tasks.base import RemoteSimulationTask
from mini_bang.tasks.raf_common.api_client import RAFSimulationAPI
from mini_bang.tasks.raf_mechanism_prob.probability_api import RAFMechanismProbabilityAPI


@dataclass
class _ValidationConfig:
    max_kl: float
    epsilon: float


class RAFLevel5MechanismProbabilityTask(RemoteSimulationTask):
    """Level 5 mechanism inference: estimate RAF probability as a function of T."""
    task_id = "raf/mechanism-prob-v1"
    config_package = __package__

    def __init__(self) -> None:
        super().__init__()
        cfg = self.config_copy()
        dataset_cfg = cfg.get("dataset", {})
        self._saturations: List[int] = [int(t) for t in dataset_cfg.get("saturations", [])]
        if not self._saturations:
            raise ValueError("Mechanism probability task requires saturations")
        self._train_seeds: List[int] = [int(s) for s in dataset_cfg.get("train_seeds", [])]
        self._test_seeds: List[int] = [int(s) for s in dataset_cfg.get("test_seeds", [])]
        if not self._train_seeds or not self._test_seeds:
            raise ValueError("Train/test seeds must be provided")
        self._snapshot_times: List[float] = [float(t) for t in dataset_cfg.get("snapshot_times", [])]

        val_cfg = cfg.get("validation", {})
        self._validation = _ValidationConfig(
            max_kl=float(val_cfg.get("max_kl", 0.6)),
            epsilon=float(val_cfg.get("epsilon", 1e-6)),
        )

        self._description = cfg.get("description", "")
        self._metadata_template: Dict[str, Any] = cfg.get("metadata", {})

        api_cfg = cfg.get("api", {})
        self._simulator_id = api_cfg.get("simulator_id", "raf")
        self._instructions = api_cfg.get("instructions") or self._description
        self._max_generate_calls = int(api_cfg.get("max_generate_calls", 50))

        self._dataset: dict[str, Any] | None = None
        self._test_data: Dict[str, List[bool]] | None = None

    def _build_remote_environment(self) -> TaskEnvironment:
        client = RAFSimulationAPI(simulator_id=self._simulator_id, instructions=self._instructions)
        test_data = self._prepare_test_data(client)
        self._dataset = None
        self._test_data = test_data

        api = RAFMechanismProbabilityAPI(description=self._instructions, dataset={}, client=client, max_generate_calls=self._max_generate_calls)
        metadata = dict(self._metadata_template)
        metadata.update(
            {
                "simulator_id": self._simulator_id,
                "saturations": list(self._saturations),
                "snapshot_times": list(self._snapshot_times),
                "test_seeds": [str(s) for s in self._test_seeds],
                "mcp": client.describe(),
            }
        )
        return TaskEnvironment(description=self._description, api=api, metadata=metadata)

    def _prepare_test_data(
        self, client: RAFSimulationAPI
    ) -> Dict[str, List[bool]]:
        """Collect RAF outcomes for test seeds only for validation."""
        test_truth: Dict[str, List[bool]] = {str(t): [] for t in self._saturations}
        for sat in self._saturations:
            sat_key = str(sat)
            for seed in self._test_seeds:
                payload = client.generate_samples(
                    saturation=sat,
                    runs=1,
                    snapshot_times=self._snapshot_times,
                    extras=["is_raf"],
                    macro_params={"seed": seed},
                )
                test_truth[sat_key].append(bool(payload.get("is_raf", False)))
        return test_truth

    def validate(self, submission: TaskSubmission) -> ValidationResult:
        if self._test_data is None:
            return ValidationResult(False, "Task dataset not initialised")
        answer = submission.answer
        if not isinstance(answer, dict):
            return ValidationResult(False, "Submission answer must be a mapping")
        probabilities = answer.get("probabilities")
        if not isinstance(probabilities, dict):
            return ValidationResult(False, "Submission must contain 'probabilities' mapping")

        kl_values: Dict[str, float] = {}
        for sat_key, truths in self._test_data.items():
            if sat_key not in probabilities:
                return ValidationResult(False, f"Missing probability for saturation {sat_key}")
            pred = float(probabilities[sat_key])
            pred = max(self._validation.epsilon, min(1.0 - self._validation.epsilon, pred))
            true_prob = sum(1 for val in truths if val) / len(truths) if truths else 0.0
            true_prob = max(self._validation.epsilon, min(1.0 - self._validation.epsilon, true_prob))
            kl = self._bern_kl(true_prob, pred)
            kl_values[sat_key] = kl

        mean_kl = sum(kl_values.values()) / len(kl_values)
        success = mean_kl <= self._validation.max_kl
        details = f"Mean Bernoulli KL {mean_kl:.3f} (threshold {self._validation.max_kl:.3f})"
        metrics = {
            "mean_kl": mean_kl,
            "per_saturation": kl_values,
        }
        return ValidationResult(success, details, metrics=metrics)

    def _bern_kl(self, p_true: float, p_pred: float) -> float:
        q_true = 1.0 - p_true
        q_pred = 1.0 - p_pred
        return p_true * math.log(p_true / p_pred) + q_true * math.log(q_true / q_pred)
