from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List

from mini_bang.framework.task import TaskEnvironment, TaskSubmission, ValidationResult
from mini_bang.tasks.base import RemoteSimulationTask
from mini_bang.tasks.raf_common.api_client import RAFSimulationClient
from mini_bang.tasks.raf_predictive.predictive_api import RAFPredictiveAPI


@dataclass
class _ValidationConfig:
    epsilon: float
    max_cross_entropy: float


class RAFLevel6PredictiveTask(RemoteSimulationTask):
    """Level 6 predictive generalisation: forecast RAF emergence probabilities."""
    task_id = "raf/predictive-v1"
    config_package = __package__

    def __init__(self) -> None:
        super().__init__()
        cfg = self.config_copy()
        dataset_cfg = cfg.get("dataset", {})
        self._sequences: List[dict[str, Any]] = dataset_cfg.get("sequences", [])
        if not self._sequences:
            raise ValueError("Predictive task requires sequence definitions")
        self._train_seeds: List[int] = [int(s) for s in dataset_cfg.get("train_seeds", [])]
        self._test_seeds: List[int] = [int(s) for s in dataset_cfg.get("test_seeds", [])]
        if not self._train_seeds or not self._test_seeds:
            raise ValueError("Train/test seeds must be provided")

        val_cfg = cfg.get("validation", {})
        self._validation = _ValidationConfig(
            epsilon=float(val_cfg.get("epsilon", 1e-6)),
            max_cross_entropy=float(val_cfg.get("max_cross_entropy", 0.65)),
        )

        self._description = cfg.get("description", "")
        self._metadata_template: Dict[str, Any] = cfg.get("metadata", {})
        api_cfg = cfg.get("api", {})
        self._simulator_id = api_cfg.get("simulator_id", "raf")
        self._instructions = api_cfg.get("instructions") or self._description

        self._dataset: dict[str, Any] | None = None
        self._test_data: Dict[str, Dict[str, Any]] | None = None

    def _build_remote_environment(self, server_url: str) -> TaskEnvironment:
        client = RAFSimulationClient(server_url, self._simulator_id, instructions=self._instructions)
        dataset, test_data = self._prepare_dataset(client)
        self._dataset = dataset
        self._test_data = test_data

        api = RAFPredictiveAPI(description=self._instructions, dataset=dataset, client=client)
        metadata = dict(self._metadata_template)
        metadata.update(
            {
                "api_base_url": server_url,
                "simulator_id": self._simulator_id,
                "sequences": self._sequences,
                "test_seeds": [str(s) for s in self._test_seeds],
            }
        )
        return TaskEnvironment(description=self._description, api=api, metadata=metadata)

    def _prepare_dataset(
        self, client: RAFSimulationClient
    ) -> tuple[dict[str, Any], Dict[str, Dict[str, Any]]]:
        """Build monotonic sequences with shared macro runs for training and evaluation."""
        dataset: dict[str, Any] = {"sequences": {}}
        test_truth: Dict[str, Dict[str, Any]] = {}

        for seq_cfg in self._sequences:
            seq_id = seq_cfg["id"]
            saturations = [int(t) for t in seq_cfg["saturations"]]
            dataset["sequences"][seq_id] = {
                "saturations": list(saturations),
                "train": [],
            }
            test_truth[seq_id] = {
                "saturations": list(saturations),
                "samples": [],
            }

            for seed in self._train_seeds:
                response = client.generate_samples(
                    saturation=saturations,
                    runs=1,
                    snapshot_times=None,
                    extras=["is_raf"],
                    macro_params={"seed": seed},
                )
                record = {"macro_seed": seed, "is_raf": {}}
                for entry in response.get("sequence", []):
                    record["is_raf"][str(entry.get("saturation"))] = bool(entry.get("is_raf", False))
                dataset["sequences"][seq_id]["train"].append(record)

            for seed in self._test_seeds:
                response = client.generate_samples(
                    saturation=saturations,
                    runs=1,
                    snapshot_times=None,
                    extras=["is_raf"],
                    macro_params={"seed": seed},
                )
                result = {"macro_seed": seed, "is_raf": {}}
                for entry in response.get("sequence", []):
                    result["is_raf"][str(entry.get("saturation"))] = bool(entry.get("is_raf", False))
                test_truth[seq_id]["samples"].append(result)
        return dataset, test_truth

    def validate(self, submission: TaskSubmission) -> ValidationResult:
        if self._test_data is None:
            return ValidationResult(False, "Task dataset not initialised")

        answer = submission.answer
        if not isinstance(answer, dict):
            return ValidationResult(False, "Submission answer must be a mapping")
        probabilities = answer.get("probabilities")
        if not isinstance(probabilities, dict):
            return ValidationResult(False, "Submission must contain 'probabilities' mapping")

        per_seq_entropy: Dict[str, float] = {}
        for seq_id, info in self._test_data.items():
            samples = info["samples"]
            saturations = info["saturations"]
            if seq_id not in probabilities:
                return ValidationResult(False, f"Missing probabilities for sequence {seq_id}")
            seq_pred = probabilities[seq_id]
            seq_entropy = 0.0
            count = 0
            for sat in saturations:
                sat_key = str(sat)
                if sat_key not in seq_pred:
                    return ValidationResult(False, f"Missing probability for saturation {sat_key} in sequence {seq_id}")
                pred = float(seq_pred[sat_key])
                pred = max(self._validation.epsilon, min(1.0 - self._validation.epsilon, pred))
                truth_values = [sample["is_raf"][sat_key] for sample in samples]
                true_prob = sum(1 for v in truth_values if v) / len(truth_values)
                true_prob = max(self._validation.epsilon, min(1.0 - self._validation.epsilon, true_prob))
                entropy = -(
                    true_prob * math.log(pred)
                    + (1.0 - true_prob) * math.log(1.0 - pred)
                )
                seq_entropy += entropy
                count += 1
            per_seq_entropy[seq_id] = seq_entropy / count if count else 0.0

        mean_entropy = sum(per_seq_entropy.values()) / len(per_seq_entropy)
        success = mean_entropy <= self._validation.max_cross_entropy
        details = f"Mean cross-entropy {mean_entropy:.3f} (threshold {self._validation.max_cross_entropy:.3f})"
        metrics = {
            "mean_cross_entropy": mean_entropy,
            "per_sequence": per_seq_entropy,
        }
        return ValidationResult(success, details, metrics=metrics)
