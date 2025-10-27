from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from mini_bang.framework.task import TaskEnvironment, TaskSubmission, ValidationResult
from mini_bang.tasks.base import RemoteSimulationTask
from mini_bang.tasks.raf_common.api_client import RAFSimulationAPI
from mini_bang.tasks.raf_signature.signature_api import RAFSignatureAPI


@dataclass
class _ValidationConfig:
    min_macro_f1: float
    epsilon: float


class RAFLevel4SignatureTask(RemoteSimulationTask):
    """Level 4 signature recognition: classify RAF presence across saturations."""
    task_id = "raf/signature-v1"
    config_package = __package__

    def __init__(self) -> None:
        super().__init__()
        cfg = self.config_copy()
        dataset_cfg = cfg.get("dataset", {})
        self._saturations: List[int] = [int(t) for t in dataset_cfg.get("saturations", [])]
        if not self._saturations:
            raise ValueError("Signature task requires at least one saturation value")
        self._snapshot_times: List[float] = [float(t) for t in dataset_cfg.get("snapshot_times", [])]
        self._train_seeds: List[int] = [int(s) for s in dataset_cfg.get("train_seeds", [])]
        self._test_seeds: List[int] = [int(s) for s in dataset_cfg.get("test_seeds", [])]
        if not self._train_seeds or not self._test_seeds:
            raise ValueError("Train and test seeds must be provided")
        self._train_runs = int(dataset_cfg.get("train_runs", 3))
        self._test_runs = int(dataset_cfg.get("test_runs", 2))

        val_cfg = cfg.get("validation", {})
        self._validation = _ValidationConfig(
            min_macro_f1=float(val_cfg.get("min_macro_f1", 0.4)),
            epsilon=float(val_cfg.get("epsilon", 1e-6)),
        )

        api_cfg = cfg.get("api", {})
        self._simulator_id = api_cfg.get("simulator_id", "raf")
        self._instructions = api_cfg.get("instructions") or cfg.get("description", "")
        self._max_generate_calls = int(api_cfg.get("max_generate_calls", 50))
        self._description = cfg.get("description", "")
        self._metadata_template: Dict[str, Any] = cfg.get("metadata", {})

        self._dataset: dict[str, Any] | None = None
        self._test_data: Dict[str, Dict[str, bool]] | None = None

    def _build_remote_environment(self) -> TaskEnvironment:
        client = RAFSimulationAPI(simulator_id=self._simulator_id, instructions=self._instructions)
        test_data = self._prepare_test_data(client)
        self._dataset = None
        self._test_data = test_data

        api = RAFSignatureAPI(description=self._instructions, dataset={}, client=client, max_generate_calls=self._max_generate_calls)
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
    ) -> Dict[str, Dict[str, bool]]:
        """Prepare only held-out truth for validation."""
        test_truth: Dict[str, Dict[str, bool]] = {str(t): {} for t in self._saturations}
        for sat in self._saturations:
            sat_key = str(sat)
            for seed in self._test_seeds:
                payload = client.generate_samples(
                    saturation=sat,
                    runs=self._test_runs,
                    snapshot_times=self._snapshot_times,
                    extras=["is_raf"],
                    macro_params={"seed": seed},
                )
                test_truth[sat_key][str(seed)] = bool(payload.get("is_raf", False))
        return test_truth

    def validate(self, submission: TaskSubmission) -> ValidationResult:
        if self._test_data is None:
            return ValidationResult(False, "Task dataset not initialised")

        answer = submission.answer
        if not isinstance(answer, dict):
            return ValidationResult(False, "Submission answer must be a mapping")
        predictions = answer.get("predictions")
        if not isinstance(predictions, dict):
            return ValidationResult(False, "Submission must contain 'predictions' mapping")

        per_sat_f1: Dict[str, float] = {}
        for sat_key, truth_map in self._test_data.items():
            pred_map = predictions.get(sat_key)
            if pred_map is None:
                return ValidationResult(False, f"Missing predictions for saturation {sat_key}")
            tp = fp = fn = 0
            for seed_str, truth in truth_map.items():
                if seed_str not in pred_map:
                    return ValidationResult(False, f"Missing prediction for seed {seed_str} at saturation {sat_key}")
                value = pred_map[seed_str]
                if isinstance(value, dict) and "probability" in value:
                    prob = float(value["probability"])
                elif isinstance(value, (int, float)):
                    prob = float(value)
                else:
                    prob = 1.0 if bool(value) else 0.0
                prob = max(self._validation.epsilon, min(1.0 - self._validation.epsilon, prob))
                pred = prob >= 0.5
                if pred and truth:
                    tp += 1
                elif pred and not truth:
                    fp += 1
                elif not pred and truth:
                    fn += 1
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2 * precision * recall / (precision + recall)
            per_sat_f1[sat_key] = f1

        macro_f1 = sum(per_sat_f1.values()) / len(per_sat_f1)
        success = macro_f1 >= self._validation.min_macro_f1
        details = f"Macro F1 {macro_f1:.3f} (threshold {self._validation.min_macro_f1:.3f})"
        metrics = {
            "macro_f1": macro_f1,
            "per_saturation": per_sat_f1,
        }
        return ValidationResult(success, details, metrics=metrics)
