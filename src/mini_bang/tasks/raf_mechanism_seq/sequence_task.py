from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List

from mini_bang.framework.task import TaskEnvironment, TaskSubmission, ValidationResult
from mini_bang.tasks.base import RemoteSimulationTask
from mini_bang.tasks.raf_common.api_client import RAFSimulationAPI
from mini_bang.tasks.raf_mechanism_seq.sequence_api import RAFMechanismSequenceAPI


@dataclass
class _ValidationConfig:
    epsilon: float
    max_ll_gap: float


class RAFLevel5MechanismSequenceTask(RemoteSimulationTask):
    """Level 5 mechanism inference: model distributions along a RAF sequence."""
    task_id = "raf/mechanism-seq-v1"
    config_package = __package__

    def __init__(self) -> None:
        super().__init__()
        cfg = self.config_copy()
        dataset_cfg = cfg.get("dataset", {})
        self._sequences: List[dict[str, Any]] = dataset_cfg.get("sequences", [])
        if not self._sequences:
            raise ValueError("Mechanism sequence task requires sequence definitions")
        self._train_seeds: Dict[str, List[int]] = {
            seq_id: [int(s) for s in seeds]
            for seq_id, seeds in (dataset_cfg.get("train_seeds") or {}).items()
        }
        self._test_seeds: Dict[str, List[int]] = {
            seq_id: [int(s) for s in seeds]
            for seq_id, seeds in (dataset_cfg.get("test_seeds") or {}).items()
        }
        self._runs = int(dataset_cfg.get("runs", 3))
        self._snapshot_times: List[float] = [float(t) for t in dataset_cfg.get("snapshot_times", [])]

        val_cfg = cfg.get("validation", {})
        self._validation = _ValidationConfig(
            epsilon=float(val_cfg.get("epsilon", 1e-6)),
            max_ll_gap=float(val_cfg.get("max_ll_gap", 80.0)),
        )

        self._description = cfg.get("description", "")
        self._metadata_template: Dict[str, Any] = cfg.get("metadata", {})
        api_cfg = cfg.get("api", {})
        self._simulator_id = api_cfg.get("simulator_id", "raf")
        self._instructions = api_cfg.get("instructions") or self._description
        self._max_generate_calls = int(api_cfg.get("max_generate_calls", 50))

        self._dataset: dict[str, Any] | None = None
        self._test_data: Dict[str, Dict[str, Any]] | None = None

    def _build_remote_environment(self) -> TaskEnvironment:
        client = RAFSimulationAPI(simulator_id=self._simulator_id, instructions=self._instructions)
        test_data = self._prepare_test_data(client)
        self._dataset = None
        self._test_data = test_data

        api = RAFMechanismSequenceAPI(description=self._instructions, dataset={}, client=client, max_generate_calls=self._max_generate_calls)
        metadata = dict(self._metadata_template)
        metadata.update(
            {
                "simulator_id": self._simulator_id,
                "snapshot_times": list(self._snapshot_times),
                "sequences": self._sequences,
                "mcp": client.describe(),
            }
        )
        return TaskEnvironment(description=self._description, api=api, metadata=metadata)

    def _prepare_test_data(
        self, client: RAFSimulationAPI
    ) -> Dict[str, Dict[str, Any]]:
        """Generate held-out sequences for validation only."""
        test_data: Dict[str, Dict[str, Any]] = {}
        for seq_cfg in self._sequences:
            seq_id = seq_cfg["id"]
            saturations = [int(t) for t in seq_cfg["saturations"]]
            if seq_id not in self._test_seeds:
                raise ValueError(f"Missing test seeds for sequence {seq_id}")
            test_data[seq_id] = {
                "saturations": list(saturations),
                "samples": [],
            }
            for macro_seed in self._test_seeds.get(seq_id, []):
                response = client.generate_samples(
                    saturation=saturations,
                    runs=self._runs,
                    snapshot_times=self._snapshot_times,
                    extras=[],
                    macro_params={"seed": macro_seed},
                )
                sequence_records = response.get("sequence", [])
                final_record = sequence_records[-1] if sequence_records else {}
                test_data[seq_id]["samples"].append(
                    {
                        "macro_seed": macro_seed,
                        "trajectories": final_record.get("trajectories", []),
                    }
                )
        return test_data

    def validate(self, submission: TaskSubmission) -> ValidationResult:
        if self._test_data is None:
            return ValidationResult(False, "Task dataset not initialised")
        answer = submission.answer
        if not isinstance(answer, dict):
            return ValidationResult(False, "Submission answer must be a mapping")
        predictions = answer.get("predicted_means")
        if not isinstance(predictions, dict):
            return ValidationResult(False, "Submission must contain 'predicted_means' mapping")

        ll_gaps: Dict[str, float] = {}
        for seq_id, info in self._test_data.items():
            samples = info["samples"]
            if seq_id not in predictions:
                return ValidationResult(False, f"Missing predictions for sequence {seq_id}")
            pred_map = {str(k): float(v) for k, v in predictions[seq_id].items()}
            if not samples:
                continue
            # Compute empirical means from test data (oracle)
            empirical: Dict[str, float] = {}
            counts: Dict[str, List[int]] = {}
            for sample in samples:
                for traj in sample.get("trajectories", []):
                    for species, series in traj.items():
                        counts.setdefault(species, []).append(int(series[-1]))
            for species, values in counts.items():
                empirical[species] = sum(values) / len(values)

            predicted_ll = self._poisson_log_likelihood(pred_map, counts)
            best_ll = self._poisson_log_likelihood(empirical, counts)
            ll_gaps[seq_id] = max(0.0, best_ll - predicted_ll)

        mean_gap = sum(ll_gaps.values()) / len(ll_gaps) if ll_gaps else 0.0
        success = mean_gap <= self._validation.max_ll_gap
        details = f"Mean log-likelihood gap {mean_gap:.3f} (threshold {self._validation.max_ll_gap:.3f})"
        metrics = {
            "mean_ll_gap": mean_gap,
            "per_sequence": ll_gaps,
        }
        return ValidationResult(success, details, metrics=metrics)

    def _poisson_log_likelihood(
        self,
        rates: Dict[str, float],
        observations: Dict[str, List[int]],
    ) -> float:
        epsilon = self._validation.epsilon
        total = 0.0
        for species, values in observations.items():
            lam = max(epsilon, rates.get(species, epsilon))
            for value in values:
                total += value * math.log(lam) - lam - math.lgamma(value + 1)
        return total
