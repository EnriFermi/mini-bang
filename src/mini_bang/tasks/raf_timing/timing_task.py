from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

from mini_bang.framework.task import TaskEnvironment, TaskSubmission, ValidationResult
from mini_bang.tasks.base import RemoteSimulationTask
from mini_bang.tasks.raf_common.api_client import RAFSimulationAPI
from mini_bang.tasks.raf_timing.timing_api import RAFTimingAPI


@dataclass
class _ValidationConfig:
    loglik_tolerance: float
    epsilon: float


class RAFLevel2TimingTask(RemoteSimulationTask):
    """Level 2 task: estimate first-hit distributions per species for fixed saturation."""
    task_id = "raf/timing-v1"
    config_package = __package__

    def __init__(self) -> None:
        super().__init__()
        cfg = self.config_copy()
        dataset_cfg = cfg.get("dataset", {})
        self._saturation = int(dataset_cfg.get("saturation", 18))
        self._snapshot_times: List[float] = list(dataset_cfg.get("snapshot_times", []))
        self._simulator_seeds: List[int] = [int(s) for s in dataset_cfg.get("simulator_seeds", [])]
        if not self._simulator_seeds:
            raise ValueError("Simulator seeds must be provided")
        self._test_runs = int(dataset_cfg.get("test_runs", 8))

        val_cfg = cfg.get("validation", {})
        self._validation = _ValidationConfig(
            loglik_tolerance=float(val_cfg.get("loglik_tolerance", 0.3)),
            epsilon=float(val_cfg.get("epsilon", 1e-6)),
        )

        api_cfg = cfg.get("api", {})
        self._simulator_id = api_cfg.get("simulator_id", "raf")
        self._instructions = api_cfg.get("instructions", "")
        self._max_generate_calls = int(api_cfg.get("max_generate_calls", 50))
        self._description = cfg.get("description", "")
        self._metadata_template: Dict[str, Any] = cfg.get("metadata", {})

        self._latest_test_data: dict[str, Any] | None = None

    def _build_remote_environment(self) -> TaskEnvironment:
        trajectory_client = RAFSimulationAPI(simulator_id=self._simulator_id)
        # Only prepare held-out test data for validation; agents gather their own training data
        test_data = self._prepare_test_data(trajectory_client)
        self._latest_dataset = None
        self._latest_test_data = test_data

        api = RAFTimingAPI(description=self._instructions, client=trajectory_client, max_generate_calls=self._max_generate_calls)
        metadata = dict(self._metadata_template)
        metadata.update(
            {
                "simulator_id": self._simulator_id,
                "snapshot_times": list(self._snapshot_times),
                "saturation": self._saturation,
                "mcp": trajectory_client.describe(),
            }
        )
        return TaskEnvironment(description=self._description, api=api, metadata=metadata)

    def _prepare_test_data(self, client: RAFSimulationAPI) -> dict[str, Any]:
        """Prepare only held-out test data for validation."""
        snapshot_times = tuple(self._snapshot_times)
        test_data: dict[str, Any] = {}
        for seed in self._simulator_seeds:
            macro_params = {"seed": seed}
            test_payload = client.generate_samples(
                saturation=self._saturation,
                runs=self._test_runs,
                snapshot_times=snapshot_times,
                extras=["first_hits"],
                macro_params=macro_params,
                micro_params={},
                sample_params={},
            )
            test_data[str(seed)] = {
                "macro_seed": seed,
                "first_hits": test_payload.get("first_hits", []),
            }
        return test_data

    def validate(self, submission: TaskSubmission) -> ValidationResult:
        answer = submission.answer
        if not isinstance(answer, dict):
            return ValidationResult(False, "Submission answer must be a mapping")
        distributions = answer.get("distributions")
        if not isinstance(distributions, dict):
            return ValidationResult(False, "Missing 'distributions' mapping in submission")

        if self._latest_test_data is None:
            return ValidationResult(False, "Task dataset not initialized")

        snapshot_times = tuple(self._snapshot_times)
        categories = [str(t) for t in snapshot_times] + ["none"]

        metrics: Dict[str, Any] = {}
        ll_diffs: List[float] = []

        for seed_str, test_info in self._latest_test_data.items():
            seed_dist = distributions.get(seed_str)
            if seed_dist is None:
                # Accept a default distribution under special keys or a single provided distribution
                seed_dist = distributions.get("default") or distributions.get("*")
            if seed_dist is None and len(distributions) == 1:
                seed_dist = next(iter(distributions.values()))
            if seed_dist is None:
                return ValidationResult(False, f"Missing distribution for simulator seed {seed_str}")
            if not isinstance(seed_dist, dict):
                return ValidationResult(False, f"Distribution for seed {seed_str} must be a mapping")

            species_keys = self._extract_species_from_hits(test_info["first_hits"])

            for species in species_keys:
                pred = seed_dist.get(species)
                if pred is None:
                    return ValidationResult(False, f"Missing distribution for species {species} in seed {seed_str}")
                try:
                    probs = self._normalize_distribution(pred, categories)
                except ValueError as exc:
                    return ValidationResult(False, str(exc))

                ll_true, ll_pred = self._compute_loglik(species, test_info["first_hits"], snapshot_times, probs)
                ll_diffs.append(ll_true - ll_pred)

        avg_diff = sum(ll_diffs) / len(ll_diffs) if ll_diffs else float("inf")
        metrics["avg_loglik_gap"] = avg_diff

        success = avg_diff <= self._validation.loglik_tolerance
        details = f"Average log-likelihood gap {avg_diff:.3f} (threshold {self._validation.loglik_tolerance:.3f})"
        return ValidationResult(success, details, metrics=metrics)

    def _extract_species(self, trajectories: Iterable[Dict[str, Any]]) -> List[str]:
        seen: set[str] = set()
        for traj in trajectories:
            seen.update(traj.keys())
        return sorted(seen, key=lambda s: int(s))

    def _extract_species_from_hits(self, first_hits: Iterable[Dict[str, Any]]) -> List[str]:
        seen: set[str] = set()
        for record in first_hits:
            seen.update(str(k) for k in record.keys())
        return sorted(seen, key=lambda s: int(s))

    def _normalize_distribution(self, data: Dict[str, Any], categories: List[str]) -> Dict[str, float]:
        total = 0.0
        probs: Dict[str, float] = {}
        for cat in categories:
            raw = data.get(cat, None)
            if raw is None and cat != "none":
                try:
                    numeric_key = float(cat)
                except ValueError:
                    numeric_key = None
                if numeric_key is not None and numeric_key in data:
                    raw = data[numeric_key]
            if raw is None and cat == "none":
                raw = data.get("none", data.get(None, 0.0))
            value = float(raw if raw is not None else 0.0)
            if value < 0:
                raise ValueError(f"Negative probability for category {cat}")
            probs[cat] = value
            total += value
        if total <= 0:
            raise ValueError("Distribution probabilities must sum to positive value")
        return {cat: val / total for cat, val in probs.items()}

    def _compute_loglik(
        self,
        species: str,
        first_hits: List[Dict[str, Any]],
        snapshot_times: Iterable[float],
        probs: Dict[str, float],
    ) -> tuple[float, float]:
        epsilon = self._validation.epsilon
        categories = [str(t) for t in snapshot_times]
        total = 0
        ll_true = 0.0
        ll_pred = 0.0

        counts: Dict[str, int] = {cat: 0 for cat in categories}
        counts["none"] = 0

        for hit in first_hits:
            value = hit.get(species)
            if value is None:
                cat = "none"
            else:
                cat = str(value)
                if cat not in counts:
                    cat = "none"
            counts[cat] += 1
            total += 1

        if total == 0:
            return 0.0, 0.0

        for cat, count in counts.items():
            if count == 0:
                continue
            freq = count / total
            ll_true += freq * math.log(freq + epsilon)
            ll_pred += freq * math.log(probs.get(cat, 0.0) + epsilon)

        return ll_true, ll_pred
