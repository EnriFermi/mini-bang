from __future__ import annotations

from typing import Any, Callable

from mini_bang.simulators.raf.macro.simulator import MasterModel


def _alpha(i: int) -> float:
    return 0.05 * (3 - i)


def create_master_model(**overrides: Any) -> MasterModel:
    defaults = {
        "complexity": 0.5,
        "M0": 2,
        "alpha": _alpha,
        "K": 3,
        "p": 0.5,
        "k_lig": 1.0,
        "k_unlig": 0.05,
        "step_limit": 50_000,
    }
    params = {**defaults, **{k: v for k, v in overrides.items() if v is not None}}
    return MasterModel(**params)
