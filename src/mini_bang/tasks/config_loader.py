from __future__ import annotations

import copy
import json
from importlib import resources
from typing import Any


def load_task_config(package: str, filename: str = "config.json") -> dict[str, Any]:
    resource = resources.files(package).joinpath(filename)
    with resource.open("r", encoding="utf-8") as cfg:
        data = json.load(cfg)
    return copy.deepcopy(data)
