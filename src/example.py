from __future__ import annotations

import multiprocessing
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from mini_bang.agent_connector import AgentConnector
from mini_bang.agents import (
    LangChainRAFMechanismProbabilityAgent,
    LangChainRAFMechanismSequenceAgent,
    LangChainRAFPredictiveAgent,
    LangChainRAFSignatureAgent,
    LangChainRAFTimingAgent,
)


def main() -> None:
    connector = AgentConnector()
    experiments = [
        # ("raf/timing-v1", LangChainRAFTimingAgent),
        # ("raf/signature-v1", LangChainRAFSignatureAgent),
        ("raf/mechanism-prob-v1", LangChainRAFMechanismProbabilityAgent),
        # ("raf/mechanism-seq-v1", LangChainRAFMechanismSequenceAgent),
        # ("raf/predictive-v1", LangChainRAFPredictiveAgent),
    ]

    for task_id, agent_cls in experiments:
        agent = agent_cls()
        result = connector.run(agent, task_id=task_id)
        print(f"{task_id}: {result.validation.details}")
        if result.validation.metrics:
            print(result.validation.metrics)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
