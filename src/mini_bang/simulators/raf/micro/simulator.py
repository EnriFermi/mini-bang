from __future__ import annotations

import json
import random
from collections import Counter
from typing import Any

from mini_bang.simulators.raf.utils import ChemicalReactionNetwork
from mini_bang.simulators.base.micro.simulator import MicroSimulatorBase


class CRNSimulator(MicroSimulatorBase):
    def __init__(
        self,
        crn: ChemicalReactionNetwork,
        min_food_conc: int = 5,
        step_limit: int = 10000,
        V: float = 1.0,
    ):
        self.crn = crn
        self.min_food_conc = min_food_conc
        self.step_limit = step_limit
        self.V = V

    def sample(self, seed: int | None = None, snapshot_times: tuple[float, ...] = (1.0,)) -> dict[str, list[int]]:
        """
        Simulates chemical reaction network dynamics and records species traces over time.
        """
        if seed is not None:
            random.seed(seed)

        species = self.crn.species
        food = self.crn.food_set
        reactions = self.crn.reaction_dict
        catalysis = self.crn.catalysis
        rates = self.crn.rates

        counts = {s: (self.min_food_conc if s in food else 0) for s in species}

        times = []
        traces = {s: [] for s in species}
        snapshot_times = sorted(snapshot_times)
        st = 0

        t = 0.0
        events = 0

        while events < self.step_limit:
            props: list[float] = []
            actions: list[tuple[str, int]] = []

            for rid in reactions:
                rx = reactions[rid]
                reactants = list(rx["reactants"])
                products = list(rx["products"])
                n_cat = sum(counts.get(c, 0) for c in catalysis[rid])
                effective_rate = rates[rid]["k_unlig"] + rates[rid]["k_lig"] * n_cat

                prop_f = self._get_propensity(counts, reactants, effective_rate)
                props.append(prop_f)
                actions.append(("lig", rid))

                prop_b = self._get_propensity(counts, products, effective_rate)
                props.append(prop_b)
                actions.append(("cle", rid))

            total_prop = sum(props)
            if total_prop <= 0:
                break

            dt = random.expovariate(total_prop)
            t += dt
            pick = random.uniform(0, total_prop)
            cum = 0.0
            sel = None
            for idx, pr in enumerate(props):
                cum += pr
                if pick <= cum:
                    sel = idx
                    break
            if sel is None:
                break
            typ, rid = actions[sel]
            rx = reactions[rid]

            if typ == "lig":
                for r in rx["reactants"]:
                    counts[r] -= 1
                    if counts[r] < 0:
                        counts[r] = 0
                for p in rx["products"]:
                    counts[p] = counts.get(p, 0) + 1
            else:
                for p in rx["products"]:
                    counts[p] -= 1
                    if counts[p] < 0:
                        counts[p] = 0
                for r in rx["reactants"]:
                    counts[r] = counts.get(r, 0) + 1

            for f in food:
                if counts.get(f, 0) < self.min_food_conc:
                    counts[f] = self.min_food_conc

            events += 1
            if (events + 1e-10) / self.step_limit >= snapshot_times[st]:
                times.append(t)
                for s in species:
                    traces[s].append(counts.get(s, 0))
                st += 1
                if st >= len(snapshot_times):
                    break

        return traces

    def _get_propensity(self, counts: dict[str, int], items: Any, effective_rate: float) -> float:
        if not items:
            return 0.0
        counter = Counter(items)
        can_react = all(counts.get(s, 0) >= mult for s, mult in counter.items())
        if not can_react:
            return 0.0
        contrib = 1.0
        for s, mult in counter.items():
            ff = 1.0
            for i in range(mult):
                ff *= counts[s] - i
            contrib *= ff
        order = len(items)
        div = self.V ** (order - 1) if order > 1 else 1.0
        return effective_rate * contrib / div
