from __future__ import annotations

import random
from collections import defaultdict
from math import comb
from typing import Callable, Iterable, Sequence, Union

import numpy as np
from pydantic import BaseModel, Field, create_model

from mini_bang.simulators.raf.utils import ChemicalReactionNetwork
from mini_bang.simulators.raf.micro.simulator import CRNSimulator
from mini_bang.simulators.base.macro.simulator import MacroSimulatorBase

COMPLEXITY_LIMIT = 1000


class MasterModel(MacroSimulatorBase):
    def __init__(
        self,
        complexity: float = 0.5,
        M0: int = 10,
        alpha: Callable[[int], float] = lambda i: 0.01**i,
        K: int = 4,
        p: float = 0.005,
        k_lig: float = 1.0,
        k_unlig: float = 0.05,
        step_limit: int = 10000,
        seed: int | None = None,
    ):
        """
        Initialize TAP model with parameters from the paper.
        """
        super().__init__(complexity=complexity)
        self.M0 = M0
        self.alpha = alpha
        self.K = K
        self.p = p
        self.k_lig = k_lig
        self.k_unlig = k_unlig
        self.step_limit = step_limit

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

    def get_saturation_description(self) -> type[BaseModel]:
        return create_model(
            "ParameterSaturation",
            value=(int, Field(gt=self.M0, le=COMPLEXITY_LIMIT)),
            description=(
                str,
                Field(default="Approximate size of CRN to grow in the micro simulator"),
            ),
        )

    def get_micro_simulator(
        self,
        M: Union[int, Sequence[int]],
        max_raf: bool = False,
        prune_catalysts: bool = False,
    ):
        """
        Build one simulator (M is int) or a sequence (M is list/tuple of ints).
        """
        if isinstance(M, Iterable) and not isinstance(M, (str, bytes)):
            Ms = [int(x) for x in M]
            if any(m <= self.M0 for m in Ms):
                raise ValueError(f"All M must be > M0={self.M0}")
            return self._build_chain(Ms, max_raf=max_raf, prune_catalysts=prune_catalysts)
        m = int(M)
        if m <= self.M0:
            raise ValueError(f"M must be > M0={self.M0}")
        return self._build_chain([m], max_raf=max_raf, prune_catalysts=prune_catalysts)[0]

    def _build_chain(
        self,
        Ms: Sequence[int],
        max_raf: bool,
        prune_catalysts: bool,
    ) -> list[CRNSimulator]:
        Ms_sorted = sorted(Ms)
        Mmax = Ms_sorted[-1]

        species = [str(i) for i in range(1, self.M0 + 1)]
        reactions: list[tuple[tuple[str, ...], str]] = []
        catalysis: defaultdict[int, list[str]] = defaultdict(list)
        produced_by: dict[str, int] = {}

        food_set = set(str(i) for i in range(1, self.M0 + 1))
        snapshots: dict[int, CRNSimulator] = {}

        Mt = self.M0
        next_idx = 0

        def _emit_snapshot_for_current_Mt() -> CRNSimulator:
            """Build a CRN from reactions/catalysis produced so far and emit a simulator."""
            rcount = max(0, Mt - self.M0)
            crn_reactions = []
            for rid in range(rcount):
                reactants, product = reactions[rid]
                cats = catalysis.get(rid, [])
                crn_reactions.append(
                    (
                        list(reactants),
                        [product],
                        list(cats),
                        self.k_lig,
                        self.k_unlig,
                        True,
                    )
                )
            crn = ChemicalReactionNetwork(crn_reactions, food_set)
            if max_raf:
                crn = self._cut_to_max_raf(crn)
            if prune_catalysts:
                crn = self._prune_catalysts(crn)
            return CRNSimulator(crn, step_limit=self.step_limit)

        while Mt < Mmax:
            Mt_prev = Mt
            for i in range(1, min(self.K + 1, Mt_prev + 1)):
                alpha_i = self.alpha(i)
                n_combinations = comb(Mt_prev, i)
                s_i = alpha_i * n_combinations
                r_i = np.random.poisson(s_i)

                for _ in range(r_i):
                    Mt += 1
                    new_species = str(Mt)

                    reactant_ids = np.random.choice(Mt_prev, size=i, replace=False) + 1
                    reactants = tuple(sorted(str(k) for k in reactant_ids))

                    rid = len(reactions)
                    reactions.append((reactants, new_species))
                    produced_by[new_species] = rid
                    species.append(new_species)

                    for y_int in range(self.M0 + 1, Mt + 1):
                        y = str(y_int)

                        if y == new_species:
                            if random.random() < self.p:
                                catalysis[rid].append(new_species)
                        else:
                            if y in produced_by and random.random() < self.p:
                                catalysis[produced_by[y]].append(new_species)

                            if random.random() < self.p:
                                catalysis[rid].append(y)

                    while next_idx < len(Ms_sorted) and Mt >= Ms_sorted[next_idx]:
                        m_hit = Ms_sorted[next_idx]
                        snapshots[m_hit] = _emit_snapshot_for_current_Mt()
                        next_idx += 1

                    if Mt >= Mmax:
                        break
                if Mt >= Mmax:
                    break
            if Mt >= Mmax:
                break
        return [snapshots[m] for m in Ms]

    def _cut_to_max_raf(self, crn):
        """
        Cut the ChemicalReactionNetwork to its maximal RAF set.
        """

        def is_raf_subset(reaction_subset):
            if not reaction_subset:
                return False
            temp_crn = ChemicalReactionNetwork(
                [
                    (
                        list(crn.reaction_dict[rid]["reactants"]),
                        list(crn.reaction_dict[rid]["products"]),
                        list(crn.catalysis[rid]),
                        crn.rates[rid]["k_lig"],
                        crn.rates[rid]["k_unlig"],
                        crn.reaction_dict[rid]["reversible"],
                    )
                    for rid in reaction_subset
                ],
                crn.food_set,
            )
            return temp_crn.is_raf()

        current_set = set(crn.all_reactions)
        max_raf = set(current_set)

        if not crn.is_raf():
            while current_set:
                found = False
                for rid in list(current_set):
                    test_set = current_set - {rid}
                    if is_raf_subset(test_set):
                        current_set = test_set
                        max_raf = test_set
                        found = True
                        break
                if not found:
                    break
            if not max_raf:
                return ChemicalReactionNetwork([], crn.food_set)

        max_raf_reactions = [
            (
                list(crn.reaction_dict[rid]["reactants"]),
                list(crn.reaction_dict[rid]["products"]),
                list(crn.catalysis[rid]),
                crn.rates[rid]["k_lig"],
                crn.rates[rid]["k_unlig"],
                crn.reaction_dict[rid]["reversible"],
            )
            for rid in max_raf
        ]
        return ChemicalReactionNetwork(max_raf_reactions, crn.food_set)

    def _prune_catalysts(self, crn):
        """
        Prune catalysts in the ChemicalReactionNetwork to remove as many as possible while keeping it a RAF.
        """
        cl = crn.compute_closure()

        new_crn_reactions = []
        for rid in crn.all_reactions:
            valid_cats = crn.catalysis[rid] & cl
            catalysts_list = []
            if valid_cats:
                one_cat = next(iter(valid_cats))
                catalysts_list = [one_cat]

            new_crn_reactions.append(
                (
                    list(crn.reaction_dict[rid]["reactants"]),
                    list(crn.reaction_dict[rid]["products"]),
                    catalysts_list,
                    crn.rates[rid]["k_lig"],
                    crn.rates[rid]["k_unlig"],
                    crn.reaction_dict[rid]["reversible"],
                )
            )

        return ChemicalReactionNetwork(new_crn_reactions, crn.food_set)
