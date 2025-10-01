import random
import numpy as np
from typing import Callable
from scipy.special import comb
from collections import defaultdict
from simulators.raf.utils import ChemicalReactionNetwork
from simulators.raf.subordinate.simulator import CRNSimulator



class MasterModel:
    def __init__(self, 
                 M0=10, 
                 alpha: Callable[[int], float] = lambda i: 0.01**i, 
                 K=4, 
                 p=0.005, 
                 k_lig=1.0, 
                 k_unlig=0.05, 
                 max_events = 25000,
                 seed=None):
        """
        Initialize TAP model with parameters from the paper.
        
        Parameters:
        - M0: Number of initial molecular species (default: 10)
        - alpha: Base reaction probability (default: 0.01)
        - K: Maximum number of reactants (default: 4)
        - p: Catalysis probability (default: 0.005)
        - k_lig: Catalyzed ligation rate (default: 1.0)
        - k_unlig: Uncatalyzed rate (default: 0.05)
        - max_events: Number of events in simulation
        - seed: Random seed for reproducibility
        """
        
        self.M0 = M0
        self.alpha = alpha
        self.K = K
        self.p = p
        self.k_lig = k_lig
        self.k_unlig = k_unlig
        self.max_events = max_events
        
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

    def get_slave_simulator(self, M, max_raf=False, prune_catalysts=False):
        _, reactions, catalysis, _ = self._build_chemical_network(M)
        
        food_set = set(str(i) for i in range(1, self.M0 + 1))
        
        crn_reactions = []
        for idx, (reactants, product) in enumerate(reactions):
            reactants_list = list(reactants)
            products_list = [product]
            catalysts_list = catalysis[idx]
            crn_reactions.append((reactants_list, products_list, catalysts_list, self.k_lig, self.k_unlig, True))
        
        crn = ChemicalReactionNetwork(crn_reactions, food_set)
        if max_raf:
            crn = self._cut_to_max_raf(crn)
        if prune_catalysts:
            crn = self._prune_catalysts(crn)
        simulator = CRNSimulator(crn, max_events=self.max_events)
        return simulator

    def _build_chemical_network(self, M):
        """
        Run the TAP model until exactly M molecular species have been produced.
        """

        species = [str(i) for i in range(1, self.M0 + 1)]
        catalysis = defaultdict(list)
        reactions = []
        produced_by = {}

        Mt = self.M0
        t = 0
        
        while Mt < M:
            t += 1
            Mt_prev = Mt
            
            # For each possible number of reactants
            for i in range(1, min(self.K + 1, Mt_prev + 1)):

                alpha_i = self.alpha(i)
                n_combinations = comb(Mt_prev, i, exact=True)
                s_i = alpha_i * n_combinations                
                r_i = np.random.poisson(s_i)
                
                for j in range(r_i):
                    Mt += 1
                    new_species = str(Mt)
                    
                    reactant_ids = np.random.choice(Mt_prev, size=i, replace=False) + 1
                    reactants = tuple(sorted(str(k) for k in reactant_ids))
                    
                    reaction_idx = len(reactions)
                    reactions.append((reactants, new_species))
                    produced_by[new_species] = reaction_idx
                    species.append(new_species)
                    
                    for y_int in range(self.M0 + 1, Mt + 1):
                        y = str(y_int)

                        if y in produced_by and random.random() < self.p:
                            catalysis[produced_by[y]].append(new_species)
                        
                        if random.random() < self.p:
                            catalysis[reaction_idx].append(y)
        return species, reactions, catalysis, produced_by

    def _cut_to_max_raf(self, crn):
        """
        Cut the ChemicalReactionNetwork to its maximal RAF set.
        Returns a new ChemicalReactionNetwork containing only the reactions in the max RAF.
        """
        def is_raf_subset(reaction_subset):
            if not reaction_subset:
                return False
            temp_crn = ChemicalReactionNetwork(
                [
                    (
                        list(crn.reaction_dict[rid]['reactants']),
                        list(crn.reaction_dict[rid]['products']),
                        list(crn.catalysis[rid]),
                        crn.rates[rid]['k_lig'],
                        crn.rates[rid]['k_unlig'],
                        crn.reaction_dict[rid]['reversible']
                    ) for rid in reaction_subset
                ],
                crn.food_set
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
                list(crn.reaction_dict[rid]['reactants']),
                list(crn.reaction_dict[rid]['products']),
                list(crn.catalysis[rid]),
                crn.rates[rid]['k_lig'],
                crn.rates[rid]['k_unlig'],
                crn.reaction_dict[rid]['reversible']
            ) for rid in max_raf
        ]
        return ChemicalReactionNetwork(max_raf_reactions, crn.food_set)

    def _prune_catalysts(self, crn):
        """
        Prune catalysts in the ChemicalReactionNetwork to remove as many as possible while keeping it a RAF.
        For each reaction, keep only one valid catalyst in the closure.
        Returns a new ChemicalReactionNetwork with pruned catalysts.
        """
        cl = crn.compute_closure()
        
        new_crn_reactions = []
        for rid in crn.all_reactions:
            valid_cats = crn.catalysis[rid] & cl
            catalysts_list = []
            if valid_cats:
                # Keep one arbitrary catalyst
                one_cat = next(iter(valid_cats))
                catalysts_list = [one_cat]
            
            new_crn_reactions.append((
                list(crn.reaction_dict[rid]['reactants']),
                list(crn.reaction_dict[rid]['products']),
                catalysts_list,
                crn.rates[rid]['k_lig'],
                crn.rates[rid]['k_unlig'],
                crn.reaction_dict[rid]['reversible']
            ))
        
        new_crn = ChemicalReactionNetwork(new_crn_reactions, crn.food_set)
        return new_crn