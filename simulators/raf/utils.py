from matplotlib import pyplot as plt
from typing import Any

class ChemicalReactionNetwork:
    """
    A class representing a Chemical Reaction Network (CRN) with a method to check if it forms a Reflexively Autocatalytic and F-generated (RAF) set.
    Includes ligation (forward) and unligation (reverse) rates for each reaction.
    
    Assumptions:
    - Reactions are provided as a list of tuples: (reactants, products, catalysts, k_lig, k_unlig), where each is a list of strings (species names) for reactants/products/catalysts, and floats for rates.
    - Food set is a list or set of strings (species that are always available).
    - All reactions require at least one catalyst.
    - The check is for the entire network being a RAF set.
    """
    
    def __init__(self, reactions, food_set):
        self.food_set = set(food_set)
        self.reaction_dict = {}
        self.catalysis = {}
        self.rates = {}
        self.species: set[Any] = set()

        for i, (reactants, products, catalysts, k_lig, k_unlig, reversible) in enumerate(reactions):
            rid = i  # Use index as reaction ID
            self.reaction_dict[rid] = {
                'reactants': set(reactants),
                'products': set(products),
                'reversible': reversible,
            }
            self.catalysis[rid] = set(catalysts)
            self.rates[rid] = {'k_lig': k_lig, 'k_unlig': k_unlig}
            self.species |= set(reactants) | set(products)

        self.all_reactions = set(self.reaction_dict.keys())
    
    def compute_closure(self, reaction_set=None):
        """
        Compute the F-closure (producible species) for a given set of reactions.
        Starts with the food set and iteratively adds products of reactions whose reactants are available.
        """
        if reaction_set is None:
            reaction_set = self.all_reactions
        W = set(self.food_set)
        changed = True
        while changed:
            changed = False
            for rid in list(reaction_set):  # Copy to avoid modification issues
                reactants = self.reaction_dict[rid]['reactants']
                if reactants.issubset(W):
                    products = self.reaction_dict[rid]['products']
                    if not products.issubset(W):
                        W.update(products)
                        changed = True
        return W
    
    def is_raf(self):
        """
        Check if the entire network is a RAF set.
        
        - F-generated: All reactants of every reaction are in the F-closure.
        - Reflexively autocatalytic: Every reaction has at least one catalyst in the F-closure.
        
        Returns True if it is a RAF, False otherwise.
        """
        cl = self.compute_closure()
        
        # Check F-generated
        for rid in self.all_reactions:
            reactants = self.reaction_dict[rid]['reactants']
            if not reactants.issubset(cl):
                return False
        
        # Check reflexively autocatalytic
        for rid in self.all_reactions:
            catalysts = self.catalysis[rid]
            if not any(c in cl for c in catalysts):
                return False
        
        return True
    
    def __repr__(self):
        reactions_repr = []
        for rid in sorted(self.all_reactions):
            rx = self.reaction_dict[rid]
            reactants = sorted(rx['reactants'])
            products = sorted(rx['products'])
            catalysts = sorted(self.catalysis[rid])
            k_lig = self.rates[rid]['k_lig']
            k_unlig = self.rates[rid]['k_unlig']
            reversible = rx['reversible']
            reactions_repr.append((reactants, products, catalysts, k_lig, k_unlig, reversible))
        food_repr = sorted(self.food_set)
        return f"ChemicalReactionNetwork(reactions={reactions_repr}, food_set={food_repr})"

    def __str__(self):
        lines = []
        lines.append(f"Chemical Reaction Network:")
        lines.append(f"  Food set: {sorted(self.food_set)}")
        lines.append(f"  Species: {len(self.species)}")
        lines.append(f"  Reactions: {len(self.all_reactions)}")
        for rid in sorted(self.all_reactions):
            rx = self.reaction_dict[rid]
            reactants = ' + '.join(sorted(rx['reactants']))
            products = ' + '.join(sorted(rx['products']))
            catalysts = ', '.join(sorted(self.catalysis[rid]))
            k_lig = self.rates[rid]['k_lig']
            k_unlig = self.rates[rid]['k_unlig']
            rev = ' (reversible)' if rx['reversible'] else ''
            lines.append(f"    {rid}: {reactants} -> {products} [catalysts: {catalysts}, k_lig={k_lig}, k_unlig={k_unlig}{rev}]")
        is_raf_str = "Yes" if self.is_raf() else "No"
        lines.append(f"  Is RAF: {is_raf_str}")
        return '\n'.join(lines)