from simulators.raf.utils import ChemicalReactionNetwork
from simulators.base.micro.simulator import MicroSimulatorBase
from collections import Counter
import random
import json

class CRNSimulator(MicroSimulatorBase):
    def __init__(self, 
                 crn: ChemicalReactionNetwork,
                 min_food_conc: int = 5,
                 time_limit: int = 1.0,
                 V: float = 1.0):
        self.crn = crn
        self.min_food_conc = min_food_conc
        self.time_limit = time_limit
        self.V = V

    def sample(self, seed=None, snapshot_times = (1.0, )) -> str:
        """
        Simulates chemical reaction network dynamics and records species traces over time.

        The function performs a stochastic simulation of a chemical reaction
        network (CRN), iterating through reactions to determine the state of
        species in the network over time. It uses a propensity-based sampling
        approach to determine the time evolution and outcomes of reactions.

        Attributes:
            None

        Parameters:
            seed (int, optional): Random seed for initializing the simulation to
                ensure reproducibility.
            snapshot_times (tuple[float], optional): Time points at which the
                simulation outputs species concentrations. Defaults to (1.0,).

        Returns:
            str: JSON formatted string containing the simulation traces over
                specified snapshot times. Each species' concentration is recorded
                over the course of the simulation.

        Raises:
            None
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

        while t < self.time_limit:
            props = []
            actions = []  # tuples: ('lig'/'cle', rx_index)

            for rid in reactions:
                rx = reactions[rid]
                reactants = list(rx['reactants'])
                products = list(rx['products'])
                n_cat = sum(counts.get(c, 0) for c in catalysis[rid])
                effective_rate = rates[rid]['k_unlig'] + rates[rid]['k_lig'] * n_cat

                prop_f = self._get_propensity(counts, reactants, effective_rate)
                props.append(prop_f)
                actions.append(('lig', rid))

                prop_b = self._get_propensity(counts, products, effective_rate)
                props.append(prop_b)
                actions.append(('cle', rid))

            total_prop = sum(props)
            if total_prop <= 0:
                break

            # time increment and choice
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

            if typ == 'lig':
                for r in rx['reactants']:
                    counts[r] -= 1
                    if counts[r] < 0: counts[r] = 0
                for p in rx['products']:
                    counts[p] = counts.get(p, 0) + 1
            else:
                for p in rx['products']:
                    counts[p] -= 1
                    if counts[p] < 0: counts[p] = 0
                for r in rx['reactants']:
                    counts[r] = counts.get(r, 0) + 1

            # replenish food
            for f in food:
                if counts.get(f, 0) < self.min_food_conc:
                    counts[f] = self.min_food_conc

            events += 1
            if t >= snapshot_times[st]:

                times.append(t)
                for s in species:
                    traces[s].append(counts.get(s, 0))
                st += 1

                if st >= len(snapshot_times): #finish if we sampled for all needed times
                    break

        # return events
        return json.dumps(traces)

    def _get_propensity(self, counts, items, effective_rate):
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
                ff *= (counts[s] - i)
            contrib *= ff
        order = len(items)
        div = self.V ** (order - 1) if order > 1 else 1.0
        return effective_rate * contrib / div