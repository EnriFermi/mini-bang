
from simulators.raf.macro.simulator import MasterModel
import json
def sample_trajectory(payload) -> str:
    try:
        request = payload
        T = int(request['T'])
        N = int(request['N'])
    

        if T > 100:
            return "T cannot be greater that 100"
        if N > 12:
            return "N cannot be greater than 12"
        
        simulator = MasterModel(
            M0=2,
            alpha=lambda i: 0.05*(3-i),
            K = 3,
            p = 0.5,
            k_lig=1.0, 
            k_unlig=0.05,
            max_events=10000
        ).get_micro_simulator(T, max_raf=True, prune_catalysts=True)

        last_states = []
        for n in range(N):
            last_state = simulator.sample()
            last_states.append(last_state)
        results = json.dumps({f"N={i+1}": str(state) for i, state in enumerate(last_states)})
        return results
    except Exception as e:
        return str(e)
    