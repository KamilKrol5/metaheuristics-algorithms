import sys
from agent_walk_with_sa import AgentWalkWithSA


if __name__ == '__main__':
    agent_walk: AgentWalkWithSA = AgentWalkWithSA.from_stdin()
    res, res_cost = agent_walk.simulated_annealing(
        initial_temperature=1000,
        red_factor=0.0005,
        c=-1
    )
    print(''.join(res), file=sys.stderr)
    print(res.cost)
