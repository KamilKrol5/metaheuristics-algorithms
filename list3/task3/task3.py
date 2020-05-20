import sys
from agent_walk_with_genetic import AgentWalkWithGenetic


if __name__ == '__main__':
    agent_walk: AgentWalkWithGenetic = AgentWalkWithGenetic.from_stdin()
    child_making_attempts_count = 20
    res = agent_walk.run_genetic_algorithm(
        mutation_probability=0.4
    )
    print(''.join(res), res.cost, file=sys.stderr)
    print(res.cost)
