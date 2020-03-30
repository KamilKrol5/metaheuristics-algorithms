import sys
from collections import namedtuple
import numpy as np
from agent_walk_with_tabu_search import AgentWalkWithTabuSearch

WALKABLE = 0
WALL = 1
AGENT = 5
EXIT = 8
MARK = 2

direction_actions = {
    'L': (0, -1),  # LEFT
    'U': (-1, 0),  # UP
    'R': (0, 1),  # RIGHT
    'D': (1, 0),  # DOWN
}

directions = ['L', 'U', 'R', 'D']

Position = namedtuple('Position', ['row', 'column'])


if __name__ == '__main__':
    # sw = AgentWalkWithTabuSearch.from_file('l1z3b.txt')
    sw = AgentWalkWithTabuSearch.from_stdin()
    res, iterations = sw.tabu_search(
        tabu_max_size=int(np.sqrt(sw.board.shape[0] * sw.board.shape[1])),
        max_iterations=(sw.board.shape[0] + sw.board.shape[1])**2,
        neighbours_count=int(np.sqrt(sw.board.shape[0] * sw.board.shape[1]))
    )
    print(''.join(res), file=sys.stderr)
    print(res.cost)

    # agent = Agent(sw.board, marking=True)
    # for step in res:
    #     agent.move(step)
    # np.set_printoptions(linewidth=500)
    # print(agent.board.view())
