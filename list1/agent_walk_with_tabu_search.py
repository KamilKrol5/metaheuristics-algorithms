import time
from collections import deque
from copy import copy
from typing import Tuple, List
import numpy as np

from agent_walk import Agent
from path import Path
from agent_walk import AgentWalk
from task3 import directions, WALKABLE, EXIT


class AgentWalkWithTabuSearch(AgentWalk):
    def __init__(self, board: np.ndarray, max_time: int, inversion_count_for_neighbour=2):
        super().__init__(board, max_time)
        self.inversion_count_for_neighbour = inversion_count_for_neighbour

    @staticmethod
    def generate_neighbours(path: Path, neighbour_count=1, random_inversion_count=1) -> List[Path]:
        neighbours: List[Path] = []
        path_length = len(path)
        for _ in range(neighbour_count):
            neighbour: Path = copy(path)
            randoms = np.random.randint(0, path_length, 2 * path_length)
            for j in range(random_inversion_count):
                neighbour.make_inversion(randoms[2*j], randoms[2*j+1])
            neighbours.append(neighbour)
        return neighbours

    def validate_and_shorten_path(self, path: Path):
        walker = Agent(self.board)
        for i, direction in enumerate(path, 1):
            try:
                walker.move(direction, change_own_board=False)
                if self.board[walker.current_position.row][walker.current_position.column] == EXIT:
                    return Path(path[:i]), True
            except Agent.WallException:
                return None, False
        return None, False

    def generate_acceptable_solution(self) -> Path:
        agent = Agent(self.board)
        path: Path = Path()
        for i in range(-3, 3):
            side = directions[i - 1]
            front = directions[i]
            while True:
                if agent.look(side) == EXIT:
                    agent.move(side)
                    path.append(side)
                    return path
                elif agent.look(front) == WALKABLE:
                    agent.move(front)
                    path.append(front)
                else:
                    break
        return path

    def tabu_search(self, tabu_max_size=10, max_iterations=100,
                    neighbours_count=1000) -> Tuple[Path, int]:
        end_time = time.time() + self.max_time
        # print(f'max time = {self.max_time}')
        tabu = deque(maxlen=tabu_max_size)

        current_solution = self.generate_acceptable_solution()
        current_solution_cost = current_solution.cost

        for iteration in range(max_iterations):
            # print(end_time - time.time())
            if time.time() >= end_time:
                return current_solution, iteration - 1

            neighbours = self.generate_neighbours(current_solution, neighbours_count,
                                                  random_inversion_count=current_solution_cost)
            neighbours = [n for n in neighbours if n not in tabu]

            # if iteration % 25 == 0:
            #     print(f'Iteration {iteration}\nx = {current_solution}\ncost = {current_solution.cost}',
            #     file=sys.stderr)
            #     print(f'tabu size = {len(tabu)}', file=sys.stderr)

            for neighbour in neighbours:
                neighbour, is_valid = self.validate_and_shorten_path(neighbour)
                if is_valid and neighbour.cost < current_solution_cost:
                    current_solution = neighbour
                    current_solution_cost = neighbour.cost
                tabu.append(neighbour)

            # tabu.append(current_solution)

        return current_solution, max_iterations
