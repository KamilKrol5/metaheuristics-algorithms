import sys
import time
from copy import deepcopy, copy
from agent_walk import AgentWalk, Agent, EXIT, directions, Position
import numpy as np
from path import Path


class AgentWalkWithSA(AgentWalk):
    def __init__(self, board: np.ndarray, max_time: int):
        super().__init__(board, max_time)
        self.rows, self.columns = board.shape

    # """ Generates random neighbour of provided path. Given path object is modified inside this method.
    # """
    def get_random_neighbour(self, path: Path, random_inversion_count=1) -> Path:
        walker = Agent(self.board)
        walker_start_pos: Position = walker.current_position
        while True:
            walker.current_position = walker_start_pos
            path_copy = copy(path)
            for j in range(random_inversion_count):
                path_copy.make_inversion(*np.random.random_integers(0, len(path_copy)-1, 2))
            new_path, is_valid = self.validate_and_shorten_path(path_copy, agent=walker)
            if is_valid:
                return new_path

    def validate_and_shorten_path(self, path: Path, agent=None,skrr=False):
        if agent is None:
            walker = Agent(self.board)
        else:
            walker = agent
        for i, direction in enumerate(path, 1):
            try:
                walker.move(direction, change_own_board=False,d=skrr)
                if self.board[walker.current_position.row, walker.current_position.column] == EXIT:
                    return Path(path[:i]), True

            except Agent.WallException:
                return None, False

        return None, False

    @staticmethod
    def probability(delta_f, temperature, c):
        if delta_f <= 0:
            return 1
        return np.power(np.e, -delta_f / temperature)

    # this way for generating initial solution is advised by the lecturer
    def generate_initial_solution(self) -> Path:
        walker = Agent(self.board)
        walker_start_pos: Position = walker.current_position

        while True:
            random_path = Path(np.random.choice(directions, size=self.rows * self.columns))
            walker.current_position = walker_start_pos
            path_, is_valid = self.validate_and_shorten_path(random_path, agent=walker, skrr=True)

            if is_valid:
                return path_

    def simulated_annealing(self,
                            initial_temperature,
                            red_factor,
                            c=-1):
        end_time = time.time() + self.max_time

        temperature = initial_temperature
        current_solution: Path = self.generate_initial_solution()
        working_solution: Path = deepcopy(current_solution)
        best_ever_found = deepcopy(current_solution)

        print('rows:', self.rows, 'columns:', self.columns, file=sys.stderr)
        print(f"initial: {''.join(current_solution)}, cost = {current_solution.cost}", file=sys.stderr)

        while temperature > red_factor and time.time() < end_time:

            neighbour: Path = self.get_random_neighbour(working_solution, 10)

            delta_f = neighbour.cost - current_solution.cost
            # print('delta = ', delta_f, f'probability = {self.probability(delta_f, temperature, c)}')

            if self.probability(delta_f, temperature, c) > np.random.rand():

                current_solution = deepcopy(neighbour)
                if neighbour.cost < best_ever_found.cost:
                    best_ever_found = deepcopy(neighbour)

            temperature = temperature * (1 - red_factor)

        print(f'Time left: {end_time - time.time()}', file=sys.stderr)
        return best_ever_found, best_ever_found.cost
