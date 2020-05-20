import fileinput
import sys
from collections import namedtuple
from typing import Tuple, List
import numpy as np

from path import Path

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


class Agent:
    class WallException(Exception):
        pass

    def __init__(self, board: np.ndarray, marking=False):
        self.marking = marking
        self.board = board.copy()
        agent_position = np.where(board == AGENT)
        self.current_position: Position = Position(int(agent_position[0]), int(agent_position[1]))

    def look(self, direction):
        row_action, column_action = direction_actions[direction]
        return self.board[self.current_position.row + row_action][self.current_position.column + column_action]

    def move(self, direction, change_own_board=True):
        destination: int = self.look(direction)
        if destination == WALKABLE or (not change_own_board and destination == AGENT):
            self.__make_move(direction, change_own_board)
        elif destination == WALL:
            raise Agent.WallException()
        elif destination == EXIT:
            self.__update_current_position(direction)

    def __make_move(self, direction, change_own_board=True):
        row_action, column_action = direction_actions[direction]
        if change_own_board:
            self.board[self.current_position.row][self.current_position.column] = MARK if self.marking else WALKABLE
            self.board[self.current_position.row + row_action][self.current_position.column + column_action] = AGENT
        self.current_position: Position = \
            Position(self.current_position.row + row_action, self.current_position.column + column_action)

    def __update_current_position(self, direction):
        row_action, column_action = direction_actions[direction]
        self.current_position = \
            Position(self.current_position.row + row_action, self.current_position.column + column_action)


class AgentWalk:
    def __init__(self, board: np.ndarray, initial_solutions: List[Path], max_time: int, max_population_size: int):
        self.max_population_size = max_population_size
        self.initial_solutions = initial_solutions
        self.max_time = max_time
        self.board = board
        self.population: List[Path] = []

    @classmethod
    def from_file(cls, filename: str):
        board_, init_sol, max_time,  max_population_size = AgentWalk.read_input(filename)
        return cls(board_, init_sol, max_time, max_population_size)

    @classmethod
    def from_stdin(cls):
        cities, init_sol, max_time, max_population_size = AgentWalk.read_input()
        return cls(cities, init_sol, max_time, max_population_size)

    @staticmethod
    def read_input(filename=None) -> Tuple[np.ndarray, List[Path], int, int]:
        with open(filename, 'r') if filename is not None else fileinput.input() as file:
            first_line = file.readline()
            # print([str(x) for x in first_line.split()])
            [max_time, rows, columns, initial_solutions_count, max_population_size] = \
                [int(x) for x in first_line.split()]
            board = np.ndarray((rows, columns))
            initial_solutions: List[Path] = []
            for i, line in enumerate(file, 0):
                if i < rows:
                    for j, x in enumerate(str.rstrip(line)):
                        board[i][j] = int(x)
                elif i - rows + 1 < max_population_size:
                    initial_solutions.append(Path(str.rstrip(line)))
                    print(initial_solutions)
                else:
                    print(f'Too much lines in input: {str.rstrip(line)}', file=sys.stderr)
        return board, initial_solutions, max_time, max_population_size

    def print_board_matrix(self):
        return '\n'.join([f'{row}' for row in self.board])
