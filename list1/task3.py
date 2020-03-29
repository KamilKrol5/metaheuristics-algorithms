import fileinput
from typing import Tuple
import numpy as np


class AgentWalk:
    WALKABLE = 0
    WALL = 1
    START = 5
    EXIT = 8

    def __init__(self, board: np.ndarray, max_time: int):
        self.max_time = max_time
        self.board = board

    @classmethod
    def from_file(cls, filename: str):
        board_, max_time = AgentWalk.read_input(filename)
        return cls(board_, max_time)

    @classmethod
    def from_stdin(cls):
        cities, max_time = AgentWalk.read_input()
        return cls(cities, max_time)

    @staticmethod
    def read_input(filename=None) -> Tuple[np.ndarray, int]:
        with open(filename, 'r') if filename is not None else fileinput.input() as file:
            first_line = file.readline()
            print([str(x) for x in first_line.split()])
            [max_time, rows, columns] = [int(x) for x in first_line.split()]
            board = np.ndarray((rows, columns))
            for i, line in enumerate(file, 0):
                for j, x in enumerate(str.rstrip(line)):
                    board[i][j] = int(x)

        return board, max_time

    def print_board_matrix(self):
        return '\n'.join([f'{row}' for row in self.board])


class AgentWalkWithTabuSearch(AgentWalk):
    pass


if __name__ == '__main__':
    sw = AgentWalk.from_file('l1z3a.txt')
    print(sw.print_board_matrix())
