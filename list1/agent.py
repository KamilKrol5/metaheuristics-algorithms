import numpy as np
from task3 import Position, direction_actions, EXIT, WALL, WALKABLE, AGENT, MARK


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
        if destination == WALKABLE:
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
