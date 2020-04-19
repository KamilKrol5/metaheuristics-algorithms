import fileinput
import sys
import time
import random
import numpy as np
from copy import deepcopy
from typing import Tuple, List
from matplotlib import pyplot as plt


class _Block:
    def __init__(self, x_start, y_start, x_length, y_length, value_inside):
        self.x_start = x_start
        self.y_start = y_start
        self.x_length = x_length
        self.y_length = y_length
        self.value_inside = value_inside

    def __hash__(self):
        return hash((self.x_start, self.y_start))

    def __str__(self):
        return f'<(x,y) = ({self.x_start},{self.y_start}), {self.x_length} x {self.y_length}>'

    def __repr__(self):
        return str(self)

    def contains(self, x, y):
        return self.x_start <= x < self.x_start + self.x_length and \
               self.y_start <= y < self.y_start + self.y_length


class _BlockInSpace(_Block):
    DIRECTIONS = {
        'U': (0, 1),
        'R': (1, 0),
        'D': (0, -1),
        'L': (-1, 0),
    }

    def __init__(self, x_start, y_start, x_length, y_length, value_inside, space_of_blocks):
        super().__init__(x_start, y_start, x_length, y_length, value_inside)
        self.space_of_blocks = space_of_blocks

    def is_space_valid(self, x, y):
        arr = np.full((x, y), 1, dtype=int)
        for block_ in self.space_of_blocks:
            end_x_index = block_.x_start + block_.x_length
            end_y_index = block_.y_start + block_.y_length
            arr[
                block_.x_start:end_x_index,
                block_.y_start:end_y_index
            ] = 0

        # print(f'Control sum = {np.sum(arr)}')
        return np.sum(arr)

    def neighbours_in_direction(self, direction):
        neighbours = set()
        if direction[0] == 0:  # U, D
            for x in range(self.x_start, self.x_start + self.x_length, 1):
                y = self.y_start
                if direction[1] == -1:
                    y = y + self.y_length
                for b in self.space_of_blocks:
                    if b.contains(x, y - direction[1]):
                        neighbours.add(b)
                        break

        elif direction[1] == 0:  # L, R
            for y in range(self.y_start, self.y_start + self.y_length, 1):
                x = self.x_start
                if direction[0] == 1:
                    x = x + self.x_length
                for b in self.space_of_blocks:
                    if b.contains(x + direction[0], y):
                        neighbours.add(b)
                        break

        return neighbours

    def get_neighbours(self):
        neighbours = set()
        for d in self.DIRECTIONS.values():
            new_neighbours = self.neighbours_in_direction(d)
            neighbours.update(new_neighbours)
        return neighbours

    def can_expand_in_direction(self, direction, min_block_x_size, min_block_y_size):
        neighbours_in_direction = self.neighbours_in_direction(direction)
        if len(neighbours_in_direction) != 1:
            return False, None
        neighbour = neighbours_in_direction.pop()
        if abs(direction[1]) == 0:
            return min_block_x_size < neighbour.x_length and neighbour.y_length == self.y_length, neighbour
        elif abs(direction[0] == 0):
            return min_block_y_size < neighbour.y_length and neighbour.x_length == self.x_length, neighbour

    """ Returns neighbour in the given direction which can be merged with self-block.
        If there is no possible candidate for merge, the method returns None.
    """

    def can_merge_in_direction(self, direction):
        neighbours_on_direction = self.neighbours_in_direction(direction)
        if len(neighbours_on_direction) != 1:
            return None
        neighbour = neighbours_on_direction.pop()
        if neighbour.value_inside != self.value_inside:
            return None
        if abs(direction[1]) == 0 and neighbour.y_length != self.y_length:
            return None
        elif abs(direction[0] == 0) and neighbour.x_length != self.x_length:
            return None
        return neighbour

    """ Merges given 'other' block with self-block if possible. 
        Returns bool value telling if merge was performed. Modifies 'other_blocks' collection.
    """

    def merge_in_direction(self, direction):
        other = self.can_merge_in_direction(direction)
        if other is None:
            return False
        up_or_down, left_or_right = direction[0] == 0, direction[1] == 0

        # specific situations and error handling
        if up_or_down and left_or_right:
            raise ValueError('Merging in two directions at once is not supported')
        print('BEFORE')
        print([str(bl) for bl in self.space_of_blocks])
        # merging itself
        if up_or_down:
            if direction[1] == -1:  # down
                self.y_length += other.y_length
            elif direction[1] == 1:  # up
                self.y_length += other.y_length
                self.x_start = other.x_start
                self.y_start = other.y_start
        elif left_or_right:
            if direction[0] == -1:  # left
                self.x_length += other.x_length
                self.x_start = other.x_start
                self.y_start = other.y_start
            elif direction[0] == 1:  # right
                self.x_length += other.x_length
        else:
            print(f'Warning: Cannot merge in (0,0) direction', file=sys.stderr)
            return False

        self.space_of_blocks.remove(other)
        print('AFTER')
        print([str(bl) for bl in self.space_of_blocks])
        return True

    """ Extends self-block and reduces the neighbour size by given value.
        There is an assumption that block provided as neighbour, actually is a neighbour in the provided direction.
        
        Args:
            neighbour (_BlockInSpace): neighbour (on provided direction side) of block to be extended.
            It is not checked if block provided as neighbour actually is the neighbour.
            direction (Tuple[int, int]): The direction in which extension is to be performed.
            difference (int): the increase of self-block size. Simultaneously it is the decrease of neighbour.
            
        Returns:
            bool: True if extension was performed successfully, False otherwise.
    """
    def expand_towards_neighbour(self, neighbour: '_BlockInSpace', direction, difference):
        if direction not in self.DIRECTIONS.values():
            print(f'Warning: Extension in the direction not present in DIRECTIONS dictionary is not supported.',
                  file=sys.stderr)
            return False
        if self == neighbour:
            print(f'Cannot merge with itself', file=sys.stderr)
            return False
        if direction[0] == 0:  # U or D
            self.y_length += difference
            neighbour.y_length -= difference
            if direction[1] == -1:  # D
                neighbour.y_start += difference
            elif direction[1] == 1:  # U
                self.y_start -= difference
        elif direction[1] == 0:  # L or R
            self.x_length += difference
            neighbour.x_length -= difference
            if direction[0] == -1:  # L
                self.x_start -= difference
            elif direction[0] == 1:  # R
                neighbour.x_start += difference
        return True


class _Solution:
    def __init__(self, image_matrix, blocks: List[_BlockInSpace], x_free, y_free):
        self.x_free = x_free
        self.y_free = y_free
        self.blocks: List[_BlockInSpace] = blocks
        self.matrix = image_matrix

    def validate(self):
        x, y = self.matrix.shape
        sum_ = np.sum([bl.is_space_valid(x, y) for bl in self.blocks])
        print(f'Control sum = {sum_}')
        return sum_ == 0


class ImageApproximationInstance:

    def __init__(self, matrix, k_coefficient, max_time, target_color_values: List[int]):
        self.matrix = matrix
        self.rows, self.columns = matrix.shape
        # print(self.rows, self.columns)
        self.k = k_coefficient
        self.max_time = max_time
        self.target_color_values: List[int] = target_color_values
        self.middle_points_between_target_values = self._initialize_target_colors()

    def _initialize_target_colors(self):
        return [
            np.mean(pair)
            for pair in zip(self.target_color_values, self.target_color_values[1:])]

    def _convert_value_to_closest_target_value(self, value):
        for i, t_val in enumerate(self.middle_points_between_target_values):
            if value < t_val:
                return self.target_color_values[i]
        return self.target_color_values[-1]

    """ Destroys provided solution - it does not copy the matrix nor the blocks. 
    """

    def _get_random_neighbour(self, solution: _Solution) -> _Solution:
        block: _BlockInSpace = np.random.choice(solution.blocks)
        block.value_inside = np.random.choice(self.target_color_values)
        end_row = block.x_start + block.x_length
        end_column = block.y_start + block.y_length
        solution.matrix[block.x_start:end_row, block.y_start:end_column] = block.value_inside

        # merging if possible
        block: _BlockInSpace = np.random.choice(solution.blocks)
        direction = random.choice(list(block.DIRECTIONS.values()))
        block.merge_in_direction(direction)

        # extend if possible
        block_to_resize: _BlockInSpace = np.random.choice(solution.blocks)
        for direction in block_to_resize.DIRECTIONS.values():
            can_extend, candidate = block_to_resize.can_expand_in_direction(direction, self.k, self.k)
            if can_extend:
                result = block_to_resize.expand_towards_neighbour(candidate, direction, 1)
                if result:
                    resized_end_x_index = block_to_resize.x_start + block_to_resize.x_length
                    resized_end_y_index = block_to_resize.y_start + block_to_resize.y_length
                    candidate_end_x_index = candidate.x_start + candidate.x_length
                    candidate_end_y_index = candidate.y_start + candidate.y_length
                    solution.matrix[
                        block_to_resize.x_start:resized_end_x_index,
                        block_to_resize.y_start:resized_end_y_index
                    ] = block_to_resize.value_inside
                    solution.matrix[
                        candidate.x_start:candidate_end_x_index,
                        candidate.y_start:candidate_end_y_index
                    ] = candidate.value_inside
                    # print([str(bl) for bl in block_to_resize.space_of_blocks])
                    # self.visualise_matrix(solution.matrix)
        return solution

    @classmethod
    def from_file_input(cls, target_color_values):
        matrix, max_time, k = cls._read_input()
        return cls(matrix, k, max_time, target_color_values)

    @staticmethod
    def _read_input() -> Tuple[np.ndarray, int, int]:
        with fileinput.input() as file:
            first_line = file.readline()
            [max_time, rows, columns, k] = [int(x) for x in first_line.split()]
            matrix = np.ndarray((rows, columns), dtype='int')

            for i, line in enumerate(file, 0):
                for j, x in enumerate(line.split(), 0):
                    matrix[i][j] = int(x)
        return matrix, max_time, k

    @staticmethod
    def _compute_mse(matrix1, matrix2):
        if matrix1.shape != matrix2.shape:
            raise ValueError(f'Matrices does not have the same shape')
        n, m = matrix1.shape
        return np.sum((matrix1 - matrix2) ** 2) / (n * m)

    def _compute_mse_of_image_and_other(self, other_matrix):
        return self._compute_mse(self.matrix, other_matrix)

    @staticmethod
    def _probability(delta_f, temperature, c):
        if delta_f <= 0:
            return 1
        # return 1.0 / (1.0 + np.power(np.e, c * delta_f / temperature))
        return np.power(np.e, -delta_f / temperature)

    """ Generates initial solution from matrix.
        It is created as matrix of K x K blocks filled with mean color in original matrix approximated
        to the closest target color (there is a mean value of colors computed in each block and then 
        that value is used to decide which color from target colors should be taken as color for entire block)
        If matrix size is not divisible by K, then blocks at the end (right and bottom) are extended to cover
        the entire matrix.
    """

    def _generate_initial_solution(self):
        working_matrix = np.copy(self.matrix)
        blocks_ = []
        k = self.k
        for i in range(0, self.rows - k + 1, k):
            row_end_range = i + k
            if i + 2 * k > self.rows:
                row_end_range = i + k + (self.rows % k)
            for j in range(0, self.columns - k + 1, k):
                column_end_range = j + k
                if j + 2 * k > self.columns:
                    column_end_range = j + k + (self.columns % k)
                mean_value_in_submatrix = np.mean(working_matrix[i:row_end_range, j:column_end_range])
                working_matrix[i:row_end_range, j:column_end_range] = \
                    self._convert_value_to_closest_target_value(mean_value_in_submatrix)
                blocks_.append(
                    _BlockInSpace(i, j, row_end_range - i, column_end_range - j, mean_value_in_submatrix, blocks_))
        return _Solution(working_matrix, blocks_, self.rows % k, self.columns % k)

    def simulated_annealing(self,
                            initial_temperature,
                            red_factor,
                            c=-1):
        end_time = time.time() + self.max_time
        temperature = initial_temperature

        current_solution: _Solution = self._generate_initial_solution()
        working_solution: _Solution = deepcopy(current_solution)
        self.visualise_matrix(current_solution.matrix)

        while temperature > red_factor and time.time() < end_time:

            neighbour: _Solution = self._get_random_neighbour(working_solution)
            # self.visualise_matrix(neighbour.matrix)
            neighbour.validate()

            delta_f = \
                self._compute_mse_of_image_and_other(neighbour.matrix) - self._compute_mse_of_image_and_other(
                    current_solution.matrix)
            print('delta = ', delta_f, f'probability = {self._probability(delta_f, temperature, c)}')

            if self._probability(delta_f, temperature, c) > np.random.rand():
                # self.visualise_matrix(neighbour.matrix)
                current_solution = deepcopy(neighbour)
            temperature = temperature * (1 - red_factor)

        return current_solution, self._compute_mse_of_image_and_other(current_solution.matrix)

    @staticmethod
    def visualise_matrix(matrix: np.ndarray):
        plt.imshow(matrix)
        plt.gray()
        plt.show()

    def visualise(self):
        self.visualise_matrix(self.matrix)


if __name__ == '__main__':
    problemInstance = ImageApproximationInstance.from_file_input([0, 32, 64, 128, 160, 192, 223, 255])
    init_ = problemInstance._generate_initial_solution()
    print(init_.matrix.shape)
    print(init_.blocks)
    # problemInstance.visualise_matrix(init_.matrix)
    # problemInstance.visualise_matrix(problemInstance._get_random_neighbour(init_).matrix)
    # problemInstance.visualise_matrix(problemInstance.matrix)
    print('MSE for initial solution = ', problemInstance._compute_mse_of_image_and_other(init_.matrix))

    sol, val = problemInstance.simulated_annealing(1500, 0.005)
    problemInstance.visualise_matrix(sol.matrix)
    print(f'Solution value = {val}')
    # problemInstance.visualise()
    # print(problemInstance.compute_mse(np.array([[1, 1, 1],
    #                                       [2, 2, 2]]), np.array([[0, 0, 0],
    #                                                              [3, 4, 5]])))
