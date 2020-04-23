import fileinput
import sys
import time
import random
import numpy as np
from copy import deepcopy
from typing import Tuple, List
from matplotlib import pyplot as plt
from blocks import _BlockInSpace


class _Solution:
    def __init__(self, image_matrix, blocks: List[_BlockInSpace], x_free, y_free):
        self.x_free = x_free
        self.y_free = y_free
        self.blocks: List[_BlockInSpace] = blocks
        self.matrix = image_matrix

    def validate(self):
        x, y = self.matrix.shape
        sum_ = np.sum([bl.is_space_valid(x, y) for bl in self.blocks])
        # print(f'Control sum = {sum_}')
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
        free_x_present, free_y_present = solution.x_free > 0, solution.y_free > 0
        if free_x_present or free_y_present:
            form_change_weight = 6
        else:
            form_change_weight = 0

        choices = [
            *(form_change_weight * ['resize']),
            'color',
            (form_change_weight // 3 * ['merge']),
            (form_change_weight // 3 * ['split']),
        ]
        while True:
            choice = np.random.choice(choices)

            if choice == 'color':
                block: _BlockInSpace = np.random.choice(solution.blocks)
                block.value_inside = np.random.choice(self.target_color_values)
                end_row = block.x_start + block.x_length
                end_column = block.y_start + block.y_length
                solution.matrix[block.x_start:end_row, block.y_start:end_column] = block.value_inside
                break

            if choice == 'merge':
                # merging if possible
                block: _BlockInSpace = np.random.choice(solution.blocks)
                direction = random.choice(list(block.DIRECTIONS.values()))
                if block.merge_in_direction(direction):
                    break

            if choice == 'resize':
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
                            break

            if choice == 'split':
                # splitting if possible
                block_to_split: _BlockInSpace = np.random.choice(solution.blocks)
                if block_to_split.split_in_two(np.random.choice(['x', 'y']), self.k, self.k):
                    break
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
        return _Solution(working_matrix, blocks_, self.columns % k, self.rows % k)
        # color = self._convert_value_to_closest_target_value(np.mean(working_matrix))
        # working_matrix[0:, 0:] = color
        # blocks_.append(_BlockInSpace(0, 0, self.rows, self.columns, color, blocks_))
        # return _Solution(working_matrix, blocks_, 0, 0)

    def simulated_annealing(self,
                            initial_temperature,
                            red_factor,
                            c=-1):
        end_time = time.time() + self.max_time
        temperature = initial_temperature

        current_solution: _Solution = self._generate_initial_solution()
        working_solution: _Solution = deepcopy(current_solution)
        best_ever_found = deepcopy(current_solution)
        best_ever_found_cost = self._compute_mse_of_image_and_other(best_ever_found.matrix)
        # self.visualise_matrix(current_solution.matrix)

        while temperature > red_factor and time.time() < end_time:

            neighbour: _Solution = self._get_random_neighbour(working_solution)
            # self.visualise_matrix(neighbour.matrix)
            # neighbour.validate()

            mse_of_neighbour = self._compute_mse_of_image_and_other(neighbour.matrix)
            delta_f = mse_of_neighbour - self._compute_mse_of_image_and_other(current_solution.matrix)
            # print('delta = ', delta_f, f'probability = {self._probability(delta_f, temperature, c)}')

            if self._probability(delta_f, temperature, c) > np.random.rand():
                # self.visualise_matrix(neighbour.matrix)
                current_solution = deepcopy(neighbour)
                if mse_of_neighbour < best_ever_found_cost:
                    best_ever_found_cost = mse_of_neighbour
                    best_ever_found = deepcopy(neighbour)
            temperature = temperature * (1 - red_factor)

        return best_ever_found, best_ever_found_cost

    @staticmethod
    def visualise_matrix(matrix: np.ndarray):
        plt.imshow(matrix)
        plt.gray()
        plt.show()

    def visualise(self):
        self.visualise_matrix(self.matrix)


if __name__ == '__main__':
    problemInstance = ImageApproximationInstance.from_file_input([0, 32, 64, 128, 160, 192, 223, 255])

    sol, val = problemInstance.simulated_annealing(2000, 0.005)
    # problemInstance.visualise_matrix(sol.matrix)
    print(val)
    print('\n'.join([' '.join([str(num) for num in row]) for row in sol.matrix]), file=sys.stderr)
    # print(f'Solution value = {val}')
