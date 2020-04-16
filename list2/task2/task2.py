import fileinput
import time
from copy import deepcopy
from dataclasses import make_dataclass
from typing import Tuple, List

import numpy as np
from matplotlib import pyplot as plt

_Block = make_dataclass('Block', ['x_start', 'y_start', 'x_length', 'y_length', 'value_inside'])


class _Solution:
    def __init__(self, image_matrix, blocks: List[_Block]):
        self.blocks: List[_Block] = blocks
        self.matrix = image_matrix


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

    """ Destroys provided solution - it does not copy the matrix not the blocks. 
    """
    def _get_random_neighbour(self, solution: _Solution) -> _Solution:
        for block in solution.blocks:
            block.value_inside = np.random.choice(self.target_color_values)
            end_row = block.x_start + block.x_length
            end_column = block.y_start + block.y_length
            solution.matrix[block.x_start:end_row, block.y_start:end_column] = block.value_inside
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
        return 1.0 / (1.0 + np.power(np.e, c * delta_f / temperature))

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
                blocks_.append(_Block(i, j, row_end_range - i, column_end_range - j, mean_value_in_submatrix))
        return _Solution(working_matrix, blocks_)

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
            self.visualise_matrix(neighbour.matrix)

            delta_f = \
                self._compute_mse_of_image_and_other(neighbour.matrix) - self._compute_mse_of_image_and_other(
                    current_solution.matrix)
            print('delta = ', delta_f)

            if self._probability(delta_f, temperature, c) > np.random.rand():
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

    sol, val = problemInstance.simulated_annealing(1000, 0.05)
    problemInstance.visualise_matrix(sol.matrix)
    print(f'Solution value = {val}')
    # problemInstance.visualise()
    # print(problemInstance.compute_mse(np.array([[1, 1, 1],
    #                                       [2, 2, 2]]), np.array([[0, 0, 0],
    #                                                              [3, 4, 5]])))
