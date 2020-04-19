import fileinput
import sys
import time
import numpy as np
from copy import deepcopy
from typing import Tuple, List, Set
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

    def get_neighbours(self, max_number_of_neighbours_in_one_direction=np.inf):
        neighbours = set()
        for d in self.DIRECTIONS.values():
            new_neighbours = self.neighbours_in_direction(d)
            if len(new_neighbours) <= max_number_of_neighbours_in_one_direction:
                neighbours.update(new_neighbours)
        return neighbours

    def can_expand_in_direction(self, direction, min_block_x_size, min_block_y_size):
        neighbours_in_direction = self.neighbours_in_direction(direction)
        if len(neighbours_in_direction) != 1:
            return False
        neighbour = neighbours_in_direction.pop()
        return neighbour.y_length > min_block_y_size * abs(direction[1]) and \
            neighbour.x_length > min_block_x_size * abs(direction[0])

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
            raise ValueError('Merging in both directions at once is not supported')

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
        return True

    # def merge_with_same_value_blocks(self, blocks: Set['_Block']):
    #     cannot_merge_more = False
    #     while not cannot_merge_more:
    #         cannot_merge_more = True
    #         for d in self.DIRECTIONS:


class _Solution:
    def __init__(self, image_matrix, blocks: List[_Block], x_free, y_free):
        self.x_free = x_free
        self.y_free = y_free
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
        can_resize_x = solution.x_free != 0
        can_resize_y = solution.y_free != 0
        resize_x, resize_y = 0, 0
        if can_resize_x:
            resize_x = np.random.choice([0, 1])
        if can_resize_y:
            resize_y = np.random.choice([0, 1])

        # for block in solution.blocks:
        #     block.value_inside = np.random.choice(self.target_color_values)
        #     end_row = block.x_start + block.x_length
        #     end_column = block.y_start + block.y_length
        #     solution.matrix[block.x_start:end_row, block.y_start:end_column] = block.value_inside
        # return solution
        block = np.random.choice(solution.blocks)
        block.value_inside = np.random.choice(self.target_color_values)
        end_row = block.x_start + block.x_length
        end_column = block.y_start + block.y_length
        solution.matrix[block.x_start:end_row, block.y_start:end_column] = block.value_inside

        # resising
        block_to_resize: _Block = np.random.choice(solution.blocks)
        neighbours = [b for b in solution.blocks if b]
        # block_to_resize.
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
        return _Solution(working_matrix, blocks_, self.rows % k, self.columns % k)

    def simulated_annealing(self,
                            initial_temperature,
                            red_factor,
                            c=1):
        end_time = time.time() + self.max_time
        temperature = initial_temperature

        current_solution: _Solution = self._generate_initial_solution()
        working_solution: _Solution = deepcopy(current_solution)
        self.visualise_matrix(current_solution.matrix)

        while temperature > red_factor and time.time() < end_time:

            neighbour: _Solution = self._get_random_neighbour(working_solution)
            # self.visualise_matrix(neighbour.matrix)

            delta_f = \
                self._compute_mse_of_image_and_other(neighbour.matrix) - self._compute_mse_of_image_and_other(
                    current_solution.matrix)
            print('delta = ', delta_f, f'probability = {self._probability(delta_f, temperature, c)}')

            if self._probability(delta_f, temperature, c) > np.random.rand():
                self.visualise_matrix(neighbour.matrix)
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

    sol, val = problemInstance.simulated_annealing(100, 0.0005)
    problemInstance.visualise_matrix(sol.matrix)
    print(f'Solution value = {val}')
    # problemInstance.visualise()
    # print(problemInstance.compute_mse(np.array([[1, 1, 1],
    #                                       [2, 2, 2]]), np.array([[0, 0, 0],
    #                                                              [3, 4, 5]])))
