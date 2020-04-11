import fileinput
from typing import Tuple
import numpy as np
from matplotlib import pyplot as plt


class ImageApproximationInstance:
    def __init__(self, matrix, k_coefficient, max_time):
        self.matrix = matrix
        self.k = k_coefficient
        self.max_time = max_time

    @classmethod
    def from_fileinput(cls):
        matrix, max_time, k = cls._read_input()
        return cls(matrix, k, max_time)

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

    def visualise(self):
        plt.imshow(self.matrix)
        plt.gray()
        plt.show()


if __name__ == '__main__':
    problemInstance = ImageApproximationInstance.from_fileinput()
    problemInstance.visualise()
