import fileinput
import sys
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


def x_s_yang(x, epsilons_):
    return np.sum(epsilons_[i] * np.abs(x_) ** i for i, x_ in enumerate(x, 0))


def get_input():
    max_time_, initial_solution_, random_variables_ = None, None, None
    if len(sys.argv) > 1 and sys.argv[1] == '--arguments':
        max_time_ = sys.argv[2]
        initial_solution_ = sys.argv[3:8]
        random_variables_ = sys.argv[8:]
    else:
        for line in fileinput.input():
            data = line.rstrip().split()
            max_time_ = data[2]
            initial_solution_ = data[3:8]
            random_variables_ = data[8:]
    return int(max_time_), \
           np.array([int(x) for x in initial_solution_]), \
           np.array([float(eps) for eps in random_variables_])


@dataclass
class Configuration:
    dimension: int
    x_ranges: List[Tuple[int, int]]
    required_precision: float = 0.1
    population_size: int = 10

    def validate(self):
        return 0 < self.dimension == len(self.x_ranges) and \
               all(len(range_) == 2 for range_ in self.x_ranges) and \
               self.required_precision > 0 and \
               self.population_size > 1


class SolutionRepresentation:
    def __init__(self, x: np.ndarray, x_ranges: List[Tuple[int, int]], precision: float):
        self._value = np.array([3.174, 2.362, 1.301, 2.292, 4.035])
        self.precision = precision
        self._x_ranges = x_ranges
        self._x_ranges_lengths = [abs(upper - lower) for lower, upper in x_ranges]
        print(self._x_ranges_lengths)
        self.chromosome_lengths = [int(np.ceil(np.log2(ranges_length / precision))) for ranges_length in
                                   self._x_ranges_lengths]
        print(self.chromosome_lengths)
        print(self._value)
        self.chromosomes = self.map_to_binary_sequence()
        self.compute_value()

    @property
    def value(self):
        if self._value is not None:
            return self._value
        return self.compute_value()

    def compute_value(self):
        print('compute value')
        chromosomes = [0b100010110010, 0b010101110010, 0b000100110100, 0b010100101011, 0b110000100011]
        chromosomes = self.chromosomes
        print([bin(t) for t in chromosomes])
        value = [
            rng_len * chromosome / (2 ** chromosome_len - 1) + rng[0]
            for chromosome, rng_len, rng, chromosome_len
            in zip(chromosomes, self._x_ranges_lengths, self._x_ranges, self.chromosome_lengths)
        ]
        print(value)
        return value

    def map_to_binary_sequence(self):
        print('map bin to seq')
        seq = []
        for x_i, chromosome_len, rng_len, rng \
                in zip(self._value, self.chromosome_lengths, self._x_ranges_lengths, self._x_ranges):
            print(format(int((x_i - rng[0]) / rng_len * (2 ** chromosome_len - 1)), 'b').zfill(chromosome_len))
            seq.append(int(format(int((x_i - rng[0]) / rng_len * (2 ** chromosome_len - 1)), 'b'), 2))

        print(seq)
        return seq


if __name__ == '__main__':
    max_time, initial_solution, epsilons = get_input()
    print(initial_solution)
    SolutionRepresentation(initial_solution, [(1, 5) for _ in initial_solution], 0.001)
