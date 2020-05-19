import fileinput
import sys
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional, Any

import numpy as np

from utils.utils import grouped_by_2, negate_bit


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
    max_time: int
    mutation_probability: float
    required_precision: float = 0.1
    population_size: int = 5

    def validate(self):
        return 0 < self.dimension == len(self.x_ranges) and \
               0 <= self.mutation_probability <= 1 and \
               all(len(range_) == 2 for range_ in self.x_ranges) and \
               self.required_precision > 0 and \
               self.population_size > 1 and \
               self.max_time > 0


class SolutionRepresentation:
    def __init__(self, x_ranges: List[Tuple[int, int]], precision: float):
        # self._value = np.array([3.174, 2.362, 1.301, 2.292, 4.035])
        self.value: Optional[np.ndarray] = None
        self.chromosomes: Optional[List] = None
        self.precision = precision
        self.x_ranges = x_ranges
        self._x_ranges_lengths = [abs(upper - lower) for lower, upper in x_ranges]
        # print(self._x_ranges_lengths)
        self.chromosome_lengths = [int(np.ceil(np.log2(ranges_length / precision))) for ranges_length in
                                   self._x_ranges_lengths]
        # print(self.chromosome_lengths)
        # print(self._value)
        # self.chromosomes = self.map_to_binary_sequence()
        # self.compute_value()

    @classmethod
    def from_value(cls,
                   x_ranges: List[Tuple[int, int]],
                   precision: float,
                   value: np.ndarray
                   ) -> 'SolutionRepresentation':
        solution = cls(x_ranges, precision)
        solution.value = value
        solution.chromosomes = solution.map_to_binary_sequence()
        return solution

    @classmethod
    def from_chromosomes(cls,
                         x_ranges: List[Tuple[int, int]],
                         precision: float,
                         chromosomes: List[Any]
                         ) -> 'SolutionRepresentation':
        solution = cls(x_ranges, precision)
        solution.chromosomes = chromosomes
        solution.value = solution.compute_value()
        return solution

    # @property
    # def value(self):
    #     # if self._value is not None:
    #     #     return self._value
    #     return self.compute_value()

    def compute_value(self):
        # print('compute value')
        # chromosomes = [0b100010110010, 0b010101110010, 0b000100110100, 0b010100101011, 0b110000100011]
        chromosomes = self.chromosomes
        # print([bin(t) for t in chromosomes])
        value = [
            rng_len * chromosome / (2 ** chromosome_len - 1) + rng[0]
            for chromosome, rng_len, rng, chromosome_len
            in zip(chromosomes, self._x_ranges_lengths, self.x_ranges, self.chromosome_lengths)
        ]
        # print(value)
        return value

    def map_to_binary_sequence(self):
        # print('map bin to seq')
        seq = []
        for x_i, chromosome_len, rng_len, rng \
                in zip(self.value, self.chromosome_lengths, self._x_ranges_lengths, self.x_ranges):
            # print(format(int((x_i - rng[0]) / rng_len * (2 ** chromosome_len - 1)), 'b').zfill(chromosome_len))
            seq.append(int(
                format(int((x_i - rng[0]) / rng_len * (2 ** chromosome_len - 1)), 'b').zfill(chromosome_len), 2)
            )

        # print(seq)
        return seq

    def __str__(self):
        return f'value = {self.value}, chromosomes = {self.chromosomes}, cost = {fitness_fun(self)}'

    def __repr__(self):
        return str(self)


class XSYangFitnessFunction:
    def __init__(self, epsilons_):
        self.epsilons = epsilons_

    @staticmethod
    def x_s_yang(x, epsilons_):
        return np.sum([epsilons_[i - 1] * (np.abs(x_) ** i) for i, x_ in enumerate(x, 1)])

    def __call__(self, x: SolutionRepresentation, *args, **kwargs):
        return self.x_s_yang(x.value, self.epsilons)


class Optimization:
    def __init__(self, conf: Configuration, fitness_function):
        if not conf.validate():
            raise ValueError(f'Invalid configuration provided. {str(conf)}')

        self.fitness_function = fitness_function
        self.conf = conf
        self.population: List[SolutionRepresentation] = []
        # self.current_population_size: int = 0
        # self._solution: Optional[SolutionRepresentation] = None

    @classmethod
    def with_initial_population(cls,
                                conf: Configuration,
                                fitness_function,
                                initial_population: List[SolutionRepresentation]
                                ) -> 'Optimization':
        instance = cls(conf, fitness_function)
        if len(initial_population) != instance.conf.population_size:
            raise ValueError('Initial population size is different from the size specified in provided configuration')

        instance.population.extend(initial_population)
        return instance

    @property
    def solution(self) -> SolutionRepresentation:
        return sorted(self.population, key=lambda i: self.fitness_function(i), reverse=True)[0]

    @staticmethod
    def cross_operator(parent1: SolutionRepresentation,
                       parent2: SolutionRepresentation,
                       mutation_chance
                       ) -> Tuple[SolutionRepresentation, SolutionRepresentation]:
        child1_chromosomes: List[int] = []
        child2_chromosomes: List[int] = []
        for parent1_chromosome, parent2_chromosome, new_chromosome_len \
                in zip(parent1.chromosomes, parent2.chromosomes, parent1.chromosome_lengths):
            index1 = np.random.randint(0, new_chromosome_len // 2, 1)[0]
            index2 = np.random.randint(index1 + 1, new_chromosome_len, 1)[0]

            parent1_f: str = format(parent1_chromosome, 'b').zfill(new_chromosome_len)
            parent2_f: str = format(parent2_chromosome, 'b').zfill(new_chromosome_len)

            parent1_chromosomes = "".join((parent1_f[:index1], parent2_f[index1:index2], parent1_f[index2:]))
            parent2_chromosomes = "".join((parent2_f[:index1], parent1_f[index1:index2], parent2_f[index2:]))

            child1_new_chromosome = int(parent1_chromosomes, 2)
            child2_new_chromosome = int(parent2_chromosomes, 2)

            child1_new_chromosome = Optimization._mutate(child1_new_chromosome, mutation_chance, new_chromosome_len)
            child2_new_chromosome = Optimization._mutate(child2_new_chromosome, mutation_chance, new_chromosome_len)

            child1_chromosomes.append(child1_new_chromosome)
            child2_chromosomes.append(child2_new_chromosome)

        child1 = SolutionRepresentation.from_chromosomes(parent1.x_ranges, parent1.precision, child1_chromosomes)
        child2 = SolutionRepresentation.from_chromosomes(parent1.x_ranges, parent1.precision, child2_chromosomes)
        return child1, child2

    def _selection(self) -> List[SolutionRepresentation]:
        self.population.sort(key=lambda individual: self.fitness_function(individual), reverse=True)
        selected = self.population[:int(np.ceil(self.conf.population_size / 2))]
        return selected

    def _reproduce_with_mutation(self,
                                 selected_to_reproduce: List[SolutionRepresentation]
                                 ) -> List[SolutionRepresentation]:
        if len(selected_to_reproduce) <= 1:
            raise ValueError('Cannot reproduce one individual')
        if (len(selected_to_reproduce) - 1) % 2 == 0:
            selected_to_reproduce = selected_to_reproduce[:-1]

        children = []
        for mother, father in grouped_by_2(selected_to_reproduce):
            children.extend(self.cross_operator(mother, father, self.conf.mutation_probability))

        return children

    @staticmethod
    def _mutate(chromosome: int, mutation_probability, bits_count) -> int:
        if np.random.rand() <= mutation_probability:
            index = np.random.randint(0, bits_count)
            chromosome = negate_bit(chromosome, index)
        return chromosome

    def optimize(self):
        end_time = time.time() + self.conf.max_time
        while time.time() < end_time:
            self._evolve()
            print("\n".join(map(str, self.population)))
            print('---')
        return self.solution

    def _evolve(self):
        to_reproduce: List[SolutionRepresentation] = self._selection()
        children: List[SolutionRepresentation] = self._reproduce_with_mutation(to_reproduce)
        self.population.extend(children)
        self.population.sort(key=lambda i: self.fitness_function(i))
        self.population = self.population[:self.conf.population_size]


if __name__ == '__main__':
    print(XSYangFitnessFunction.x_s_yang([0, 0, 0], [0.3, 0.5, 0.7]))
    max_time, initial_solution, epsilons = get_input()
    print(initial_solution)
    dim = initial_solution.shape[0]
    solutions_range = [(-5, 5) for _ in range(dim)]
    req_precision = 0.001

    config = Configuration(dim,
                           solutions_range,
                           max_time,
                           mutation_probability=0.2,
                           required_precision=req_precision,
                           population_size=10)
    fitness_fun = XSYangFitnessFunction(epsilons)
    r = solutions_range[0]
    initial_population_ = [
        SolutionRepresentation.from_value(solutions_range, req_precision, np.random.randint(r[0], r[1], size=dim))
        for _ in range(config.population_size)
    ]
    initial_population_[0] = SolutionRepresentation.from_value(solutions_range, req_precision, initial_solution)
    optimization = Optimization.with_initial_population(config, fitness_fun, initial_population_)
    print(optimization.optimize())
