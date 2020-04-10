import fileinput
import sys
from collections import deque
import numpy as np
from itertools import combinations_with_replacement, permutations, combinations


def salomon(x: np.ndarray):
    sqrt_of_powers = np.sqrt(np.sum(np.power(x, 2)))
    return 1.0 - np.cos(2 * np.pi * sqrt_of_powers) + 0.1 * sqrt_of_powers


def salomon_neighbourhood_generator(x: np.ndarray):
    dim = x.shape[0]
    combinations_ = (deque(combination) for combination in combinations_with_replacement([1, -1], dim))
    for c in combinations_:
        if len(set(c)) > 1:
            for _ in range(dim):
                c.rotate(1)
                yield x * 0.5 * c
        else:
            yield x * 0.5 * c


def salomon_random_neighbour(x: np.ndarray):
    return x * 0.5 * np.random.choice([1, -1], 4)


def get_input():
    _max_time, _initial_solution = None, None
    if len(sys.argv) > 1 and sys.argv[1] == '--arguments':
        _max_time = sys.argv[2]
        _initial_solution = sys.argv[3:]
    else:
        for line in fileinput.input():
            _max_time, *_initial_solution = line.rstrip().split()
    return int(_max_time), [int(x) for x in _initial_solution]


def probability(delta_f, temperature, c):
    if delta_f <= 0:
        return 1
    return 1.0 / (1.0 + np.power(np.e, c * delta_f / temperature))


def simulated_annealing(initial_solution,
                        function,
                        random_neighbour_function,
                        initial_temperature,
                        red_factor,
                        c=-1):
    temperature = initial_temperature
    current_solution = initial_solution
    while temperature > red_factor:
        neighbour = random_neighbour_function(current_solution)
        delta_f = function(neighbour) - function(current_solution)
        if probability(delta_f, temperature, c) > np.random.rand():
            current_solution = neighbour
        temperature = temperature * (1 - red_factor)

    return current_solution, function(current_solution)


if __name__ == '__main__':
    time, *v = get_input()
    v = np.array(v)
    solution, value = simulated_annealing(v,
                                          salomon,
                                          salomon_random_neighbour,
                                          initial_temperature=200,
                                          red_factor=0.1)
    print(solution)
    print(value)
