import time
import fileinput
import sys
from collections import deque
from itertools import combinations_with_replacement
import numpy as np


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
    return x + x * (np.random.random(4) * 2 - 1)


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
                        max_execution_time,
                        c=-1):
    end_time = time.time() + max_execution_time
    temperature = initial_temperature
    current_solution = initial_solution
    while temperature > red_factor and time.time() < end_time:
        neighbour = random_neighbour_function(current_solution)
        delta_f = function(neighbour) - function(current_solution)
        if probability(delta_f, temperature, c) > np.random.rand():
            current_solution = neighbour
        temperature = temperature * (1 - red_factor)

    return current_solution, function(current_solution)


if __name__ == '__main__':
    time_, *v = get_input()
    v = np.array(v)
    solution, value = simulated_annealing(v,
                                          salomon,
                                          salomon_random_neighbour,
                                          initial_temperature=200,
                                          max_execution_time=time_,
                                          red_factor=0.005)
    print(*solution.flatten(), value)
