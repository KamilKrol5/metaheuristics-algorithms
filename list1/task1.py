import sys
import numpy as np

INITIAL_RANGE = 256


def generate_random_vector(dimension, low=-INITIAL_RANGE, high=INITIAL_RANGE):
    return list(np.random.uniform(low, high, dimension))


""" The function performs local search
    Arguments:
    function - The function for which minimum needs to be found.
    neighbour_function - function which takes current solution and returns collection of its neighbours.
                         It is basically the definition of neighbourhood in local search.
    initial_solution - A start solution. For example it may be a random one.
    cond_of_satisfaction - The function which takes the current solution (x) and returns True
                           if x fills the condition of satisfaction/end condition, False otherwise.
                           It determines if the solution is good enough to end the job.
    max_iterations - An integer which determines maximum number of iterations.
"""


def local_search(function, neighbour_function, initial_solution, cond_of_satisfaction, max_iterations=100000, max_fails = 1):
    x = initial_solution
    fails = 0
    for i in range(max_iterations):
        if cond_of_satisfaction and cond_of_satisfaction(x):
            return x
        print(f'Iteration: {i}, x = {x}', file=sys.stderr)
        function_val = function(x)
        neighbourhood = neighbour_function(x)
        # trying to find better solution among the neighbours
        for neighbour in neighbourhood:
            function_val_for_neighbour = function(neighbour)
            # print(f'{neighbour} {function_val_for_neighbour}, {function(neighbour)}')
            # if better solution is found we take it as the current one and continue searching
            if function_val > function_val_for_neighbour:
                x = neighbour
                break
        # in case all neighbours has been checked and none of them is better solution the work is
        # considered to be done
        else:
            fails += 1
            if fails >= max_fails:
                return x
    # if max iterations are reached then current solution is returned
    else:
        return x


def happy_cat(x):
    n = 4
    alpha = 0.125
    x_norm = np.linalg.norm(x)
    return ((x_norm - n) ** 2) ** alpha + 1 / 4 * (0.5 * x_norm ** 2 + sum((x_i for x_i in x))) + 1 / 2


def neighbours_for_cat_random(s, number_of_neighbours=1):
    return [[-x_i for x_i in s], [np.sign(x_i) * x_i**2 for x_i in s], [2 * x_i for x_i in s]] +\
        [[x_i + np.random.uniform(-1, 1) * pow(abs(x_i), 0.3) for x_i in s] for _ in range(number_of_neighbours-3)]


def griewank(x):
    return 1 + 1/4000 * sum((x_i**2 for x_i in x)) - np.product([np.cos(x_i/np.sqrt(i)) for i, x_i in enumerate(x, 1)])


def neighbours_for_griewank(s, number_of_neighbours=1):
    return [[x_i + np.random.uniform(-1, 1) * abs(x_i) for x_i in s] for _ in range(number_of_neighbours)]


if __name__ == '__main__':
    f = happy_cat
    X = generate_random_vector(4, -2560, 2560)
    res = local_search(f, lambda t: neighbours_for_cat_random(t, 1000), X, None, max_fails=1)
    print('Happy cat')
    print(f'Solution: {res}')
    print(f'Value: {f(res)}')
    print(f'Relative error of solution: '
          f'{abs(np.linalg.norm(res) - np.linalg.norm([-1, -1, -1, -1]))/np.linalg.norm([-1, -1, -1, -1])}')

    # print(griewank([11.013]))  # = 1.0128967117029901
    # print(griewank([0]))  # = 0 - global minimum

    g = griewank
    X = generate_random_vector(4, -2560, 2560)
    res = local_search(g, lambda t: neighbours_for_griewank(t, 1000), X, None)
    print('Griewank')
    print(f'Solution: {res}')
    print(f'Value: {g(res)}')
