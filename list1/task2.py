import time
from abc import ABC
from typing import List, Tuple
import numpy as np
import fileinput


def swap_tuple(tup, i, j):
    tmp = list(tup)
    tmp[i], tmp[j] = tmp[j], tmp[i]
    return tuple(tmp)


class TSPPath:
    def __init__(self, move_sequence: Tuple[int]):
        self.move_sequence = move_sequence

    """Returns a new TSPPath object with swapped cities"""

    def swap_cities(self, city1, city2) -> 'TSPPath':
        new_move_sequence: Tuple[int] = swap_tuple(self.move_sequence, city1, city2)
        return TSPPath(new_move_sequence)

    def __eq__(self, other: 'TSPPath'):
        return self.move_sequence == other.move_sequence

    def __hash__(self):
        return hash(self.move_sequence)

    def __str__(self):
        return f'<TSPPath: {self.move_sequence}>'


class TSPSolution:
    def __init__(self, tsp_path: TSPPath, tsp_instance: 'TSPInstance'):
        self.tsp_path = tsp_path
        self.cost_table = tsp_instance.cities
        self._cost = None

    @classmethod
    def with_cost(cls, tsp_path: TSPPath, tsp_instance: 'TSPInstance', given_cost: int) -> 'TSPSolution':
        solution = TSPSolution(tsp_path, tsp_instance)
        solution._cost = given_cost
        return solution

    @property
    def cost(self):
        if self._cost is not None:
            return self._cost

        total_cost = sum(
            self.cost_table[self.tsp_path.move_sequence[i]][self.tsp_path.move_sequence[i + 1]] for i
            in range(len(self.tsp_path.move_sequence) - 1))

        total_cost += self.cost_table[self.tsp_path.move_sequence[-1]][self.tsp_path.move_sequence[0]]
        self._cost = total_cost
        return total_cost

    def __hash__(self):
        return self.tsp_path.__hash__()

    def __eq__(self, other: 'TSPSolution'):
        return self.tsp_path.move_sequence == other.tsp_path.move_sequence

    def __str__(self):
        return f'{self.tsp_path}, cost = {self.cost};'


class TSPInstance(ABC):
    def __init__(self, cities: np.ndarray, max_time: int):
        self.cities = cities
        self.cities_count = len(cities)
        self.max_time = max_time

    @classmethod
    def from_file(cls, filename: str):
        cities, max_time = TSPInstance.read_input(filename)
        return cls(cities, max_time)

    @classmethod
    def from_stdin(cls):
        cities, max_time = TSPInstance.read_input()
        return cls(cities, max_time)

    @staticmethod
    def read_input(filename=None) -> Tuple[np.ndarray, int]:
        with open(filename, 'r') if filename is not None else fileinput.input() as file:
            first_line = file.readline()
            print([str(x) for x in first_line.split()])
            [max_time, cities_count] = [int(x) for x in first_line.split()]
            cities = np.ndarray((cities_count, cities_count))

            for i, line in enumerate(file, 0):
                for j, x in enumerate(line.split()):
                    cities[i][j] = int(x)

            if cities_count != len(cities):
                print(f'Number of read cities is different from declared.')
                exit(1)

        return cities, max_time


class TabuSearchTSP(TSPInstance):
    def generate_neighbours(self, solution: TSPSolution, neighbours_max_count=None) -> List[TSPSolution]:
        neighbours: List[TSPSolution] = []
        for i in range(self.cities_count):
            for j in range(self.cities_count):
                if j > i > 0:
                    neighbour_path = solution.tsp_path.swap_cities(i, j)
                    neighbour_cost = self.compute_cost_from_existing_path(solution.tsp_path, solution.cost, (i, j))
                    neighbours.append(TSPSolution.with_cost(neighbour_path, self, neighbour_cost))
        if neighbours_max_count:
            np.random.shuffle(neighbours)
            return neighbours[:neighbours_max_count]
        else:
            return neighbours

    def get_cost(self, k, l):
        return self.cities[k, l]

    def compute_cost_from_existing_path(
            self, base_path: TSPPath, base_path_cost: int, *inversions: Tuple[int, int]) -> int:
        sequence = base_path.move_sequence
        sequence_len = len(sequence)
        for i, j in inversions:
            if abs(i - j) == 1:
                base_path_cost -= self.cities[sequence[i - 1]][sequence[i]] + \
                    self.cities[sequence[i]][sequence[j]] + \
                    self.cities[sequence[j]][sequence[(j + 1) % sequence_len]]
                base_path_cost += self.cities[sequence[i - 1]][sequence[j]] + \
                    self.cities[sequence[j]][sequence[i]] + \
                    self.cities[sequence[i]][sequence[(j + 1) % sequence_len]]
                return base_path_cost
            base_path_cost -= self.cities[sequence[i - 1]][sequence[i]] + \
                self.cities[sequence[i]][sequence[(i + 1) % sequence_len]] + \
                self.cities[sequence[j - 1]][sequence[j]] + \
                self.cities[sequence[j]][sequence[(j + 1) % sequence_len]]
            base_path_cost += self.cities[sequence[i - 1]][sequence[j]] + \
                self.cities[sequence[j]][sequence[(i + 1) % sequence_len]] + \
                self.cities[sequence[j - 1]][sequence[i]] + \
                self.cities[sequence[i]][sequence[(j + 1) % sequence_len]]
            return base_path_cost

    def print_cities_matrix(self):
        return '\n'.join([f'{row}' for row in self.cities])

    def generate_random_initial_solution(self) -> TSPSolution:
        best = None
        best_cost = np.inf
        for _ in range(max(3, self.cities_count // 2)):
            random_sequence = tuple([0] + list(np.random.permutation(range(1, self.cities_count))))
            solution = TSPSolution(TSPPath(random_sequence), self)

            if best_cost > solution.cost:
                best = solution
                best_cost = solution.cost
        return best

    def tabu_search_basic(self, initial_solution=None,
                          tabu_round_memory=1500, max_iterations=3000, worsen_factor=1.0) -> Tuple[TSPSolution, int]:
        start_time = time.time()
        if initial_solution is None:
            initial_solution = self.generate_random_initial_solution()

        tabu = {}
        current_solution = initial_solution
        currently_best_known_solution = initial_solution

        for i in range(max_iterations):
            if time.time() - start_time > self.max_time:
                return currently_best_known_solution, i-1

            neighbourhood = self.generate_neighbours(current_solution)

            if i % 25 == 0:
                print(f'Iteration {i}\nx = {current_solution}\ncost = {current_solution.cost}')
                print(f'tabu size = {len(tabu.values())}')
                # print(f'tabu: {[(str(x), i ,v) for x, (i, v) in tabu.items()]}')
                print([n for n in neighbourhood if n not in tabu.keys()] == neighbourhood)

            best_neighbour = None
            best_neighbour_cost = current_solution.cost * worsen_factor

            for neighbour in (n for n in neighbourhood if n not in tabu.keys()):

                # tabu[neighbour] = (i, neighbour.cost)
                if neighbour.cost < best_neighbour_cost:
                    best_neighbour = neighbour
                    best_neighbour_cost = neighbour.cost

            # if i != 0 and currently_best_known_solution.tsp_path == current_solution.tsp_path:
            #     return currently_best_known_solution, i
            currently_best_known_solution = \
                current_solution if current_solution.cost < currently_best_known_solution.cost \
                else currently_best_known_solution

            if best_neighbour:
                current_solution = best_neighbour
                tabu[best_neighbour] = (i, best_neighbour_cost)

                # clean old tabu data
                usability_threshold = pow(worsen_factor, 3) * best_neighbour_cost
                tabu = {k: (j, val)
                        for k, (j, val) in tabu.items()
                        if i - tabu_round_memory < j and val < usability_threshold}

            else:
                return currently_best_known_solution, max_iterations

        return currently_best_known_solution, max_iterations


if __name__ == '__main__':
    ts = TabuSearchTSP.from_stdin()
    t = time.time()
    solution, iterations = ts.tabu_search_basic(worsen_factor=1.1)
    print(f'Time: {time.time() - t}')
    print(f'Iterations: {iterations}')
    print(solution.tsp_path)
    print(solution.cost)
