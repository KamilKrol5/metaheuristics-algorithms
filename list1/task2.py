import time
from typing import List, Tuple
import numpy as np


def swap_tuple(tup, i, j):
    tmp = list(tup)
    tmp[i], tmp[j] = tmp[j], tmp[i]
    return tuple(tmp)


class TabuSearchTSP:
    def __init__(self, cities: np.ndarray, max_time: int):
        self.cities = cities
        self.cities_count = len(cities)
        self.max_time = max_time

    @classmethod
    def from_file(cls, filename: str):
        cities, max_time = TabuSearchTSP.read_input(filename)
        return cls(cities, max_time)

    @staticmethod
    def read_input(filename: str) -> Tuple[np.ndarray, int]:
        with open(filename, 'r') as file:
            first_line = file.readline()
            print([str(x) for x in first_line.split()])
            [max_time, cities_count] = [int(x) for x in first_line.split()]
            cities = np.ndarray((cities_count, cities_count))
            # print(cities)
            for i, line in enumerate(file, 0):
                for j, x in enumerate(line.split()):
                    cities[i][j] = int(x)
                # cities[i] = np.ndarray(int(x) for x in line.split())
            if cities_count != len(cities):
                print(f'Number of read cities is different from declared.')
                exit(1)
        # print(cities)
        return cities, max_time

    class TSPPath:
        def __init__(self, move_sequence: Tuple[int]):
            self.move_sequence = move_sequence

        def __copy__(self):
            return TabuSearchTSP.TSPPath(self.move_sequence)

        def __str__(self):
            return f'<TSPPath: {self.move_sequence}>'

        """Returns a new TSPPath object with swapped cities"""

        def swap_cities(self, city1, city2) -> 'TabuSearchTSP.TSPPath':
            new_move_sequence: Tuple[int] = swap_tuple(self.move_sequence, city1, city2)
            return TabuSearchTSP.TSPPath(new_move_sequence)

        def __eq__(self, other: 'TabuSearchTSP.TSPPath'):
            return self.move_sequence == other.move_sequence

        def __hash__(self):
            return hash(self.move_sequence)

    def compute_path_cost(self, path: TSPPath):
        if len(path.move_sequence) == 1:
            return self.cities[path.move_sequence[0]][path.move_sequence[0]]
        # print([self.cities[i][k] for k, i in zip(path.move_sequence, roll(path.move_sequence, 1))])
        total_cost = sum(
            self.cities[path.move_sequence[i]][path.move_sequence[i+1]] for i
            in range(len(path.move_sequence) - 1))
        total_cost += self.cities[path.move_sequence[-1]][path.move_sequence[0]]
        return total_cost

    def generate_neighbours(self, path: TSPPath, neighbours_max_count=None) -> List[TSPPath]:
        neighbours: List[TabuSearchTSP.TSPPath] = []
        for i in range(self.cities_count):
            for j in range(self.cities_count):
                if j > i > 0:
                    neighbours.append(path.swap_cities(i, j))
        if neighbours_max_count:
            np.random.shuffle(neighbours)
            return neighbours[:neighbours_max_count]
        else:
            return neighbours

    def print_cities_matrix(self):
        return '\n'.join([f'{row}' for row in self.cities])

    def tabu_search_basic(self, initial_solution=None,
                          tabu_round_memory=150, max_iterations=500, worsen_factor=1.0):
        if not initial_solution:
            initial_solution = TabuSearchTSP.TSPPath(tuple(range(self.cities_count)))

        tabu = {}
        x = initial_solution
        the_best_of_the_best = (initial_solution, self.compute_path_cost(x))

        for i in range(max_iterations):
            neighbourhood = self.generate_neighbours(x, neighbours_max_count=1_000_000_000)
            x_cost = self.compute_path_cost(x)

            if i % 25 == 0:
                print(f'Iteration {i}\nx = {x}\ncost = {x_cost}\ntabu size = {sum(len(b) for b in tabu.values())}')
                # print(f'tabu: {[(str(x), i ,v) for x, (i, v) in tabu.items()]}')

            best_neighbour = None
            best_neighbour_cost = x_cost * worsen_factor

            for neighbour in (n for n in neighbourhood if n not in tabu.keys()):
                neighbour_cost = self.compute_path_cost(neighbour)
                # tabu[neighbour] = (i, neighbour_cost)  # this cost is unnecessary for now
                if neighbour_cost < best_neighbour_cost:
                    best_neighbour = neighbour
                    best_neighbour_cost = neighbour_cost

            the_best_of_the_best = (x, x_cost) if x_cost < the_best_of_the_best[1] else the_best_of_the_best
            if best_neighbour:
                x = best_neighbour
                tabu[best_neighbour] = (i, best_neighbour_cost)  # this cost is unnecessary for now
                # clean old tabu data
                tabu = {k: (j, val) for k, (j, val) in tabu.items() if i - tabu_round_memory < j}
            else:
                return the_best_of_the_best[0], the_best_of_the_best[1], max_iterations

        return the_best_of_the_best[0], the_best_of_the_best[1], max_iterations


if __name__ == '__main__':
    ts = TabuSearchTSP.from_file('l1z2b.txt')
    # print(ts.print_cities_matrix())
    # print(ts.compute_path_cost(TabuSearchTSP.TSPPath([0, 3, 2, 1, 4])))
    # print([str(x) for x in ts.generate_neighbours(TabuSearchTSP.TSPPath([1, 2, 3, 4, 5]))])
    random_permutation = tuple([0] + list(np.random.permutation(range(1, ts.cities_count))))
    print(random_permutation)
    t = time.time()
    path, cost, iterations = ts.tabu_search_basic(TabuSearchTSP.TSPPath(random_permutation), worsen_factor=1.1)
    print(f'Time: {time.time() - t}')
    print(f'Initial solution: {random_permutation} Iterations: {iterations}')
    print(path)
    print(cost)
