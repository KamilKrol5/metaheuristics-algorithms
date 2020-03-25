from typing import List, Tuple


class TabuSearchTSP:
    def __init__(self, cities: List[List[int]], max_time: int):
        self.cities = cities
        self.cities_count = len(cities)
        self.max_time = max_time

    @classmethod
    def from_file(cls, filename: str):
        cities, max_time = TabuSearchTSP.read_input(filename)
        return cls(cities, max_time)

    @staticmethod
    def read_input(filename: str) -> Tuple[List[List[int]], int]:
        cities = list()
        print(cities)
        with open(filename, 'r') as file:
            first_line = file.readline()
            print([str(x) for x in first_line.split()])
            [time, cities_count] = [int(x) for x in first_line.split()]
            for i, line in enumerate(file, 0):
                cities.append([])
                cities[i] = [int(x) for x in line.split()]
            if cities_count != len(cities):
                print(f'Number of read cities is different from declared.')
                exit(1)
        print(cities)
        return cities, time

    class TSPPath:
        def __init__(self, move_sequence: List[int]):
            self.move_sequence = move_sequence

        def __copy__(self):
            return TabuSearchTSP.TSPPath(self.move_sequence.copy())

        def __str__(self):
            return f'<TSPPath: {self.move_sequence}>'

        """Returns a new TSPPath object with swapped cities"""

        def swap_cities(self, city1, city2) -> 'TabuSearchTSP.TSPPath':
            new_path = self.__copy__()
            new_path.move_sequence[city1], new_path.move_sequence[city2] = \
                new_path.move_sequence[city2], new_path.move_sequence[city1]
            return new_path

        def __eq__(self, other: 'TabuSearchTSP.TSPPath'):
            return self.move_sequence == other.move_sequence

    def generate_neighbours(self, path: TSPPath, neighbours_max_count=1e6) -> List[TSPPath]:
        neighbours: List[TabuSearchTSP.TSPPath] = []
        for i in range(self.cities_count):
            for j in range(self.cities_count):
                if j > i:
                    neighbours.append(path.swap_cities(i, j))
        return neighbours

    def tabu_search(self):
        tabu = []


if __name__ == '__main__':
    ts = TabuSearchTSP.from_file('l1z2a.txt')

    print([str(x) for x in ts.generate_neighbours(TabuSearchTSP.TSPPath([1, 2, 3, 4, 5]))])
