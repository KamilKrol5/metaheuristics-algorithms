from collections import defaultdict


def read_input(filename: str):
    cities = defaultdict(list)
    print(cities)
    with open(filename, 'r') as file:
        first_line = file.readline(limit=1)
        time, cities_count = ((str(x) for x in first_line.split()))
        for i, line in enumerate(file):
            cities[i].extend(int(x) for x in line.split())
