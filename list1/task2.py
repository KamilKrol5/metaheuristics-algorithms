

def read_input(filename: str):
    cities = list()
    print(cities)
    with open(filename, 'r') as file:
        first_line = file.readline()
        print([str(x) for x in first_line.split()])
        [time, cities_count] = [int(x) for x in first_line.split()]
        for i, line in enumerate(file, 0):
            cities.append([])
            cities[i] = [int(x)for x in line.split()]
        if cities_count != len(cities):
            print(f'Number of read cities is different from declared.')
            exit(1)
    print(cities)


if __name__ == '__main__':
    read_input('l1z2a.txt')
