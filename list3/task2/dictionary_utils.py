from typing import Set


def read_dictionary(filename: str = 'dict.txt') -> Set[str]:
    dictionary = set()
    with open(filename) as file:
        for line in file:
            dictionary.add(str.lower(line.rstrip()))
    return dictionary
