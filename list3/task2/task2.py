import sys

from dictionary_utils import read_dictionary
from scrabble_genetic import ScrabbleGenetic

if __name__ == '__main__':
    dictionary = read_dictionary()
    scrabble_genetic = ScrabbleGenetic.from_stdin(
        max_population_size=300,
        dictionary=dictionary,
        mutation_probability=0.30
    )

    solution, points = scrabble_genetic.run_algorithm()
    print(f'End population: {scrabble_genetic.population}', file=sys.stderr)
    print(f'End population costs: {[scrabble_genetic.word_utils.points(i) for i in scrabble_genetic.population]}',
          file=sys.stderr)
    print(f'Solution: {solution}', file=sys.stderr)
    print(f'Points: {points}', file=sys.stderr)
    print(solution, file=sys.stderr)
    print(points)
