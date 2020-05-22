import fileinput
import random
import sys
import time
from collections import Counter
from itertools import chain, takewhile, count
from typing import Dict, Collection, List, Tuple, Generator, Optional, Union

import numpy as np

from scrabble import Scrabble, Word, Letter, WordUtils
from utils.utils import flatten, grouped_by_2, zip_longest


class ScrabbleGenetic(Scrabble):
    MINIMUM_SELECTED = 2

    class CannotMakeValidChildren(Exception):
        pass

    class IndividualsRandomZipper:
        def __init__(self, sg: 'ScrabbleGenetic', mother: Word, father: Word):
            self.sg = sg
            self.father = father
            self.mother = mother

            # self.fill_values = np.random.permutation(list(chain(sg.unused_letters(mother),
            # sg.unused_letters(father))))
            self.fill_values = np.random.permutation(list(sg.available_letters))
            if len(self.fill_values) == 0:
                self.fill_values = sg.available_letters

        def fill_generator(self):
            return np.random.choice(self.fill_values)

        def zip_randomly(self, reverse_mother=False, reverse_father=False) -> str:
            mother, father = self.mother, self.father
            if reverse_mother:
                mother = reversed(mother)
            if reverse_father:
                father = reversed(father)
            fetus = []
            for mother_char, father_char \
                    in zip_longest(mother, father, fill_values_generator=self.fill_generator):
                if np.random.rand() < 0.5:
                    fetus.append(mother_char)
                else:
                    fetus.append(father_char)
            return ''.join(fetus)

    def __init__(self,
                 max_time: int,
                 available_letters: Dict[str, Letter],
                 max_population_size: int,
                 dictionary: Collection[str],
                 mutation_probability: float,
                 initial_population: List[Word]
                 ):
        super().__init__(max_time, available_letters, dictionary)
        self.mutation_probability = mutation_probability
        self.improvement_rate = 1.0
        self.mutation_probability_multiplier = 1.0
        self.initial_population = initial_population
        self.max_population_size = max_population_size
        self.word_utils = WordUtils(dictionary, available_letters)
        self.population = self.initial_population
        self._initialize_population()
        self.start_population = self.population.copy()
        self.population.sort(key=lambda i: self.word_utils.points(i), reverse=True)
        print(f'STARTING POPULATION: {self.population}', file=sys.stderr)

    @property
    def solution(self) -> Tuple[Word, int]:
        solution = sorted(self.population, key=lambda i: self.word_utils.points(i), reverse=True)[0]
        return solution, self.word_utils.points(solution)

    def unused_letters(self, word: Word) -> Generator[str, None, None]:
        letters_in_word = Counter(word)
        letters_in_word.setdefault(0)
        letters_left = [list((av.count - letters_in_word[av.letter]) * av.letter)
                        for av in self.available_letters.values()
                        if av.count > letters_in_word[av.letter]]
        return flatten(letters_left)

    def _initialize_population(self) -> None:
        for i in range(self.max_population_size - len(self.initial_population)):
            new_individual = self._generate_random_word()
            if new_individual is not None and new_individual not in self.population:
                self.population.append(new_individual)
        print(f'population size: {len(self.population)}. Maximum is: {self.max_population_size}', file=sys.stderr)

    # def get_letters_with_repetitions(self) -> Generator[str, None, None]:
    #     return flatten(list(l.count * l.letter) for l in self.available_letters.values())

    def _generate_random_word(self, attempts_count=2) -> Optional[Word]:
        random_word_from_population = np.random.choice(self.population)
        for _ in range(attempts_count):
            mutated = self._mutate(random_word_from_population)
            mutated_valid_prefix = self.word_utils.longest_valid_prefix(mutated)
            if mutated_valid_prefix is not None:
                return mutated_valid_prefix
        return None

    @classmethod
    def from_stdin(cls,
                   max_population_size: int,
                   dictionary: Collection[str],
                   mutation_probability: float,
                   ) -> 'ScrabbleGenetic':
        available_letters, initial_solutions, max_time = ScrabbleGenetic.read_input()
        return cls(max_time=max_time,
                   available_letters=available_letters,
                   max_population_size=max_population_size,
                   dictionary=dictionary,
                   mutation_probability=mutation_probability,
                   initial_population=initial_solutions)

    @staticmethod
    def read_input(filename=None) -> Tuple[Dict[str, Letter], List[Word], int]:
        with open(filename, 'r') if filename is not None else fileinput.input() as file:
            first_line = file.readline()
            [max_time, letters_count, initial_solution_count] = [int(x) for x in first_line.split()]

            available_letters = {}
            initial_solutions: List[Word] = []
            for i, line in enumerate(file, 0):
                if i < letters_count:
                    letter, points = str.rstrip(line).split()
                    points = int(points)
                    if letter in available_letters:
                        available_letters[letter].count += 1
                    else:
                        available_letters[letter] = Letter(letter, points, 1)
                else:
                    initial_solutions.append(Word(str.lower(line.rstrip())))

            if len(initial_solutions) != initial_solution_count:
                raise ValueError('Number of provided solutions different from declared.')

            print(f'Provided initial solutions: {initial_solutions}', file=sys.stderr)
            print(f'Provided available letters: {available_letters}', file=sys.stderr)
        return available_letters, initial_solutions, max_time

    def _chance_for_reproduction(self, word: Word, divider=None) -> float:
        if divider is None:
            divider = np.max([self.word_utils.points(word) for word in self.population])
        return (self.word_utils.points(word) / divider) ** 2

    @staticmethod
    def _random_swap(individual: Word, swap_count=1) -> Word:
        length = len(individual)
        if len(individual) < 2:
            return individual
        individual = list(individual)
        for _ in range(swap_count):
            i = np.random.randint(0, length)
            j = np.random.randint(0, length)
            individual[i], individual[j] = individual[j], individual[i]
        return Word(''.join(individual))

    def _random_letter_change(self, individual: Word, random_changes=1) -> Tuple[Word, List[int]]:
        length = len(individual)
        mutated_indexes = []
        for _ in range(random_changes):
            mutated_index = np.random.randint(0, length)
            if mutated_index == length - 1:
                end = ''
            else:
                end = individual[mutated_index + 1:]
            free_letters = list(chain(self.unused_letters(individual), [individual[mutated_index]]))
            individual = individual[:mutated_index] + np.random.choice(free_letters) + end
            mutated_indexes.append(mutated_index)
        return individual, mutated_indexes

    def _mutate(self, individual: Word) -> Word:
        """
        May return word which is not valid.
        Given individual is not changed.
        """
        changed = individual
        for _ in range(1, int((self.mutation_probability_multiplier - 1) * 10) + 1):
            rand = np.random.rand()
            if rand < 0.78:
                changed, _ = self._random_letter_change(changed)
            elif rand < 0.86:
                changed = self._random_swap(individual)
            elif rand < 0.96:
                changed = self._random_swap(individual)
                changed, _ = self._random_letter_change(changed)

        letters_for_random_tail = list(self.unused_letters(changed))
        # letters_for_random_tail = list(self.available_letters.keys())
        if len(letters_for_random_tail) > 0:
            np.random.shuffle(letters_for_random_tail)
            random_tail = ''.join(letters_for_random_tail)
            if np.random.rand() < 0.90:
                result = changed + random_tail
            else:
                random_prefix = ''.join(
                    np.random.choice(letters_for_random_tail, np.random.randint(0, len(individual))))
                result = random_prefix + changed + random_tail
        else:
            random_tail = 'NONE'
            result = changed
        # print(f'MUTATION: {individual} -> {result}; mutated_indexes = ...; random_tail = {random_tail};',
        #       file=sys.stderr)
        return result

    def _random_mutate(self, individual: Union[Word, str]) -> Word:
        """
        May return word which is not valid.
        Given individual is not changed.
        """
        if self.mutation_probability_multiplier * np.random.rand() < self.mutation_probability:
            return self._mutate(individual)
        return individual

    def _evolve(self) -> None:
        mean_of_last_gen_elite = \
            np.mean([self.word_utils.points(i) for i in self.population[:min(2, self.max_population_size // 10)]])

        to_reproduce = self._selection()
        failed = self._reproduce(to_reproduce)
        if failed == len(to_reproduce):
            print(f'It was not possible to make a child for any of selected individuals.', file=sys.stderr)

        old_rate = self.improvement_rate
        self.improvement_rate = abs(
            np.mean([self.word_utils.points(i) for i in self.population[:min(2, self.max_population_size // 10)]]) -
            mean_of_last_gen_elite)

        if self.improvement_rate > 0:
            self.mutation_probability_multiplier = 1.0
        elif old_rate == self.improvement_rate == 0:
            self.mutation_probability_multiplier += 0.0001 / self.mutation_probability_multiplier
            self.mutation_probability_multiplier = min(self.mutation_probability_multiplier, 1.2)

        print(f'Improvement rate: {self.improvement_rate}; '
              f'Mutation chance multiplier: {self.mutation_probability_multiplier}', file=sys.stderr)

    def _selection(self) -> List[Word]:
        selected: List[Word] = []
        # self.population.sort(key=lambda h: self.word_utils.points(h), reverse=True)
        # max_points = np.max([self.word_utils.points(word) for word in self.population[:len(self.population) // 5]])
        max_points = np.max([self.word_utils.points(word) for word in self.population[:5]])
        for i in takewhile(lambda _: len(selected) < self.MINIMUM_SELECTED, count(0.0, 0.1)):
            selected.clear()
            for individual in self.population:
                if i + np.random.rand() < self._chance_for_reproduction(individual, max_points):
                    selected.append(individual)

        print(f'Selected for reproduction: {len(selected)} of {len(self.population)}', file=sys.stderr)
        np.random.shuffle(selected)
        return selected

    @staticmethod
    def _connect_randomly(mother: Word, father: Word):
        split_index_mother = random.randrange(len(mother))
        split_index_father = random.randrange(len(father))
        return mother[:split_index_mother] + father[split_index_father]

    def _cross(self, mother: Word, father: Word, attempts_count: int) -> List[Word]:
        zipper = self.IndividualsRandomZipper(self, mother, father)
        parents_points_min = min(self.word_utils.points(mother), self.word_utils.points(father))
        children = []
        # fetuses = [
        #     zipper.zip_randomly(reverse_mother=True),
        #     zipper.zip_randomly(reverse_father=True),
        #     zipper.zip_randomly(True, True),
        # ]
        # fetuses = filter(lambda x: x is not None and
        #                            self.word_utils.points(x) > parents_points_min and
        #                            x not in self.population,
        #                  [self.word_utils.longest_valid_prefix(self._random_mutate(f)) for f in fetuses]
        #                  )
        # children.extend(set(fetuses))
        # print(mother, father, children)

        for _ in range(max(0, attempts_count - len(children))):
            fetus = zipper.zip_randomly()
            # fetus = self._connect_randomly(mother, father)
            fetus = self._random_mutate(Word(fetus))
            fetus = self.word_utils.longest_valid_prefix(fetus)
            if fetus is not None and self.word_utils.points(fetus) > parents_points_min and \
                    fetus != mother and fetus != father and fetus not in children:
                print(f'SUCCESSFUL MUTATION: {mother} x {father} -> {fetus};', file=sys.stderr)
                children.append(fetus)

        if len(children) == 0:
            raise self.CannotMakeValidChildren('Failed to make children in given number of attempts')
        return children

    def _reproduce(self, selected_for_reproduction: List[Word]) -> int:
        failed = 0
        if len(selected_for_reproduction) <= 1:
            raise ValueError('Cannot reproduce one individual')
        if (len(selected_for_reproduction) - 1) % 2 == 0:
            selected_for_reproduction = selected_for_reproduction[:-1]

        for mother, father in grouped_by_2(selected_for_reproduction):
            try:
                children = self._cross(mother, father, attempts_count=8)
                self.population.extend(children)
            except self.CannotMakeValidChildren:
                failed += 1

        self.population = list(set(self.population))
        self.population.sort(key=lambda i: self.word_utils.points(i), reverse=True)
        np.random.shuffle(self.population[int(self.max_population_size / 3):])
        self.population = self.population[:self.max_population_size]

        return failed

    def run_algorithm(self):
        end_time = time.time() + self.max_time
        i = 0
        while time.time() < end_time:
            self._evolve()
            i += 1

        print(f'Initial population: {self.start_population}', file=sys.stderr)
        print(f'Initial population costs: {[self.word_utils.points(i) for i in self.start_population]}', file=sys.stderr)
        print(f'iterations = {i}', file=sys.stderr)
        return self.solution
