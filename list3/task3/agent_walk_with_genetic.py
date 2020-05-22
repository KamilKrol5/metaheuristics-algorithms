import sys
import time
from typing import List, Optional, Tuple
import numpy as np
from agent_walk import AgentWalk, Agent, Position, directions, EXIT
from path import Path
from utils.utils import grouped_by_2


class CannotMakeValidChild(Exception):
    pass


class Individual(Path):
    pass


class AgentWalkWithGenetic(AgentWalk):
    MINIMUM_POPULATION_SIZE = 2

    def __init__(self, board: np.ndarray, initial_solutions: List[Path], max_time: int, max_population_size: int):
        super().__init__(board, initial_solutions, max_time, max_population_size)
        self.rows, self.columns = board.shape
        self.child_making_attempts_count = 10
        self.mutation_probability = 0.0
        self.population.extend(initial_solutions)
        for i in range(max_population_size - len(self.population)):
            self.population.append(self._generate_random_individual())
        print(f'population size: {len(self.population)}. Maximum is: {max_population_size}', file=sys.stderr)

    @property
    def solution(self) -> Path:
        return sorted(self.population, key=lambda i: i.cost)[0]

    def _generate_random_individual(self) -> Path:
        walker = Agent(self.board)
        walker_start_pos: Position = walker.current_position

        while True:
            random_path = Path(np.random.choice(directions, size=self.rows * self.columns))
            walker.current_position = walker_start_pos
            path_, is_valid = self._validate_and_shorten_path(random_path, agent=walker)

            if is_valid and path_ not in self.population:
                return path_

    def _validate_and_shorten_path(self, path: Path, agent) -> Tuple[Optional[Path], bool]:
        walker = agent
        for i, direction in enumerate(path, 1):
            try:
                walker.move(direction, change_own_board=False)
                if self.board[walker.current_position.row, walker.current_position.column] == EXIT:
                    return Path(path[:i]), True
                for direction_ in directions:
                    if walker.look(direction_) == EXIT:
                        return Path(path[:i] + [direction_]), True

            except Agent.WallException:
                return None, False

        return None, False

    def _chance_for_reproduction(self, individual: Path) -> float:
        sum_of_costs = np.sum([path.cost for path in self.population])
        # print(1.0 - individual.cost / sum_of_costs)
        return 1.0 - individual.cost / sum_of_costs

    def _selection(self) -> List[Path]:
        selected: List[Path] = []
        while len(selected) < self.MINIMUM_POPULATION_SIZE:
            selected.clear()
            # print('len=', len(selected))
            for individual in self.population:
                if np.random.rand() < self._chance_for_reproduction(individual):
                    selected.append(individual)

        # print(f'Selected for reproduction: {len(selected)} of {len(self.population)}', file=sys.stderr)
        return selected

    def _reproduce(self, to_reproduce: List[Path]) -> int:
        """ Reproduces selected individuals. Returns number of failed making child attempts. """
        failed = 0
        to_remove = []
        children = []
        if len(to_reproduce) <= 1:
            raise ValueError('Cannot reproduce one individual')
        if (len(to_reproduce) - 1) % 2 == 0:
            to_reproduce = to_reproduce[:-1]

        for mother, father in grouped_by_2(to_reproduce):
            # print(f'Mother: {mother}; Father: {father}', file=sys.stderr)
            try:
                # mc, fc = mother.copy(), father.copy()
                child = self._try_for_child(mother, father, self.child_making_attempts_count)
                # assert mc == mother and fc == father
                if mother.cost < father.cost:
                    to_kill = father
                else:
                    to_kill = mother
                to_remove.append(to_kill)
                children.append(child)
            except CannotMakeValidChild:
                failed += 1

        # print(f'Removing: {to_remove} from {self.population}', file=sys.stderr)
        for dead in to_remove:
            self.population.remove(dead)
        # print(f'Adding: {children} to {self.population}', file=sys.stderr)
        self.population.extend(children)
        # print(self.population, file=sys.stderr)
        # print(f'Failed child making attempts: {failed} out of {len(to_reproduce)}', file=sys.stderr)
        # print('---', file=sys.stderr)
        return failed

    def _mutate(self, individual: Path) -> bool:
        if np.random.rand() < self.mutation_probability:
            # before = individual.copy()
            # if np.random.rand() < 0.5:
            #     i = np.random.randint(0, len(individual) // 2+1)
            #     j = np.random.randint(len(individual) // 2+1, len(individual))
            #     Path(individual).make_inversion(i, j)

            rand_index = np.random.randint(0, len(individual))
            mutated_direction_index = directions.index(individual[rand_index]) - np.random.choice([1, 2, 3])
            mutated_direction = directions[mutated_direction_index]
            individual[rand_index] = mutated_direction

            # print(f'MUTATION successful: {before} -> {individual}', file=sys.stderr)
            return True
        return False

    def _try_for_child(self, parent1: Path, parent2: Path, attempts_count) -> Path:
        walker = Agent(self.board)
        walker_start_pos: Position = walker.current_position

        for _ in range(attempts_count):
            possible_children = self._cross(parent1, parent2)
            for possible_child in possible_children:
                self._mutate(possible_child)
                walker.current_position = walker_start_pos
                path_, is_valid = self._validate_and_shorten_path(possible_child, agent=walker)
                if is_valid:
                    # print(f'NEW CHILD: {path_}; parents: {parent1}, {parent2};', file=sys.stderr)
                    return path_

        else:
            raise CannotMakeValidChild(f'Failed to make child. Number of attempts: {attempts_count}')

    @staticmethod
    def _cross(parent1, parent2) -> Tuple[Path, Path]:
        index = np.random.randint(1, min(len(parent1), len(parent2)))
        return parent1[:index] + parent2[index:], parent2[:index] + parent1[index:]

    def _evolve(self) -> None:
        to_reproduce = self._selection()
        failed = self._reproduce(to_reproduce)
        if failed == len(to_reproduce):
            print(f'It was not possible to make a child for any of selected individuals.', file=sys.stderr)

    def run_genetic_algorithm(self, mutation_probability: float) -> Path:
        if not 0 <= mutation_probability <= 1:
            raise ValueError(f'Mutation probability must be a number between 0 and 1 (inclusive)')

        initial_population = self.population.copy()
        self.mutation_probability = mutation_probability
        end_time = time.time() + self.max_time
        while time.time() < end_time:
            self._evolve()

        print(f'Initial population: {initial_population}', file=sys.stderr)
        print(f'Initial population costs: {[ i.cost for i in initial_population]}', file=sys.stderr)
        return self.solution
