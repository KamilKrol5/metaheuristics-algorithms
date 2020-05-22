from itertools import repeat, starmap
from typing import Generator, Tuple, TypeVar, Iterable

T = TypeVar('T')


def grouped_by_2(it: Iterable[T]) -> Generator[Tuple[T, T], None, None]:
    it = iter(it)
    while True:
        try:
            yield next(it), next(it)
        except StopIteration:
            # no more elements in the iterator
            return


def negate_bit(value: int, bit_index) -> int:
    return value ^ (1 << bit_index)


def flatten(collection):
    for element in collection:
        if isinstance(element, list):
            for x in flatten(element):
                yield x
        else:
            yield element


def zip_longest(*args, fill_values_generator, generator_args=None):
    """
    Implementation of zim_longest from itertools but with ability to provide function generating fill values.
    """
    if generator_args is None:
        generator_args = []
    iterators = [iter(it) for it in args]
    num_active = len(iterators)
    if not num_active:
        return
    while True:
        values = []
        for i, it in enumerate(iterators):
            try:
                value = next(it)
            except StopIteration:
                num_active -= 1
                if not num_active:
                    return
                iterators[i] = starmap(fill_values_generator, repeat(generator_args))
                value = fill_values_generator(*generator_args)
            values.append(value)
        yield tuple(values)
