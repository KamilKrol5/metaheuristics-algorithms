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
