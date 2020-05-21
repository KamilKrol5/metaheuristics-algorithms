from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Container, Dict, Optional

from dictionary_utils import read_dictionary


@dataclass
class Letter:
    letter: str
    points: int
    count: int

    def __str__(self):
        return f'{self.letter} [points={self.points}; count={self.count}]'


class Scrabble:
    def __init__(self, max_time: int, available_letters: Dict[str, Letter], dictionary: Iterable[str]):
        self.dictionary = dictionary
        self.max_time = max_time
        self.available_letters = available_letters


class Word(str):
    pass


@dataclass
class WordUtils:
    dictionary: Container[str]
    allowed_letters: Dict[str, Letter]

    def _are_all_letters_allowed(self, word: Word) -> bool:
        letters_in_word = Counter(word)
        for letter, count in letters_in_word.items():
            if letter in self.allowed_letters:
                if count > self.allowed_letters[letter]:
                    return False
            else:
                return False
        return True

    def is_word_valid(self, word: Word) -> bool:
        return self._are_all_letters_allowed(word) and word in self.dictionary

    def points(self, word: Word) -> int:
        return sum([self.allowed_letters[letter].points for letter in word])

    def longest_valid_prefix(self, word: Word) -> Optional[Word]:
        length = len(word)
        for i in range(1, length):
            prefix: Word = Word(word[:-i])
            if self.is_word_valid(prefix) and prefix in self.dictionary:
                return prefix
        else:
            return None


if __name__ == '__main__':
    print(len(read_dictionary()))
