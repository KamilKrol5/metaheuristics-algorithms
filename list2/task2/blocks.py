import sys
from typing import MutableSequence

import numpy as np


class _Block:
    def __init__(self, x_start, y_start, x_length, y_length, value_inside):
        self.x_start = x_start
        self.y_start = y_start
        self.x_length = x_length
        self.y_length = y_length
        self.value_inside = value_inside

    def __hash__(self):
        return hash((self.x_start, self.y_start))

    def __str__(self):
        return f'<(x,y) = ({self.x_start},{self.y_start}), {self.x_length} x {self.y_length}>'

    def __repr__(self):
        return str(self)

    def contains(self, x, y):
        return self.x_start <= x < self.x_start + self.x_length and \
               self.y_start <= y < self.y_start + self.y_length


class _BlockInSpace(_Block):
    DIRECTIONS = {
        'U': (0, 1),
        'R': (1, 0),
        'D': (0, -1),
        'L': (-1, 0),
    }

    def __init__(self, x_start, y_start, x_length, y_length, value_inside, space_of_blocks):
        super().__init__(x_start, y_start, x_length, y_length, value_inside)
        self.space_of_blocks: MutableSequence[_BlockInSpace] = space_of_blocks

    def is_space_valid(self, x, y, x_start=0, y_start=0):
        arr = np.full((x, y), 1, dtype=int)
        for block_ in self.space_of_blocks:
            end_x_index = block_.x_start + block_.x_length
            end_y_index = block_.y_start + block_.y_length
            arr[
                block_.x_start:end_x_index,
                block_.y_start:end_y_index
            ] = 0

        # print(f'Control sum = {np.sum(arr)}')
        return np.sum(arr[x_start:, y_start:])

    def neighbours_in_direction(self, direction):
        neighbours = set()
        if direction[0] == 0:  # U, D
            for x in range(self.x_start, self.x_start + self.x_length, 1):
                y = self.y_start
                if direction[1] == -1:
                    y = y + self.y_length
                for b in self.space_of_blocks:
                    if b.contains(x, y - direction[1]):
                        neighbours.add(b)
                        break

        elif direction[1] == 0:  # L, R
            for y in range(self.y_start, self.y_start + self.y_length, 1):
                x = self.x_start
                if direction[0] == 1:
                    x = x + self.x_length
                for b in self.space_of_blocks:
                    if b.contains(x + direction[0], y):
                        neighbours.add(b)
                        break

        return neighbours

    def get_neighbours(self):
        neighbours = set()
        for d in self.DIRECTIONS.values():
            new_neighbours = self.neighbours_in_direction(d)
            neighbours.update(new_neighbours)
        return neighbours

    def can_expand_in_direction(self, direction, min_block_x_size, min_block_y_size):
        neighbours_in_direction = self.neighbours_in_direction(direction)
        if len(neighbours_in_direction) != 1:
            return False, None
        neighbour = neighbours_in_direction.pop()
        if abs(direction[1]) == 0:
            return min_block_x_size < neighbour.x_length and neighbour.y_length == self.y_length, neighbour
        elif abs(direction[0] == 0):
            return min_block_y_size < neighbour.y_length and neighbour.x_length == self.x_length, neighbour

    """ Returns neighbour in the given direction which can be merged with self-block.
        If there is no possible candidate for merge, the method returns None.
    """

    def can_merge_in_direction(self, direction):
        neighbours_on_direction = self.neighbours_in_direction(direction)
        if len(neighbours_on_direction) != 1:
            return None
        neighbour = neighbours_on_direction.pop()
        if neighbour.value_inside != self.value_inside:
            return None
        if abs(direction[1]) == 0 and neighbour.y_length != self.y_length:
            return None
        elif abs(direction[0] == 0) and neighbour.x_length != self.x_length:
            return None
        return neighbour

    """ Merges given 'other' block with self-block if possible. 
        Returns bool value telling if merge was performed. Modifies 'other_blocks' collection.
    """

    def merge_in_direction(self, direction):
        other = self.can_merge_in_direction(direction)
        if other is None:
            return False
        up_or_down, left_or_right = direction[0] == 0, direction[1] == 0

        # specific situations and error handling
        if up_or_down and left_or_right:
            raise ValueError('Merging in two directions at once is not supported')
        # print('BEFORE')
        # print([str(bl) for bl in self.space_of_blocks])
        # merging itself
        if up_or_down:
            if direction[1] == -1:  # down
                self.y_length += other.y_length
            elif direction[1] == 1:  # up
                self.y_length += other.y_length
                self.x_start = other.x_start
                self.y_start = other.y_start
        elif left_or_right:
            if direction[0] == -1:  # left
                self.x_length += other.x_length
                self.x_start = other.x_start
                self.y_start = other.y_start
            elif direction[0] == 1:  # right
                self.x_length += other.x_length
        else:
            print(f'Warning: Cannot merge in (0,0) direction', file=sys.stderr)
            return False

        self.space_of_blocks.remove(other)
        # print('AFTER')
        # print([str(bl) for bl in self.space_of_blocks])
        return True

    """ Extends self-block and reduces the neighbour size by given value.
        There is an assumption that block provided as neighbour, actually is a neighbour in the provided direction.
        
        Args:
            neighbour (_BlockInSpace): neighbour (on provided direction side) of block to be extended.
            It is not checked if block provided as neighbour actually is the neighbour.
            direction (Tuple[int, int]): The direction in which extension is to be performed.
            difference (int): the increase of self-block size. Simultaneously it is the decrease of neighbour.
            
        Returns:
            bool: True if extension was performed successfully, False otherwise.
    """

    def expand_towards_neighbour(self, neighbour: '_BlockInSpace', direction, difference):
        if direction not in self.DIRECTIONS.values():
            print(f'Warning: Extension in the direction not present in DIRECTIONS dictionary is not supported.',
                  file=sys.stderr)
            return False
        if self == neighbour:
            print(f'Cannot merge with itself', file=sys.stderr)
            return False
        if direction[0] == 0:  # U or D
            self.y_length += difference
            neighbour.y_length -= difference
            if direction[1] == -1:  # D
                neighbour.y_start += difference
            elif direction[1] == 1:  # U
                self.y_start -= difference
        elif direction[1] == 0:  # L or R
            self.x_length += difference
            neighbour.x_length -= difference
            if direction[0] == -1:  # L
                self.x_start -= difference
            elif direction[0] == 1:  # R
                neighbour.x_start += difference
        return True

    """ Returns 2-tuple of bool values. First value tells if the block is x-splittable. 
        Second: if the block is y-splittable.
    """

    def can_split_in_two(self, min_x_length, min_y_length):
        return self.x_length >= 2 * min_x_length, self.y_length >= 2 * min_y_length

    def _split_in_two_on_x(self):
        print('BEFORE')
        print([str(bl) for bl in self.space_of_blocks])
        new_block_x_start = self.x_start + int(np.ceil(self.x_length / 2))
        new_block_y_start = self.y_start
        new_block_x_length = self.x_length // 2
        new_block_y_length = self.y_length
        self.x_length = int(np.ceil(self.x_length / 2))
        self.space_of_blocks.append(_BlockInSpace(
            new_block_x_start,
            new_block_y_start,
            new_block_x_length,
            new_block_y_length,
            self.value_inside,
            self.space_of_blocks
        ))
        print('AFTER')
        print([str(bl) for bl in self.space_of_blocks])

    def _split_in_two_on_y(self):
        print('BEFORE')
        print([str(bl) for bl in self.space_of_blocks])
        new_block_x_start = self.x_start
        new_block_y_start = self.y_start + int(np.ceil(self.y_length / 2))
        new_block_y_length = self.y_length // 2
        new_block_x_length = self.x_length
        self.y_length = int(np.ceil(self.y_length / 2))
        self.space_of_blocks.append(_BlockInSpace(
            new_block_x_start,
            new_block_y_start,
            new_block_x_length,
            new_block_y_length,
            self.value_inside,
            self.space_of_blocks
        ))
        print('AFTER')
        print([str(bl) for bl in self.space_of_blocks])

    """ Splits the block in two. If axis is 'x' then the block is split towards x axis. Analogously for axis being 'y'.
        Returns True if split was successful, False otherwise.
    """

    def split_in_two(self, axis: str, min_x_length=0, min_y_length=0):
        can_split_on_x, can_split_on_y = self.can_split_in_two(min_x_length, min_y_length)
        if axis == 'x':
            if not can_split_on_x:
                # print(f'Warning: Cannot split block {self} on x.', file=sys.stderr)
                return False
            self._split_in_two_on_x()
        elif axis == 'y':
            if not can_split_on_y:
                # print(f'Warning: Cannot split block {self} on y.', file=sys.stderr)
                return False
            self._split_in_two_on_y()
        else:
            raise ValueError(f'Split can be performed only for axis "x" or "y". Invalid axis: {axis}')

        return True
