from copy import copy
from unittest import TestCase
from .task2 import _BlockInSpace


class TestBlockInSpace(TestCase):

    def setUp(self):
        self.blocks = []
        self.block = _BlockInSpace(100, 200, 100, 200, value_inside=255, space_of_blocks=self.blocks)
        self.on_right = _BlockInSpace(200, 200, 100, 150, value_inside=255, space_of_blocks=self.blocks)
        self.on_right_bottom = _BlockInSpace(200, 350, 100, 100, value_inside=255, space_of_blocks=self.blocks)
        self.on_bottom = _BlockInSpace(100, 400, 100, 50, value_inside=255, space_of_blocks=self.blocks)
        self.on_left = _BlockInSpace(50, 200, 50, 200, value_inside=255, space_of_blocks=self.blocks)
        self.on_left_bottom = _BlockInSpace(50, 400, 50, 50, value_inside=255, space_of_blocks=self.blocks)
        self.blocks.extend([
            self.block,
            self.on_right,
            self.on_right_bottom,
            self.on_bottom,
            self.on_left,
            self.on_left_bottom
        ])
        self.block_neighbours = [self.on_right, self.on_right_bottom, self.on_bottom, self.on_left]
        self.total_neighbours_count = len(self.blocks) - 2

    def test_neighbours_in_direction(self):
        bottom_neighbours = self.block.neighbours_in_direction(self.block.DIRECTIONS['D'])
        self.assertIn(self.on_bottom, bottom_neighbours)
        self.assertEqual(len(bottom_neighbours), 1)

        right_neighbours = self.block.neighbours_in_direction(self.block.DIRECTIONS['R'])
        self.assertIn(self.on_right, right_neighbours)
        self.assertIn(self.on_right_bottom, right_neighbours)
        self.assertEqual(len(right_neighbours), 2)

        left_neighbours = self.block.neighbours_in_direction(self.block.DIRECTIONS['L'])
        self.assertIn(self.on_left, left_neighbours)
        self.assertEqual(len(left_neighbours), 1)

    def test_get_neighbours(self):
        neighbours = self.block.get_neighbours()
        self.assertEqual(len(neighbours), self.total_neighbours_count)
        for bl in self.block_neighbours:
            if bl != self.block:
                self.assertIn(bl, neighbours)
            else:
                self.assertNotIn(bl, neighbours)

    def test_can_expand_in_direction(self):
        result, candidate = self.block.can_expand_in_direction(self.block.DIRECTIONS['D'], 20, 20)
        self.assertTrue(result)
        self.assertEqual(candidate, self.on_bottom)
        result, candidate = self.block.can_expand_in_direction(self.block.DIRECTIONS['D'], 100, 20)
        self.assertTrue(result)
        self.assertEqual(candidate, self.on_bottom)
        result, _ = self.block.can_expand_in_direction(self.block.DIRECTIONS['D'], 20, 50)
        self.assertFalse(result)
        result, _ = self.block.can_expand_in_direction(self.block.DIRECTIONS['D'], 100, 50)
        self.assertFalse(result)

        result, candidate = self.block.can_expand_in_direction(self.block.DIRECTIONS['L'], 20, 20)
        self.assertTrue(result)
        self.assertEqual(candidate, self.on_left)
        result, candidate = self.block.can_expand_in_direction(self.block.DIRECTIONS['L'], 20, 200)
        self.assertTrue(result)
        self.assertEqual(candidate, self.on_left)
        result, _ = self.block.can_expand_in_direction(self.block.DIRECTIONS['L'], 50, 20)
        self.assertFalse(result)
        result, _ = self.block.can_expand_in_direction(self.block.DIRECTIONS['L'], 50, 200)
        self.assertFalse(result)

        result, _ = self.block.can_expand_in_direction(self.block.DIRECTIONS['R'], 20, 20)
        self.assertFalse(result)
        result, _ = self.block.can_expand_in_direction(self.block.DIRECTIONS['R'], 20, 150)
        self.assertFalse(result)
        result, _ = self.block.can_expand_in_direction(self.block.DIRECTIONS['R'], 100, 20)
        self.assertFalse(result)
        result, _ = self.block.can_expand_in_direction(self.block.DIRECTIONS['R'], 100, 150)
        self.assertFalse(result)

    def test_can_merge_in_direction(self):
        self.assertTrue(self.block.can_merge_in_direction(self.block.DIRECTIONS['D']))

        self.assertTrue(self.block.can_merge_in_direction(self.block.DIRECTIONS['L']))

        self.assertFalse(self.block.can_merge_in_direction(self.block.DIRECTIONS['R']))

    def test_merge_in_direction(self):
        blocks_size = len(self.blocks)
        is_successful = self.block.merge_in_direction(self.block.DIRECTIONS['D'])
        self.assertTrue(is_successful)
        self.assertEqual(blocks_size, len(self.block.space_of_blocks) + 1)
        is_successful = self.block.merge_in_direction(self.block.DIRECTIONS['L'])
        self.assertFalse(is_successful)
        self.assertEqual(blocks_size - 1, len(self.block.space_of_blocks))

    def test_extend_towards_neighbour(self):
        is_successful = self.block.extend_towards_neighbour(self.on_left, (1, 1), 10)
        self.assertFalse(is_successful)
        old_block = copy(self.block)
        old_left = copy(self.on_left)
        is_successful = self.block.extend_towards_neighbour(self.on_left, self.block.DIRECTIONS['L'], 2)
        self.assertTrue(is_successful)

        # y did not change
        self.assertEqual(self.block.y_length, old_block.y_length)
        self.assertEqual(self.block.y_start, old_block.y_start)
        self.assertEqual(self.on_left.y_length, old_left.y_length)
        self.assertEqual(self.on_left.y_start, old_left.y_start)
        # x changed
        self.assertEqual(self.block.x_length, old_block.x_length + 2)
        self.assertEqual(self.block.x_start, old_block.x_start - 2)
        self.assertEqual(self.on_left.x_length, old_left.x_length - 2)
        self.assertEqual(self.on_left.x_start, old_left.x_start)
