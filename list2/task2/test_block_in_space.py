from unittest import TestCase
from .task2 import _BlockInSpace


class TestBlockInSpace(TestCase):
    blocks = []
    block = _BlockInSpace(100, 200, 100, 200, value_inside=255, space_of_blocks=blocks)
    on_right = _BlockInSpace(200, 200, 100, 150, value_inside=255, space_of_blocks=blocks)
    on_right_bottom = _BlockInSpace(200, 350, 100, 100, value_inside=255, space_of_blocks=blocks)
    on_bottom = _BlockInSpace(100, 400, 100, 50, value_inside=255, space_of_blocks=blocks)
    on_left = _BlockInSpace(50, 200, 50, 200, value_inside=255, space_of_blocks=blocks)
    blocks.extend([
        block,
        on_right,
        on_right_bottom,
        on_bottom,
        on_left,
    ])
    total_neighbours_count = len(blocks) - 1

    def test_neighbours_in_direction(self):
        bottom_neighbours = self.block.neighbours_in_direction(self.block.DIRECTIONS['D'])
        self.assertIn(self.on_bottom, bottom_neighbours)
        self.assertEqual(len(bottom_neighbours), 1)

        right_neighbours = self.block.neighbours_in_direction(self.block.DIRECTIONS['R'])
        self.assertIn(self.on_right, right_neighbours)
        self.assertIn(self.on_right_bottom, right_neighbours)
        self.assertEquals(len(right_neighbours), 2)

        left_neighbours = self.block.neighbours_in_direction(self.block.DIRECTIONS['L'])
        self.assertIn(self.on_left, left_neighbours)
        self.assertEqual(len(left_neighbours), 1)

    def test_get_neighbours(self):
        neighbours = self.block.get_neighbours()
        self.assertEqual(len(neighbours), self.total_neighbours_count)
        for bl in self.blocks:
            if bl != self.block:
                self.assertIn(bl, neighbours)
            else:
                self.assertNotIn(bl, neighbours)

    def test_can_expand_in_direction(self):
        pass

    #
    # def test_can_merge_in_direction(self):
    #     self.fail()
    #
    # def test__merge_in_direction(self):
    #     self.fail()
