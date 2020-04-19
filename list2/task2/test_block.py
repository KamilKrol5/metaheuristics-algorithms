from unittest import TestCase
from task2 import _Block


class TestBlock(TestCase):
    def test_contains(self):
        block = _Block(105, 120, 10, 20, value_inside=255)
        self.assertTrue(block.contains(105, 120))
        self.assertTrue(block.contains(106, 139))
        self.assertTrue(block.contains(114, 120))
        self.assertTrue(block.contains(110, 125))
        self.assertFalse(block.contains(106, 141))
        self.assertFalse(block.contains(105, 119))
        self.assertFalse(block.contains(104, 125))
        self.assertFalse(block.contains(115, 120))
        self.assertFalse(block.contains(100, 145))
