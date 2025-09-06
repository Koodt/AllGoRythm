import unittest
import os
from allgorythms.scripts.generate_dataset import generate_tags


class TestGenerateTags(unittest.TestCase):
    def test_generate_tags(self):
        path = os.path.join(
            "algorithms",
            "python",
            "sort",
            "insertion",
            "main.py",
        )
        expected = ["python", "sort", "insertion"]
        self.assertEqual(generate_tags(path), expected)
