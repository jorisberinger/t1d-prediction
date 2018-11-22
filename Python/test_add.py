from unittest import TestCase


class TestAdd(TestCase):
    def test_add(self):
        from Python.readData import add
        self.assertEqual(add(2, 4), 2 + 4)

    def test_add_2(self):
        from Python.readData import add
        self.assertNotEqual(add(2, 5), 2 + 4)
