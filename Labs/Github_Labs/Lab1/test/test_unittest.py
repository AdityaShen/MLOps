import sys
import os
import unittest

# Get the path to the project's root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from src import calculator


class TestCalculator(unittest.TestCase):

    def test_fun1(self):
        # average
        self.assertEqual(calculator.fun1(2, 3), 2.5)
        self.assertEqual(calculator.fun1(5, 0), 2.5)
        self.assertEqual(calculator.fun1(-1, 1), 0)
        self.assertEqual(calculator.fun1(-1, -1), -1)

    def test_fun2(self):
        # absolute difference
        self.assertEqual(calculator.fun2(2, 3), 1)
        self.assertEqual(calculator.fun2(5, 0), 5)
        self.assertEqual(calculator.fun2(-1, 1), 2)
        self.assertEqual(calculator.fun2(-1, -1), 0)

    def test_fun3(self):
        # power
        self.assertEqual(calculator.fun3(2, 3), 8)
        self.assertEqual(calculator.fun3(5, 0), 1)
        self.assertEqual(calculator.fun3(-1, 1), -1)
        self.assertEqual(calculator.fun3(-1, 2), 1)

    def test_fun4(self):
        # max of three
        self.assertEqual(calculator.fun4(2, 3, 5), 5)
        self.assertEqual(calculator.fun4(5, 0, -1), 5)
        self.assertEqual(calculator.fun4(-1, -1, -1), -1)
        self.assertEqual(calculator.fun4(-1, -1, 100), 100)


if __name__ == '__main__':
    unittest.main()
