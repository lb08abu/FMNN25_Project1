"""
Test case for testing the Splines class.

Run by `nosetests`
"""

import unittest
import numpy as np
import splines


class SplinesTestCase(unittest.TestCase):

    # def setUp(self):
    #     pass

    def test_invalid_grid_values_greater_than_1(self):
        """
        Test that initialization raises a value error if values in `grid`
        are greater than 1.
        """
        self.assertRaises(ValueError, splines.Spline.__init__, splines.Spline,
                          [0, 0.5, 1.1], np.array([1, 2, 3, 4]))

    def test_invalid_grid_values_less_than_1(self):
        """
        Test that initialization raises a value error if values in `grid`
        are less than 0.
        """
        self.assertRaises(ValueError, splines.Spline.__init__, splines.Spline,
                          [-0.1, 0.5, 0.9], np.array([1, 2, 3, 4]))

    def test_deBoor_vs_blossom(self):
        ds = np.array([
                [ -20,     10],
                [ -20,     10],
                [ -20,     10],
                [ -50,     20],
                [ -25,      5],
                [-100,    -15],
                [ -25,    -65],
                [  10,    -80],
                [  60,    -30],
                [  10,     20],
                [  20,      0],
                [  40,     20],
                [  40,     20],
                [  40,     20]])

        x = np.linspace(0, 1, 150)
        s = splines.Spline(x, ds)
        self.assertEqual(s.eval_by_sum(), s.blossom())


if __name__ == '__main__':
    unittest.main()