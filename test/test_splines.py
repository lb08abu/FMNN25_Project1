"""
FMNN25 Project 1 - Splines

test_splines.py
Test case for testing the Splines class.
Execute by running `nosetests` from the command line in the parent directory.

Contributors:
Lewis Belcher
Angelos Toytziaridis
Simon Ydman
"""

import unittest
import numpy as np
import splines


class SplinesTestCase(unittest.TestCase):

    def test_invalid_grid_values_greater_than_1(self):
        """
        Test that initialization raises a value error if values in `grid`
        are greater than 1.
        """
        self.assertRaises(ValueError, splines.Spline.__init__, splines.Spline,
                          np.array([0, 0.5, 1.1]), np.array([1, 2, 3, 4]))

    def test_invalid_grid_values_less_than_1(self):
        """
        Test that initialization raises a value error if values in `grid`
        are less than 0.
        """
        self.assertRaises(ValueError, splines.Spline.__init__, splines.Spline,
                          np.array([-0.1, 0.5, 0.9]), np.array([1, 2, 3, 4]))

    def test_invalid_grid_values_non_ascending(self):
        """
        Test that initialization raises a value error if values in `grid`
        are not in ascending order.
        """
        self.assertRaises(ValueError, splines.Spline.__init__, splines.Spline,
                          np.array([0.1, 0.5, 0.4]), np.array([1, 2, 3, 4]))

    def test_non_2D_dvalues_array(self):
        """
        Test that initialization raises an error if dvalues is not 2D.
        """
        self.assertRaises(splines.DValuesError, splines.Spline.__init__,
                          splines.Spline, np.array([0.1, 0.5, 0.9]),
                          np.array([1, 2, 3, 4]))

    def test_non_numpy_array_raises_error_for_grid(self):
        """
        Test that initialization raises an error if grid is not passed as a
        numpy array.
        """
        self.assertRaises(TypeError, splines.Spline.__init__,
                          splines.Spline, [0.1, 0.5, 0.9],
                          np.array([1, 2, 3, 4]))

    def test_non_numpy_array_raises_error_for_dvalues(self):
        """
        Test that initialization raises an error if dvalues is not passed as a
        numpy array.
        """
        self.assertRaises(TypeError, splines.Spline.__init__,
                          splines.Spline, np.array([0.1, 0.5, 0.9]),
                          [1, 2, 3, 4])

    def test_deBoor_vs_blossom(self):
        """
        Test that the returned array from blossoms is the same as from
        the deBoor algorithm (it currently isn't so it seems that our
        implementation is wrong somewhere...)
        """
        ds = np.array([
                [ -20,     10],
                [ -50,     20],
                [ -25,      5],
                [-100,    -15],
                [ -25,    -65],
                [  10,    -80],
                [  60,    -30],
                [  10,     20],
                [  20,      0],
                [  40,     20]])

        x = np.linspace(0., 1., 150)
        s = splines.Spline(x, ds)
        self.assertEqual(s.eval_by_sum(), s.blossom())


if __name__ == '__main__':
    unittest.main()