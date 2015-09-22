"""
FMNN25 Project 1 - Splines

sp_func.py
Contains functions for creating basis functions, also using optimisation.
"""

import collections
import numpy as np


class memoized:
    """
    Decorator. Caches a function's return value each time it is called.
    If called later with the same arguments, the cached value is returned
    (not reevaluated).
    """

    def __init__(self, func):
        self.func = func
        self.cache = {}

    def __call__(self, *args):
        if not isinstance(args, collections.Hashable):
            # uncacheable. a list, for instance.
            # better to not cache than blow up.
            return self.func(*args)

        if args in self.cache:  # Check if function call with args is in cache
            return self.cache[args]
        else:
            value = self.func(*args)
            self.cache[args] = value
            return value


def get_basis_func(knot_seq, j, n=3):
    """
    Return the basis function N^n_j for the knot sequence argument.

    :param knot_seq: list/numpy.ndarray, knot sequence
    :param j: int, index
    :param n: int, Optional argument. Degree of spline. (default=3)
    :return: callable, basis function to calculate basis N^n_j
    """

    @memoized  # Memoizes function.
    def N(x, n, i):
        """
        Recursive calculation of basis function values.
        """
        if n == 0:  # If degree == 0 return N^0_j(x)
            return (knot_seq[i] <= x) * (x < knot_seq[i+1])

        c1, c2 = 0, 0
        # Only evaluate c1 and c2 when the denominators are non-zero.
        if (knot_seq[i + n] - knot_seq[i]):
            c1 = (x - knot_seq[i]) / \
                 (knot_seq[i + n] - knot_seq[i])

        if (knot_seq[i + n + 1] - knot_seq[i + 1]):
            c2 = (knot_seq[i + n + 1] - x) / \
                 (knot_seq[i + n + 1] - knot_seq[i + 1])

        if not (c1 or c2):  # If c1 and c2 both 0, stop recursive calls.
            return 0
        return c1 * N(x, n - 1, i) + c2 * N(x, n - 1, i + 1)

    @np.vectorize  # Vectorizes function such that it can accept array inputs.
    def basis_func(x):
        """
        Return basis function value N^{0}_{1}(u)
        """.format(n, j)
        return N(x, n, j)

    return basis_func
