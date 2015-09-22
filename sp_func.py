#!/usr/bin/env python3

import numpy as np
from functools import lru_cache
import matplotlib.pyplot as plt
import warnings

# raise warnings as exceptions to catch RuntimeWarning
warnings.filterwarnings('error')


def get_basis_func(knot_seq, j, n=3, max_cache=16):
    """
    Return the basis function N^n_j for the knot sequence argument.

    Args:
        knot_seq: Knot sequence.
        j: index
        n: Optional argument. Degree of spline. (default=3)
        max_cache: Optional argument. Number of cached function returns
            for the recursive formula.

    """

    @lru_cache(maxsize=max_cache)
    def N(x, n, i):
        if n == 0:
            return (knot_seq[i] <= x) * (x < knot_seq[i+1])

        try:
            c1 = ((x - knot_seq[i]) /
                  (knot_seq[i + n] - knot_seq[i]))
        except (ZeroDivisionError, RuntimeWarning):
            c1 = 0.

        try:
            c2 = ((knot_seq[i + n + 1] - x) /
                  (knot_seq[i + n + 1] - knot_seq[i + 1]))
        except (ZeroDivisionError, RuntimeWarning):
            c2 = 0.

        return c1 * N(x, n - 1, i) + c2 * N(x, n - 1, i + 1)

    def basis_func(x):
        """
        Return basis function value N^{0}_{1}(x)
        """.format(n, j)

        # Avoid error with lru_cache when x is type list/ndarray.
        if hasattr(x, '__iter__'):
            return np.array([N(xi, n, j) for xi in x])

        return N(x, n, j)

    return basis_func


if __name__ == '__main__':
    knots = [0, 0, 1, 2, 3, 3, 4, 5, 5, 5]
    f = get_basis_func(knots, 1)

    x = np.linspace(0., 10, 50)
    fx = f(x)
    print(x)
    print(fx)
    plt.plot(x, f(x))
    plt.show()
