#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import collections
import functools


class memoized(object):
   '''Decorator. Caches a function's return value each time it is called.
   If called later with the same arguments, the cached value is returned
   (not reevaluated).
   '''
   def __init__(self, func):
      self.func = func
      self.cache = {}
   def __call__(self, *args):
      if not isinstance(args, collections.Hashable):
         # uncacheable. a list, for instance.
         # better to not cache than blow up.
         return self.func(*args)
      if args in self.cache:
         return self.cache[args]
      else:
         value = self.func(*args)
         self.cache[args] = value
         return value
   def __repr__(self):
      '''Return the function's docstring.'''
      return self.func.__doc__
   def __get__(self, obj, objtype):
      '''Support instance methods.'''
      return functools.partial(self.__call__, obj)


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
    
    @memoized
    def N(x, n, i):
        if n == 0:
            return (knot_seq[i] <= x) * (x < knot_seq[i+1])

        c1, c2 = 0, 0
        if (knot_seq[i + n] - knot_seq[i]):
            c1 = (x - knot_seq[i]) / \
                 (knot_seq[i + n] - knot_seq[i])

        if (knot_seq[i + n + 1] - knot_seq[i + 1]):
            c2 = (knot_seq[i + n + 1] - x) / \
                 (knot_seq[i + n + 1] - knot_seq[i + 1])

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
    knots = [0, 0, 1, 2, , 3, 4, 5, 5, 5]
    f = get_basis_func(knots, 1)

    x = np.linspace(0., 5, 1000)
    fx = f(x)
    plt.plot(x, fx)
    plt.show()
