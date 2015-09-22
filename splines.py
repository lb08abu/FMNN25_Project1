"""
Class to represent splines for 2D curve design.

To design a Spline we need:
 - Control points
 - Knots
 - Coefficients (one for each control point)

Calculate all basis functions N^k_i(u)
"""

from __future__ import division
from functools import partial

import math
import matplotlib.pyplot as plt
import numpy as np



class DValuesError(Exception):
    pass


class Spline(object):
    def __init__(self, grid, dvalues, degree=3):
        """
        Initiates all instance variables. 
        Calculates equidistant knot points.
        """

        if not isinstance(grid, np.ndarray):
            raise TypeError("grid must be a numpy array")
        if not isinstance(dvalues, np.ndarray):
            raise TypeError("dvalues must be a numpy array")

        if max(grid) > 1.0:
            raise ValueError("grid values must be in range 0 to 1")
        if min(grid) < 0.0:
            raise ValueError("grid values must be in range 0 to 1")
        if (grid != sorted(grid)).any():
            raise ValueError("grid values must be in ascending order")

        # check shape of dvalues
        try:
            if dvalues.shape[1] != 2:
                raise DValuesError("expected dvalues shape as (:, 2)")
        except IndexError:
            raise DValuesError("expected dvalues shape as (:, 2)")

        # Set degree
        self.degree = degree

        # Save control points
        self.ds = dvalues
        self.nbr_ds = len(dvalues)

        # Find equidistant knot points
        nbr_knots = self.degree + self.nbr_ds  # add 2 on each end
        self.nbr_knots = nbr_knots
        self.grid = grid
        len_grid = len(grid)
        indices = math.ceil(len_grid / nbr_knots) * np.arange(nbr_knots)
        self.us = grid[indices.astype(int)]
        self.us[-1] = grid[-1]  # Otherwise the end point might be excluded

        # attribute for deBoor points
        self.deBoor_points = []

        # Apparently the values of the knots can be arbitrary. 
        # The only thing that matters is that we have a enough of them.

    def __call__(self, u):
        """
        This returns a point s(u) = [s_x(u) s_y(u)]
        """
        idx = self.us.searchsorted(u)
        return self.d(idx, u, 3)

    def plot(self, plot_control_poly=False, plot_deBoor_points=False):
        """
        Plots the whole spline
        """
        plt.figure()
        if plot_control_poly:
            plt.plot(self.ds[:, 0], self.ds[:, 1], 'g', label='Control polygon')

        points = self.blossom()
        plt.plot(points[:, 0], points[:, 1], 'b--',
                 label='%s-order spline' % self.degree)

        if plot_deBoor_points:
            dp = np.array(self.deBoor_points)
            plt.scatter(dp[:, 0], dp[:, 1], c='r', marker='x',
                        label='deBoor points')

        plt.legend(loc='best')
        plt.show()

    def get_basis_func(self, us, i):  # needs checking
        """
        Task 3
        """
        return partial(self.N3, us=us, i=i)

    def N0(self, us, i, x):
        """
        Base case for deBoor's algorithm
        """
        return (self.us[i] <= x) * (x < self.us[i + 1])

    def N(self, us, i, x, n):
        """
        deBoor's algorithm for finding basis functions. 
        Not memoized
        """
        if n == 0:
            return self.N0(us, i, x)
        else: 
            c1 = (x - us[i]) / (us[i + n] - us[i])
            c2 = (us[i + n + 1] - x) / (us[i + n + 1] - us[i + 1])
            return c1*self.N(us, i, x, n - 1) + c2*self.N(us, i + 1, x, n - 1)

    def N2(self, us, i, x, n, memo): 
        """
        deBoor's algorithm for finding basis functions. 
        Memoized. Stores previous calculations in a dictionary. 
        The function returns a new dictionary at every return. 
        This is avoided in N3

        memoized, returns new dictionaries all the time
        """
        if n == 0:
            if (i, 0) not in memo:
                memo[(i, 0)] = self.N0(us, i, x)
            return memo
        else: 
            c1 = (x - us[i]) / (us[i + n] - us[i])
            c2 = (us[i + n + 1] - x) / (us[i + n + 1] - us[i + 1])

            if (i, n - 1) not in memo:
                memo = self.N2(us, i, x, n - 1, memo)

            memo = self.N2(us, i + 1, x, n - 1, memo)
            memo[(i, n)] = c1 * memo[(i, n - 1)] + c2 * memo[(i + 1, n - 1)]

            return memo

    def N3(self, us, i, x, n, memo): 
        """
        deBoor's algorithm for finding basis functions. 
        Memoized. Stores previous calculations in a dictionary. 
        The function utilizes that the dictonary object is mutable.
        Hence it has no return value.
        """
        if n == 0:
            if (i, 0) not in memo:
                memo[(i, 0)] = self.N0(us, i, x)
        else: 
            c1 = (x - us[i]) / (us[i + n] - us[i])
            c2 = (us[i + n + 1] - x) / (us[i + n + 1] - us[i + 1])

            if (i, n - 1) not in memo:
                self.N3(us, i, x, n - 1, memo)
            
            self.N3(us, i + 1, x, n - 1, memo)
            memo[(i, n)] = c1 * memo[(i, n - 1)] + c2 * memo[(i + 1, n - 1)]

    def d(self, i, x, n): 
        """
        Blossom algorithm. Not memoized.
        """
        if n == 0:
            return self.ds[i, :]
        else:
            us = self.us
            a = (x - us[i]) / (us[i + self.degree + 1 - n] - us[i])
            return (1 - a) * self.d(i - 1, x, n - 1) + a * self.d(i, x, n - 1)

    def blossom(self):
        """
        Runs the entire blossom algorithm for the given data at object creation.
        deBoor points are saved as list items in the attribute `deBoor_points`.
        """
        grid = self.grid
        points = []

        if self.deBoor_points:
            self.deBoor_points = []  # erase previous deBoor points calculations

        for i in np.arange(3, self.nbr_knots - 3):  # Avoid the 3 dummy points
            ui0 = self.us[i]
            ui1 = self.us[i+1]
            x = grid[(ui0 <= grid) & (grid <= ui1)]
            points.extend(self.d(i, x.reshape(len(x), 1), 3))

            if not self.deBoor_points:
                self.deBoor_points.append(points[0])  # add first point
            self.deBoor_points.append(points[-1])

        return np.array(points)

    def eval_by_sum(self):
        """
        Needed for task 4. Evaluates the spline by using the sum approach
        instead of Blossoms algorithm
        """
        grid = self.grid  # xi
        nbr_ds = self.nbr_ds

        # Calculate "vandermonde like" matrix
        memo = {}
        l_grid = len(grid)
        N = np.zeros((l_grid, nbr_ds))
        for i in np.arange(nbr_ds - 1):  # -1 due to recursion i + 1 in N3?
            self.N3(self.us, i, grid, 3, memo)
            N[:, i] = memo[(i, 3)]

        # Calculate sum
        xs = np.dot(N, self.ds[:, 0])
        ys = np.dot(N, self.ds[:, 1])
        
        # Possible solution? Throw away points
        # xs = xs[20:-20]
        # ys = ys[20:-20]
        
        return (xs, ys)


def main():
    plt.close("all")
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
    s = Spline(x, ds)
    s.plot(plot_control_poly=True, plot_deBoor_points=True)

    """
    This test shows that:
    N much slower than N2
    a = np.arange(10)
    s = splines(a, a)
    %timeit t1 = s.N(0, x, 10) # 49.7 ms
    %timeit t2 = s.N2(0, x, 10, {})[(0,10)] #2.24 ms
    t3 = {}
    %timeit s.N3(0, x, 10, t3) # 954 ns!

    np.all(t1 == t2) * all(t1 == t3[(0,10)])
    """

    # test d
    # plt.figure()
    # plt.plot(ds[:,0], ds[:,1])

    # I = arange(3, 14)
    # I_len = len(I)

    # for i in I:
    #     x = np.linspace(i, i + 1 ,100)
    #     p = 3  # min(i, 3)
    #     # p = min(min(i, 3), I_len - i - 2)
    #     points = s.d(i, x, p)
    #     # print("i = {}, p = {}, I_len - i = {}".format(i, p, I_len - i - 1))
    #     plt.plot(points[:,0],points[:,1])

    # Test eval_by_sum
    xs, ys = s.eval_by_sum()
    plt.plot(xs, ys)
    plt.show()


if __name__ == '__main__':
    main()
