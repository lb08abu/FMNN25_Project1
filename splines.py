"""
Class to represent splines for 2D curve design.

To design a Spline we need:
 - Control points
 - Knots
 - Coefficients (one for each control point)

Calculate all basis functions N^k_i(u)
"""


import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


class Splines(object):
    def __init__(self, knots, dvalues):  # grid = u? se slide 11
        self.n = len(knots)
        self.m = len(dvalues)
        self.p = self.m - self.n - 1
        self.knots = knots

        # self.degree = 3
        # self.nbr_ds = len(dvalues)
        # self.u_max  = self.degree + self.nbr_ds  # + 1?
        # self.us     = np.arange(self.u_max + 1)  # i st. u?
        # self.ds     = dvalues

    def __call__(self, u, *args, **kwargs):
        """
        :param u:
        :param args:
        :param kwargs:
        :return:
        """
        pass

    def plot(self):
        pass

    def N0(self, i, x):
        if (self.knots[i] <= x < self.knots[i+1]):
            return 1
        else:
            return 0

    def N(self, i, x, n):
        if n == 0:
            return self.N0(i, x)
        else:
            us = self.knots
            c1 = (x - us[i]) / (us[i + n] - us[i])
            c2 = (us[i + n + 1] - x) / (us[i + n + 1] - us[i + 1])
            return c1 * self.N(i, x, n - 1) + c2 * self.N(i + 1, x, n - 1)

    def N2(self, i, x, n, memo):  # memoized, returns new dictionaries all the time
        if n == 0:
            if (i, 0) not in memo:
                memo[(i, 0)] = self.N0(i,x)
            return memo
        else: 
            if (i, n) in memo:  # needed?
                return

            us = self.us
            c1 = (x - us[i]) / (us[i + n] - us[i])
            c2 = (us[i + n + 1] - x) / (us[i + n + 1] - us[i + 1])

            if (i, n - 1) not in memo:
                memo = self.N2(i, x, n - 1, memo)

            memo = self.N2(i + 1, x, n - 1, memo)
            memo[(i, n)] = c1 * memo[(i, n - 1)] + c2 * memo[(i + 1, n - 1)]

            return memo

    def N3(self, i, x, n, memo):  # memoized, utilizes that an dictonary is mutable like a C pointer
        if n == 0:
            if (i, 0) not in memo:
                memo[(i, 0)] = self.N0(i,x)
            return 
        else: 
            if (i, n) in memo: #needed?
                return

            us = self.us # reference or copy??
            c1 = (x - us[i]) / (us[i + n] - us[i])
            c2 = (us[i + n + 1] - x) / (us[i + n + 1] - us[i + 1])

            if (i, n - 1) not in memo:
                self.N3(i, x, n - 1, memo)
            
            self.N3(i + 1, x, n - 1, memo)
            memo[(i, n)] = c1 * memo[(i, n - 1)] + c2 * memo[(i + 1, n - 1)]
            return 

    def d(self, i, x, n):
        if n == 0:
            return self.ds[:, i]
        else:
            us = self.us
            a = (x - us[i]) / (us[i + self.degree + 1 - n] - us[i])
            return (1 - a) * self.d(i - 1, x , n - 1) + a * self.d(i, x, n - 1)


def memoize(f):
    memo = {}

    def helper(x):
        if x not in memo:            
            memo[x] = f(x)
        return memo[x]

    return helper


def main():

    plt.close("all")

    ds = np.array([
        [-20,   10],
        [-50,   20],
        [-25,    5],
        [-100, -15],
        [-25,  -65]
    ]).T

    print(ds)

    knots = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    s = Splines(knots, ds)

    x = np.linspace(0, 5, 100)
    # plt.scatter(ds[0, :], ds[1, :])
    N0 = s.N(0, x, 3)
    N1 = s.N(1, x, 3)
    N2 = s.N(2, x, 3)
    N3 = s.N(3, x, 3)
    plt.plot(x, N0)
    # plt.plot(x, N1)
    # plt.plot(x, N2)
    # plt.plot(x, N3)

    plt.show()


if __name__ == '__main__':
    main()