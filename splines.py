"""
Class to represent splines for 2D curve design.

To design a Spline we need:
 - Control points
 - Knots
 - Coefficients (one for each control point)

Calculate all basis functions N^k_i(u)
"""


import numpy as np
import matplotlib.pyplot as plt
import scipy as sp


class Splines(object):
    def __init__(self, knots, dvalues):  # grid = u? se slide 11
        # self.n = len(knots)
        # self.m = len(dvalues)
        # self.p = self.m - self.n - 1
        # self.knots = knots

        self.degree = 3
        self.nbr_ds = len(dvalues)
        self.u_max  = self.degree + self.nbr_ds  # + 1?
        self.us     = np.arange(self.u_max + 1)  # i st. u?
        self.ds     = dvalues

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
        # if (self.knots[i] <= x < self.knots[i+1]):
        #     return 1
        # else:
        #     return 0
        return (i <= x) * (x < i + 1);

    def N(self, i, x, n):
        if n == 0:
            return self.N0(i, x)
        else: 
            us = self.us
            c1 = (x - us[i]) / (us[i + n] - us[i])
            c2 = (us[i + n + 1] - x) / (us[i + n + 1] - us[i + 1])
            return c1 * self.N(i, x, n - 1) + c2 * self.N(i + 1, x, n - 1)

    def N2(self, i, x, n, memo):  # memoized, returns new dictionaries all the time
        if n == 0:
            if (i, 0) not in memo:
                memo[(i, 0)] = self.N0(i,x)
            return memo
        else:
            if (i, n) in memo: #needed?
                return

            us = self.us
            c1 = (x - us[i]) / (us[i + n] - us[i])
            c2 = (us[i + n + 1] - x) / (us[i + n + 1] - us[i + 1])

            if (i, n - 1) not in memo:
                memo = self.N2(i, x, n - 1, memo)
            memo         = self.N2(i + 1, x, n - 1, memo)
            memo[(i, n)]     = c1 * memo[(i, n - 1)] + c2 * memo[(i + 1, n - 1)]

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
            return self.ds[i, :]
        else:
            us = self.us
            x = x.reshape(len(x), 1)
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
    print("Shape of ds: %s" % (ds.shape,))

    s = Splines(np.arange(5), ds)

    # test d
    plt.figure()
    plt.plot(ds[:, 0], ds[:, 1])

    I = np.arange(3, 14)
    I_len = len(I)

    for i in I:
        x = np.linspace(i, i + 1, 100)
        # p = min(i, 3)
        p = 3
        # p = min((i, 3, I_len - i))
        print("i = %s, Running bloom for degree: %s" % (i, p))
        points = s.d(i, x, p)
        plt.plot(points[:, 0], points[:, 1], '*')

    plt.show()


if __name__ == '__main__':
    main()