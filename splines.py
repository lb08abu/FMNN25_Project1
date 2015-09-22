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
import scipy as sp


class Spline(object):
    def __init__(self, grid, dvalues):
        """
        Initiates all instance variables. 
        Calculates equidistant knot points.
        Adds 3 copies at the start and end point of the control points and knot 
        points.
        """
        # Set degree
        self.degree     = 3

        # Save control points
        self.nbr_ds     = len(dvalues)
        ds              = np.zeros((self.nbr_ds + 6, 2))
        ds[3:-3, :]     = dvalues
        ds[:3, :]       = ds[3, :]
        ds[-3:, :]      = ds[-4, :]
        self.ds         = ds

        # Find equidistant knot points
        nbr_knots       = self.degree + self.nbr_ds  # add 2 on each end
        self.nbr_knots  = nbr_knots
        self.grid       = grid
        len_grid        = len(grid)
        indices         = math.ceil(len_grid / nbr_knots) * np.arange(nbr_knots)

        us              = np.zeros(self.nbr_knots +  6)
        tmp             = grid[indices.astype(int)]
        tmp[-1]         = grid[-1]
        us[3:-3]        = tmp
        us[:3]          = us[3]
        us[-3:]         = us[-4]
        self.us         = us

        # Apparently the values of the knots can be arbitrary. 
        # The only thing that matters is that we have a enough of them.

    def __call__(self, u):
        """
        This returns a point s(u) = [s_x(u) s_y(u)]
        """
        idx = self.us.searchsorted(u)
        i = idx
        return self.d(i, u, 3)

    def plot(self, plot_control_poly=False, plot_deBoor_points=False):
        """
        Plots the whole spline
        """
        plt.figure()
        if plot_control_poly:
            plt.plot(self.ds[:, 0], self.ds[:, 1])

        points = self.blossom()
        plt.plot(points[:, 0], points[:, 1], 'r')

        # if plot_deBoor_points:
        #     plt.plot(points[0, 0], points[0, 1], 'r*')
        #     plt.plot(points[-1, 0], points[-1, 1], 'r*')

        plt.show()

    def get_basis_func(self, us, i):  # needs checking
        """
        Task 3
        """
        return partial(self.N3, us=us, i=i)

    def N0(self, us, i, x):
        """
        Base case for deBoor's algorithm
        Note: returns 0 at endpoint when it should return 1.
        """
        return (self.us[i] <= x) * (x < self.us[i + 1]) #+ (x == self.us[self.nbr_knots - 7])

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

            # Set 0/0 = 0. Be careful of infinity? (a.k.a. constant / 0)
            with np.errstate(divide = 'ignore', invalid = 'ignore'):
                c1 = (x - us[i]) / (us[i + n] - us[i])
                c2 = (us[i + n + 1] - x) / (us[i + n + 1] - us[i + 1])
                c1 = np.nan_to_num(c1)
                c2 = np.nan_to_num(c2)

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

            # Set 0/0 = 0. Be careful of infinity? (a.k.a. constant / 0)
            with np.errstate(divide = 'ignore', invalid = 'ignore'):
                a = (x - us[i]) / (us[i + self.degree + 1 - n] - us[i])
                a = np.nan_to_num(a)
            
            return (1 - a) * self.d(i - 1, x, n - 1) + a * self.d(i, x, n - 1)

    def blossom(self): 
        """
        Runs the entire blossom algo for the given data at object creation.
        Not memoized.
        """
        grid = self.grid
        points = []
        for i in np.arange(3, self.nbr_knots + 3):  #why 3 and + 3 ??? due to enpoints?
            ui0 = self.us[i]
            ui1 = self.us[i + 1]
            x = grid[(ui0 <= grid) & (grid <= ui1)]
            points.extend(self.d(i, x.reshape(len(x), 1), 3))

        return np.array(points)

    def eval_by_sum(self, u):
        """
        Needed for task 4. Evaluates the spline by using the sum approach
        instead of Blossoms algorithm.
        Does not utilize bandedness yet.
        Note: Does not return endpoint due to problem with the last basis function
        """
        grid    = self.grid 
        nbr_ds  = self.nbr_ds

        # Calculate "vandermonde like" matrix
        memo    = {}
        l_grid  = len(grid)
        N       = np.zeros((l_grid, nbr_ds + 6 )) # OBS +6
        for i in np.arange(nbr_ds + 5): # Correct indexing? Why + 5?
            self.N3(self.us, i, grid, 3, memo)
            N[:, i] = memo[(i, 3)]

        #print("N.shape = {}".format(N.shape))
        #plt.figure("test")
        #plt.plot(grid, N[:,-4])
        # Calculate sum
        #print(N[10:12,:])
        xs = np.dot(N, self.ds[:, 0])
        ys = np.dot(N, self.ds[:, 1])
        
        # print("N.shape = {}, self.ds.shape = {}".format(N.shape, self.ds.shape))
        # p = 0
        # q = -2
        # xs = np.dot(N[:,p:q], self.ds[p:q, 0])
        # ys = np.dot(N[:,p:q], self.ds[p:q, 1])
        
        return (xs[:-1], ys[:-1]) 


plt.close("all")
ds = np.array([ [ -20,   10],
                [ -50,   20],
                [ -25,    5],
                [-100,  -15],
                [ -25,  -65],
                [  10,  -80],
                [  60,  -30],
                [  10,   20],
                [  20,    0],
                [  40,   20]])

x = np.linspace(0,1,150)
s = Spline(x, ds)
s.plot(1,1)

# Test eval_by_sum
xs, ys = s.eval_by_sum(s.us)
plt.plot(xs, ys,'*')
plt.show()

# plt.figure()
# t3 = {}
# s.N3(s.us, 14, x, 3, t3)
# a = t3[(14,3)]
# a[-1] = 1
# plt.plot(x, t3[(14,3)]) 

# def main():
#     plt.close("all")
#     ds = np.array([
#             [ -20,     10],
#             [ -20,     10],
#             [ -20,     10],
#             [ -50,     20],
#             [ -25,      5],
#             [-100,    -15],
#             [ -25,    -65],
#             [  10,    -80],
#             [  60,    -30],
#             [  10,     20],
#             [  20,      0],
#             [  40,     20],
#             [  40,     20],
#             [  40,     20]])

#     x = np.linspace(0,1,150)
#     s = Spline(x, ds)
#     s.plot()

#     # print(shape(x))
#     # plt.plot(x, s.N(s.us, 1,x,3))
#     # plt.plot(x, s.N2(1,x,3,{})[(0,3)])
#     # memo = {}
#     # s.N3(1,x,3,memo)
#     # plt.plot(x, memo[(0,3)])

#     """
#     This test shows that:
#     N much slower than N2
#     a = np.arange(10)
#     s = splines(a, a)
#     %timeit t1 = s.N(0, x, 10) # 49.7 ms
#     %timeit t2 = s.N2(0, x, 10, {})[(0,10)] #2.24 ms
#     t3 = {}
#     %timeit s.N3(0, x, 10, t3) # 954 ns!

#     np.all(t1 == t2) * all(t1 == t3[(0,10)])
#     """

#     # test d
#     # plt.figure()
#     # plt.plot(ds[:,0], ds[:,1])

#     # I = arange(3, 14)
#     # I_len = len(I)

#     # for i in I:
#     #     x = np.linspace(i, i + 1 ,100)
#     #     p = 3  # min(i, 3)
#     #     # p = min(min(i, 3), I_len - i - 2)
#     #     points = s.d(i, x, p)
#     #     # print("i = {}, p = {}, I_len - i = {}".format(i, p, I_len - i - 1))
#     #     plt.plot(points[:,0],points[:,1])

#     # Test eval_by_sum
#     xs, ys = s.eval_by_sum(s.us)
#     plt.plot(xs, ys)
#     plt.show()


# if __name__ == '__main__':
#     main()
