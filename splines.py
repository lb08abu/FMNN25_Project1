"""
Class to represent splines for 2D curve design.

To design a Spline we need:
 - Control points
 - Knots
 - Coefficients (one for each control point)

Calculate all basis functions N^k_i(u)
"""

from __future__ import division
import numpy as np
import scipy as sp
from functools import partial

class Spline(object):

	def __init__(self, grid, dvalues):
		# Set degree
		self.degree 	= 3

		# Save control points
		self.ds		= dvalues
		self.nbr_ds 	= len(dvalues)

		# Find equidistant knot points
		nbr_knots 	= self.degree + self.nbr_ds
		self.nbr_knots	= nbr_knots
		self.grid 	= grid
		len_grid 	= len(grid)
		indices 	= ceil(len_grid / nbr_knots) * np.arange(nbr_knots)
		self.us 	= grid[indices.astype(int)]
		self.us[-1]	= grid[-1] # Otherwise the end point might be exluded

		# Apparently the values of the knots can be arbitrary. 
		# The only thing that matters is that we have a enough of them.

    	def __call__(self, u):
		"""
		This returns a point s(u) = [s_x(u) s_y(u)]
		"""
		idx 	= self.us.searchsorted(u)
		i 	= idx
		return self.d(i, u, 3)

	def plot(self, plot_control_poly = 0, plot_deBoor_points = 0):
		"""
		Plots the whole spline
		"""
		figure()
		if plot_control_poly:
			plot(ds[:,0], ds[:,1])

		grid = self.grid;
		for i in arange(3,self.nbr_knots - 3): # Avoid the 3 dummy points
			ui 	= self.us[i]
			ui1 	= self.us[i + 1]
			x 	= grid[(ui <= grid) & (grid  <= ui1)] 
			points 	= s.d(i, x.reshape(len(x),1), 3)
			plot(points[:,0],points[:,1], 'r')

			if plot_deBoor_points:
				plot(points[0,0], points[0,1], 'r*')
				plot(points[-1,0], points[-1,1], 'r*')

	def get_basis_func(self, us, i):
		return partial(self.N3, us = us, i = i)

	def N0(self, us, i, x):
		return (self.us[i] <= x) * (x < self.us[i + 1]);

	def N(self, us, i, x, n):
		if n == 0:
			return self.N0(us, i, x)
		else: 
			c1 = (x - us[i]) / (us[i + n] - us[i])
			c2 = (us[i + n + 1] - x) / (us[i + n + 1] - us[i + 1])
			return c1 * self.N(us, i, x, n - 1) + c2 * self.N(us, i + 1, x, n - 1)

	def N2(self, us, i, x, n, memo): # memoized, returns new dictionaries all the time
		if n == 0:
			if (i, 0) not in memo:
				memo[(i, 0)] = self.N0(us, i, x)
			return memo
		else: 
			c1 = (x - us[i]) / (us[i + n] - us[i])
			c2 = (us[i + n + 1] - x) / (us[i + n + 1] - us[i + 1])

			if (i, n - 1) not in memo:
				memo = self.N2(us, i, x, n - 1, memo)

			memo 		= self.N2(us, i + 1, x, n - 1, memo)
			memo[(i, n)] 	= c1 * memo[(i, n - 1)] + c2 * memo[(i + 1, n - 1)]

			return memo

	def N3(self, us, i, x, n, memo): # memoized, utilizes that an dictonary is mutable like a C pointer
		if n == 0:
			if (i, 0) not in memo:
				memo[(i, 0)] = self.N0(us, i,x)
			return 
		else: 
			c1 = (x - us[i]) / (us[i + n] - us[i])
			c2 = (us[i + n + 1] - x) / (us[i + n + 1] - us[i + 1])

			if (i, n - 1) not in memo:
				self.N3(us, i, x, n - 1, memo)
			
			self.N3(us, i + 1, x, n - 1, memo)
			memo[(i, n)] = c1 * memo[(i, n - 1)] + c2 * memo[(i + 1, n - 1)]
			return 

	def d(self, i, x, n):
		if n == 0:
			return self.ds[i,:]
		else:
			us 	= self.us
			a 	= (x - us[i]) / (us[i + self.degree + 1 - n] - us[i])
			return (1 - a) * self.d(i - 1, x , n - 1) + a * self.d(i, x, n - 1)

	def eval_by_sum(self, u):
		pass

close("all")
ds = array([	[ -20,	 10],
		[ -20,	 10],
		[ -20,	 10],
		[ -50,	 20],
		[ -25, 	  5],
		[-100,	-15],
		[ -25,	-65],
		[  10,	-80],
		[  60, 	-30],
		[  10,	 20],
		[  20, 	  0],
		[  40,	 20],
		[  40,	 20],
		[  40,	 20]])
x = linspace(0,1,100)
s = Spline(x, ds)
#print(shape(x))
#plot(x, s.N(s.us, 1,x,3))
#plot(x, s.N2(1,x,3,{})[(0,3)])
#memo = {}
#s.N3(1,x,3,memo)
#plot(x, memo[(0,3)])
"""
Test shows that:
N much slower than N2
a = np.arange(10)
s = splines(a,a)
%timeit t1 = s.N(0, x, 10) # 49.7 ms
%timeit t2 = s.N2(0, x, 10, {})[(0,10)] #2.24 ms
t3 = {}
%timeit s.N3(0, x, 10, t3) # 954 ns!

all(t1 == t2) * all(t1 == t3[(0,10)])

"""

# test d
# figure()
# plot(ds[:,0], ds[:,1])

# I = arange(3, 14)
# I_len = len(I)

# for i in I:
# 	x = linspace(i, i + 1 ,100)
# 	p = 3 #min(i, 3)
# 	#p = min(min(i, 3), I_len - i - 2)
# 	points = s.d(i, x, p)
# 	#print("i = {}, p = {}, I_len - i = {}".format(i, p, I_len - i - 1))
# 	plot(points[:,0],points[:,1])