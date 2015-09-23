"""
FMNN25 Project 1 - Splines

main.py
Runs demo code to show functionality of splines.py

Contributors:
Lewis Belcher
Angelos Toytziaridis
Simon Ydman
"""

import matplotlib.pyplot as plt
import numpy as np
import sp_func
import splines
import timeit


plt.close("all")  # close all open plots

# create control points array (somewhat matching that shown in the lectures)
ds = np.array([
    [ -20,   10],
    [ -50,   20],
    [ -25,    5],
    [-100,  -15],
    [ -25,  -65],
    [  10,  -80],
    [  60,  -30],
    [  10,   20],
    [  20,    0],
    [  40,   20]
])

# create our spline
x = np.linspace(0, 1, 150)
s = splines.Spline(x, ds)


# plot using the default algorithm (blossoms)
s.plot(plot_deBoor_points=True, plot_control_poly=True)


# Some timing tests for the memoized functions
setup = """
import splines
import numpy as np
ds = np.array([
    [ -20,   10],
    [ -50,   20],
    [ -25,    5],
    [-100,  -15],
    [ -25,  -65],
    [  10,  -80],
    [  60,  -30],
    [  10,   20],
    [  20,    0],
    [  40,   20]
])
x = np.linspace(0, 1., 150)
s = splines.Spline(x, ds)
"""

runs = 1500

t1 = timeit.timeit("s.N(s.us, 4, x, 3)", setup=setup, number=runs)
print("Time to run s.N %s times: %5.3es" % (runs, t1))

t2 = timeit.timeit("s.N2(s.us, 4, x, 3, {})", setup=setup, number=runs)
print("Time to run s.N2 %s times: %5.3es" % (runs, t2))

setup += "d = {}"
t3 = timeit.timeit("s.N3(s.us, 4, x, 3, d)", setup=setup, number=runs)
print("Time to run s.N3 %s times: %5.3es" % (runs, t3))


# Plot the spline created by `Spline.eval_by_sum`
ys = s.eval_by_sum()
plt.plot(ys[:, 0], ys[:, 1])
plt.title("Spline created by summing basis functions")
plt.xlabel("x")
plt.ylabel("y")
plt.show()


knots = [0, 0, 0, 0, 1, 2, 3, 4, 4, 4, 4]
nknots = len(knots)
fs = [sp_func.get_basis_func(knots, j) for j in range(nknots - 4)]
x = np.linspace(0., 4-1e-9, 1000)
fxs = [f(x) for f in fs]
for i, fx in enumerate(fxs):
    plt.plot(x, fx, label='$N^3_{0}$'.format(i))
plt.title("Example basis functions")
plt.xlabel("u")
plt.ylabel("N(u)")
plt.show()
