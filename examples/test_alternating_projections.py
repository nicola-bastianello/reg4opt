#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from numpy import linalg as la

from numpy.random import default_rng
ran = default_rng()

import sys
sys.path.append("..")

from reg4opt import utils


# plots setup
import matplotlib.pyplot as plt
    
plt.rc("text", usetex=True)
plt.rc("font", family="serif")
plt.rc("text.latex", preamble=r"\usepackage{bm,fixmath,amssymb}")
fontsize = 18


def alternating_projections(s, x, tol=1e-10, num_iters=1000):
    """
    Implementation of the method of alternating projections (MAP) that only 
    returns the number of iterations required to reach convergence. See
    `alternating_projections` in `reg4opt.interpolation` for documentation.
    """
    
    x_old = np.copy(x)
    
    for k in range(num_iters):
    
        for i in range(len(s)):
            
            x = s[i].projection(x)
            
        # check stopping condition
        if la.norm(x - x_old) <= tol: break
        else: x_old = np.copy(x)
    
    return k



save_fig = False


#%% HISTOGRAM

# ------ problem formulation
# domain dimensions and num. of balls
n = 25
d = 100

reps = 10

# initial condition
x0 = 1e10*ran.standard_normal((n,1))


# ------ alternating projections
results = []

for l in range(reps):

    # generate the balls
    s = utils.random_intersecting_balls(n, d)
    
    # apply alternating projections        
    k = alternating_projections(s, x0, tol=1e-15)
    
    results.append(k)
    
    utils.print_progress(l+1, reps)


# ------ results
print("Max. num. of iterations {}".format(max(results)))
print("Mean num. of iterations {} +/- {}".format(np.mean(results), np.std(results)))
   

plt.figure()

plt.hist(results, bins=max(results), rwidth=0.8)

plt.grid()
plt.xlabel("Num. of iterations", fontsize=fontsize)
plt.ylabel("Num. of occurrences", fontsize=fontsize)
    
if save_fig: plt.savefig("images/MAP_histogram.pdf", bbox_inches="tight")
else: plt.show()


#%% HISTOGRAM (different d)
        
# ------ problem formulation
# domain dimensions and num. of balls
n = 25
range_d = range(2, 100+1)

reps = 10

# initial condition
x0 = 1e10*ran.standard_normal((n,1))


# ------ alternating projections
results = {d:[] for d in range_d}


for l in range(reps):

    # generate the balls
    balls = utils.random_intersecting_balls(n, max(range_d))

    for d in range_d:
                
        # apply alternating projections        
        k = alternating_projections(balls[:d], x0, tol=1e-15)
        
        results[d].append(k)
        
    utils.print_progress(l+1, reps)


# ------ results

# mean num. of iterations
mean = [np.mean(results[d]) for d in results.keys()]

# find a fit
c, _, _, _ = la.lstsq(np.reshape(np.log(range_d), (-1,1)), np.reshape(mean, (-1,1)), rcond=None)
c = float(c)

print("The curve is approximated by c log(D), with c = {}".format(c))

# ------ plot    
plt.figure()

plt.plot(range_d, mean, ".-", markevery=5)
plt.plot(range_d, c*np.log(range_d), "--", alpha=0.5, color="#1f77b4")

plt.grid()
plt.xlabel(r"Num. of sets ($D$)", fontsize=fontsize)
plt.ylabel("Mean num. iterations", fontsize=fontsize)
    
if save_fig: plt.savefig("images/MAP_changing_d.pdf", bbox_inches="tight")
else: plt.show()