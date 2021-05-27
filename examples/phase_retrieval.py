#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Solve an online phase retrieval problem.
"""

import numpy as np
from numpy import linalg as la
from numpy.random import default_rng
ran = default_rng()

import sys
sys.path.append("..")

from tvopt.utils import orthonormal_matrix

from reg4opt.operators import DiscreteDynamicOperator
from reg4opt.regression import operator_regression
from reg4opt.utils import print_progress, generate_data

import tools


print("NOTE: running this script may take some time (around 20 minutes).\n\n")


#%% SET-UP

# ------ problem dimensions
n = 50 # domain dim.
m = 100 # num. of observations

# ------ piecewise signal properties
num_pieces = 50
len_pieces = 10

t_s = 1 # sampling time
t_max = t_s*num_pieces*len_pieces

# ------ condition number
L, mu = 1e2, 1

# ------ observation noise properties
noise_scale = 1


# ------ phase
signal = tools.dynamic_phases(n, num_pieces=num_pieces, len_pieces=len_pieces)

# ------ observation noise
noise = ran.laplace(scale=noise_scale, size=(m, signal.shape[1]))

# ------ regression cost
U = orthonormal_matrix(m)[:,:n]
D = np.diag(np.hstack((L, (L-mu)*ran.random(n-2)+mu, mu)))

A = U.dot(D)

# observations
b = A.dot(signal)**2 + noise

# initial condition
x0 = 25*ran.standard_normal((n,1))


# ------ OpReg parameters
var = 1e-2 # for choosing training data

zeta = 0.75 # contraction constant of approximate operator

# PRS parameters
rho = 1e-4
tol = 1e-4

# num. of operator calls per sampling time
oper_budget = 4

penalty = 0.01 # prox-linear penalty


# generate operator
T = DiscreteDynamicOperator([tools.PhaseRetrieval(A, b[:,[k]], penalty=penalty) for k in range(int(t_max/t_s))], t_s=t_s)


#%% TEST THE ALGORITHMS

# -------------------- PROX-LINEAR
print("Prox-linear method ...")

x = np.zeros(T.dom.shape + (T.time.num_samples+1,))
x[...,0] = x0

for k in range(T.time.num_samples):
    
    T_k = T.sample(k*t_s) # sample the cost
    
    # apply prox-linear operator
    y = x[...,k]
    
    for _ in range(oper_budget):
        y = T_k.operator(y)
    
    x[...,k+1] = y
    
    print_progress(k+1, T.time.num_samples)

# results
errors_pl = [la.norm(x[...,k] - signal[:,[k]]) for k in range(T.time.num_samples)]


# -------------------- OPERATOR REGRESSION
print("OpReg ...")

x = np.zeros(T.dom.shape + (T.time.num_samples+1,))
x[...,0] = x0

for k in range(T.time.num_samples):
    
    T_k = T.sample(k*t_s) # sampled operator
    
    # generate training data
    x_i, y_i = generate_data(T_k, x[...,k], oper_budget-1, var=var)
        
    # apply OpReg solver
    t_i = operator_regression(x_i, y_i, zeta, tol=tol, rho=rho)
        
    x[...,k+1] = t_i[0] / la.norm(t_i[0])
    
    print_progress(k+1, T.time.num_samples)

# results
errors_or = [la.norm(x[...,k] - signal[:,[k]]) for k in range(T.time.num_samples)]


#%% RESULTS

# -------------------- plot results
import matplotlib.pyplot as plt

plt.rc("text", usetex=True)
plt.rc("font", family="serif")
plt.rc("text.latex", preamble=r"\usepackage{bm,fixmath,amssymb}")
fontsize = 18

markers = ['o', 's', 'v', '^', '*', 'D', 'H', '<', '>', 'p']
markevery = 50

# ------ tracking error over time
time = np.arange(T.time.num_samples)

    
plt.figure()

# prox-linear
plt.semilogy(time, errors_pl, label="Prox-linear", marker=markers[0], 
             markevery=markevery, linewidth=2)

# OpReg-Boost
plt.semilogy(time, errors_or, label="OpReg-Boost", marker=markers[1], 
             markevery=markevery, linewidth=2)


plt.legend(fontsize=fontsize)

plt.xlabel(r"Time ($k$)", fontsize=fontsize)
plt.ylabel("Tracking error", fontsize=fontsize)

plt.grid()

plt.tight_layout()

plt.show()
