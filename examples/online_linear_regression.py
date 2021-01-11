#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Solve an online linear regression problem with l1 regularization, comparing
the operator regression solver with other methods.
"""

import numpy as np
from numpy import linalg as la
from numpy.random import default_rng
ran = default_rng()

import sys
sys.path.append("..")

from tvopt.utils import random_matrix
from tvopt import costs, solvers

from reg4opt.operators import Gradient, Proximal
from reg4opt.regression import operator_regression
from reg4opt.interpolation import interpolator
from reg4opt.utils import print_progress, generate_data
from reg4opt.solvers import fista, anderson_acceleration



#%% SET-UP

n = 10 # domain dimension
t_s, t_max = 0.1, 10 # sampling time and time horizon


# ------ signal
omega = 3 # angular velocity
phi = np.pi*ran.random(n) # phase
shift = 10*ran.random(n) # shift the signals away from the origin

# generate signal
s = np.zeros((n, int(t_max/t_s)))
for i in range(n):
    s[i,:] = np.sin(omega*np.arange(0,t_max,t_s) + phi[i]) + shift[i]

# sparsify signal
sparsity = int(n/3)

idx_z = ran.choice(range(n), sparsity, False)
s[idx_z,:] = 0

# ------ observation noise
noise_var = 1e-4
noise = np.sqrt(noise_var)*ran.standard_normal(s.shape)


# ------ regression cost
# choose condition number
L, mu = 1e8, 1

# observation matrix
rk = n//2
A = random_matrix(np.sqrt(np.hstack((L, (L-mu)*ran.random(rk-2)+mu, mu, np.zeros(n-rk)))))

# observations
b = A.dot(s) + noise


# ------ generate cost and operator
step = 2 / (L + mu) # gradient step-size
w = 5 # weight of l1 norm

# costs
f = costs.DiscreteDynamicCost([costs.LinearRegression(A, b[:,[k]]) for k in range(b.shape[1])], t_s=t_s)
g = costs.Norm_1(n, w)

# operators
T = Gradient(f, step)
P = Proximal(g, step)


# ------ OpReg parameters
num_data = 5
var = 1e-2 # for choosing training data

bar_zeta = 0.5 # contraction constant of approximate operator

num_iter = 5 # num. of iterations per sampled problem

# PRS parameters
newton_params = {'tol':1e-4, 'num_iter':5, 'b':0.5, 'c':0.1, 'max_iter':2}
rho = 1
tol = 1.5*1e-4

# parameter for Anderson acceleration (num. of past iterates in extrapolation)
m = num_data


# initial condition
x0 = 10*ran.standard_normal((n,1))


#%% TEST THE ALGORITHMS

# -------------------- GRADIENT
print("Proximal gradient method ...")

x = np.zeros(f.dom.shape + (f.time.num_samples+1,))
x[...,0] = x0

t = 0

for k in range(f.time.num_samples):
    
    f_k = f.sample(k*t_s) # sample the cost
    
    # apply proximal gradient step
    x[...,k+1] = solvers.fbs({"f":f_k, "g":g}, step, x_0=x[...,k], num_iter=num_iter)
    
    print_progress(k+1, f.time.num_samples)

# results
err_pg = [la.norm(x[...,k+1] - s[:,[k]]) for k in range(f.time.num_samples)]


# -------------------- FISTA
print("FISTA ...")

x = np.zeros(f.dom.shape + (f.time.num_samples+1,))
x[...,0] = x0

t = 0

for k in range(f.time.num_samples):
    
    f_k = f.sample(k*t_s) # sampled cost
    
    # apply proximal gradient step
    x[...,k+1] = fista({"f":f_k, "g":g}, 1/L, x_0=x[...,k], num_iter=num_iter)

    print_progress(k+1, f.time.num_samples)

# results
err_f = [la.norm(x[...,k+1] - s[:,[k]]) for k in range(f.time.num_samples)]


# -------------------- ANDERSON ACCELERATION
print("Anderson acceleration ...")

x = np.zeros(f.dom.shape + (f.time.num_samples+1,))
x[...,0] = x0
x_old = [x0]

t = 0

for k in range(f.time.num_samples):
    
    T_k = T.sample(k*t_s) # sampled operator
        
    for _ in range(num_iter):
        
        # Anderson acceleration on gradient
        y = anderson_acceleration({"T":T_k}, m, x_0=x_old, num_iter=num_iter)
        
        # proximal step
        y = g.proximal(y, step)
        x_old.append(y)
    
    x[...,k+1] = y
    
    # update list of past iterates
    x_old = [x[...,k+1-i] for i in range(min(k+1, m))][::-1]

    print_progress(k+1, f.time.num_samples)

# results
err_aa = [la.norm(x[...,k+1] - s[:,[k]]) for k in range(f.time.num_samples)]


# -------------------- PRS
print("OpReg (PRS) ...")

x = np.zeros(f.dom.shape + (f.time.num_samples+1,))
x[...,0] = x0

t = 0

for k in range(f.time.num_samples):
    
    T_k = T.sample(k*t_s) # sampled operator
    
    # apply operator regression solver
    y = x[...,k]
    
    for _ in range(num_iter):
        
        # generate training data
        x_i, y_i = generate_data(T_k, y, num_data, var)
        
        # apply OpReg solver
        t_i, _ = operator_regression(x_i, y_i, bar_zeta, solver="PRS", tol=tol, rho=rho, newton_params=newton_params)
        
        y = g.proximal(t_i[0], step)
    
    x[...,k+1] = y
    
    print_progress(k+1, f.time.num_samples)

# results
err_prs = [la.norm(x[...,k+1] - s[:,[k]]) for k in range(f.time.num_samples)]


# -------------------- CVXPY
print("OpReg (CVXPY) ...")

x = np.zeros(f.dom.shape + (f.time.num_samples+1,))
x[...,0] = x0

t = 0

for k in range(f.time.num_samples):
    
    T_k = T.sample(k*t_s) # sampled operator
    
    # apply operator regression solver
    y = x[...,k]
    
    for _ in range(num_iter):
        
        # generate training data
        x_i, y_i = generate_data(T_k, y, num_data, var)
        
        # apply OpReg solver
        t_i, _ = operator_regression(x_i, y_i, bar_zeta, solver="CVXPY")
        
        y = g.proximal(t_i[0], step)
    
    x[...,k+1] = y
    
    print_progress(k+1, f.time.num_samples)

# results
err_cp = [la.norm(x[...,k+1] - s[:,[k]]) for k in range(f.time.num_samples)]


# -------------------- INTERPOLATED (PRS)
print("OpReg (PRS, interpolated) ...")

x = np.zeros(f.dom.shape + (f.time.num_samples+1,))
x[...,0] = x0

t = 0

for k in range(f.time.num_samples):
    
    T_k = T.sample(k*t_s) # sampled operator
    
    # apply operator regression solver
    y = x[...,k]
    
    # ------ first step: solve OpReg
    # training data
    x_i, y_i = generate_data(T_k, y, num_data, var)
    
    # apply OpReg solver
    t_i, _ = operator_regression(x_i, y_i, bar_zeta, solver="PRS", tol=tol, rho=rho, newton_params=newton_params)
    
    # apply solution
    y = g.proximal(t_i[0], step)
    
    # ------ interpolation step        
    for _ in range(num_iter-1):
    
        y = interpolator(y, x_i, t_i, bar_zeta)
    
    x[...,k+1] = g.proximal(y, step)  
    
    print_progress(k+1, f.time.num_samples)

# results
err_in = [la.norm(x[...,k+1] - s[:,[k]]) for k in range(f.time.num_samples)]


#%% RESULTS

import matplotlib.pyplot as plt

plt.rc("text", usetex=True)
plt.rc("font", family="serif")
plt.rc("text.latex", preamble=r"\usepackage{bm,fixmath,amssymb}")
fontsize = 18

markers = ['o', 's', 'v', '^', '*', 'D', 'H', '<', '>', 'p']


# ------ tracking error over time
time = np.arange(0, f.time.t_max, f.time.t_s)

plt.figure()

plt.semilogy(time, err_pg, label="Proximal gradient", marker=markers[0], markevery=50)
plt.semilogy(time, err_f, label="FISTA", marker=markers[1], markevery=50)
plt.semilogy(time, err_aa, label="Anderson", marker=markers[2], markevery=50)
plt.semilogy(time, err_prs, label="OpReg (PRS)", marker=markers[3], markevery=50)
plt.semilogy(time, err_cp, label="OpReg (CVXPY)", marker=markers[4], markevery=50)
plt.semilogy(time, err_in, label="OpReg (PRS, interpolation)", marker=markers[5], markevery=50)


plt.grid()
plt.legend(fontsize=12)

plt.xlabel("Time [s]", fontsize=fontsize)
plt.ylabel("Tracking error", fontsize=fontsize)

plt.show()


# ------ cumulative tracking error per gradient calls
time = np.arange(1, f.time.num_samples+1)

plt.figure()

plt.semilogy(num_iter*time, np.cumsum(err_pg)/time, label="Proximal gradient", marker=markers[0], markevery=50)
plt.semilogy(num_iter*time, np.cumsum(err_f)/time, label="FISTA", marker=markers[1], markevery=50)
plt.semilogy((num_data+num_iter-1)*time, np.cumsum(err_aa)/time, label="Anderson", marker=markers[2], markevery=50)
plt.semilogy(num_data*num_iter*time, np.cumsum(err_prs)/time, label="OpReg (PRS)", marker=markers[3], markevery=50)
plt.semilogy(num_data*num_iter*time, np.cumsum(err_cp)/time, label="OpReg (CVXPY)", marker=markers[4], markevery=50)
plt.semilogy((num_data+num_iter-1)*time, np.cumsum(err_in)/time, label="OpReg (PRS, interpolation)", marker=markers[5], markevery=50)


plt.grid()
plt.legend(fontsize=12)

plt.xlabel("Gradient calls", fontsize=fontsize)
plt.ylabel("Cumulative tracking error", fontsize=fontsize)

plt.show()