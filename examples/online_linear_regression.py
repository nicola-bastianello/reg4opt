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

from reg4opt.operators import Gradient
from reg4opt.regression import operator_regression, convex_regression
from reg4opt.interpolation import interpolator
from reg4opt.utils import print_progress, generate_data, generate_data_cr

import tools


#%% SET-UP

# ------------------  parameters choice
n = 100 # domain dimension
t_s, t_max = 0.1, 50 # sampling time and time horizon

# signal frequence
omega = 1

# signal sparsity
sparsity = int(n/3)
idx_z = ran.choice(range(n), sparsity, False)

# observation noise variance
noise_var = 1e-2

# observation matrix
L, mu = 1e8, 1 # condition num.
rk = n//2 # rank

# initial condition
x0 = 10*ran.standard_normal((n,1))

# weight of regularization
w = 1000


# ------------------ create problem

# ------ signal
phi = np.pi*ran.random(n) # phase

# generate signal
signal = np.zeros((n, int(t_max/t_s)))
for i in range(n):
    signal[i,:] = np.sin(omega*np.arange(0,t_max,t_s) + phi[i])

signal[idx_z,:] = 0 # sparsify signal


# ------ observations
# observation noise
noise = np.sqrt(noise_var)*ran.standard_normal(signal.shape)

# observations
A = random_matrix(np.sqrt(np.hstack((L, (L-mu)*ran.random(rk-2)+mu, mu, np.zeros(n-rk)))))
b = A.dot(signal) + noise


# ------ algorithms parameters
step = 2 / (L + mu) # gradient step-size

num_data = 3 # num. data in OpReg and budget of allowed gradient calls
var = 1e-2 # for choosing training data

zeta = 0.75 # contraction constant of approximate operator
bar_mu, bar_L = 1, 100 # strong convexity and smoothness moduli

# PRS parameters
rho = 1e-6
rho_cr = 1e-3 # for CvxReg
tol = 1e-4

num_interp = 1

# num. past iterates in Anderson extrapolation
num_old = 3


# ------ set up cost and operator
# costs
g = costs.Norm_1(n, w)
f = tools.DynamicLinearRegression(A, b, t_s=t_s)
# operator
T = Gradient(f, step)


#%% TEST THE ALGORITHMS

algs = ["Forward-backward", "FISTA", "Backtracking FISTA", "Anderson acc.", 
        "OpReg-Boost (interp.)", "OpReg-Boost", "CvxReg-Boost"]


errors = {}


# -------------------- FORWARD-BACKWARD
a = "Forward-backward"
print(f"************************ {a} ************************")
    
x = np.zeros(f.dom.shape + (f.time.num_samples+1,))
x[...,0] = x0
    
for k in range(f.time.num_samples):
        
    f_k = f.sample(k*t_s) # sample the cost
    
    # apply proximal gradient step
    x[...,k+1] = solvers.fbs({"f":f_k, "g":g}, step, x_0=x[...,k], num_iter=num_data)
    
    print_progress(k+1, f.time.num_samples)

# results
errors[a] = [la.norm(x[...,k] - signal[:,[k]]) for k in range(f.time.num_samples)]


# -------------------- FISTA
a = "FISTA"
print(f"\n************************ {a} ************************")

x = np.zeros(f.dom.shape + (f.time.num_samples+1,))
x[...,0] = x0
    
for k in range(f.time.num_samples):
        
    f_k = f.sample(k*t_s) # sample the cost
    
    # apply FISTA
    x[...,k+1] = tools.fista({"f":f_k, "g":g}, 1/L, x_0=x[...,k], num_iter=num_data)
    
    print_progress(k+1, f.time.num_samples)

# results
errors[a] = [la.norm(x[...,k] - signal[:,[k]]) for k in range(f.time.num_samples)]


# -------------------- BACKTRACKING FISTA
a = "Backtracking FISTA"
print(f"\n************************ {a} ************************")

x = np.zeros(f.dom.shape + (f.time.num_samples+1,))
x[...,0] = x0
    
for k in range(f.time.num_samples):
        
    f_k = f.sample(k*t_s) # sample the cost
    
    # apply backtracking FISTA
    x[...,k+1] = tools.backtracking_fista({"f":f_k, "g":g}, 0.1, x_0=x[...,k], num_iter=num_data)
    
    print_progress(k+1, f.time.num_samples)

# results
errors[a] = [la.norm(x[...,k] - signal[:,[k]]) for k in range(f.time.num_samples)]


# -------------------- ANDERSON ACCELERATION
a = "Anderson acc."
print(f"\n************************ {a} ************************")

x = np.zeros(f.dom.shape + (f.time.num_samples+1,))
x[...,0] = x0
    
for k in range(f.time.num_samples):
        
    f_k = f.sample(k*t_s) # sample the cost
    
    # Anderson acceleration on gradient
    x[...,k+1] = tools.guarded_anderson_acceleration({"f":f_k, "g":g}, num_old, 1/L, x_0=x[...,k], num_iter=num_data)
    
    print_progress(k+1, f.time.num_samples)

# results
errors[a] = [la.norm(x[...,k] - signal[:,[k]]) for k in range(f.time.num_samples)]

    
# -------------------- OpReg
a = "OpReg-Boost"
print(f"\n************************ {a} ************************")

x = np.zeros(T.dom.shape + (T.time.num_samples+1,))
x[...,0] = x0
    
for k in range(T.time.num_samples):
        
    T_k = T.sample(k*t_s) # sampled operator
    
    # generate training data
    x_i, y_i = generate_data(T_k, x[...,k], num_data, var=var)
        
    # apply OpReg solver
    t_i = operator_regression(x_i, y_i, zeta, tol=tol, rho=rho)

    x[...,k+1] = g.proximal(t_i[0], step)
    
    print_progress(k+1, T.time.num_samples)

# results
errors[a] = [la.norm(x[...,k] - signal[:,[k]]) for k in range(T.time.num_samples)]


# -------------------- CONVEX REGRESSION
a = "CvxReg-Boost"
print(f"\n************************ {a} ************************")

x = np.zeros(f.dom.shape + (f.time.num_samples+1,))
x[...,0] = x0
    
for k in range(f.time.num_samples):
        
    f_k = f.sample(k*t_s) # sampled cost
    
    # generate training data
    x_i, y_i, w_i = generate_data_cr(f_k, x[...,k], num_data, gradient=True, var=var)
    
    # apply CvxReg solver
    _, g_i = convex_regression(x_i, y_i, bar_mu, bar_L, w=w_i, tol=tol, rho=rho_cr)
    
    x[...,k+1] = g.proximal(x[...,k] - step*g_i[0], step)
    
    print_progress(k+1, f.time.num_samples)

# results
errors[a] = [la.norm(x[...,k] - signal[:,[k]]) for k in range(f.time.num_samples)]


# -------------------- OpReg
a = "OpReg-Boost (interp.)"
print(f"\n************************ {a} ************************")

x = np.zeros(T.dom.shape + (T.time.num_samples+1,))
x[...,0] = x0
    
for k in range(T.time.num_samples):
            
    T_k = T.sample(k*t_s) # sampled operator
    
    # ------ apply operator regression solver
    if k % (num_interp+1) == 0:
    
        # training data
        x_i, y_i = generate_data(T_k, x[...,k], num_data, var=var)
        
        # apply OpReg solver
        t_i = operator_regression(x_i, y_i, zeta, tol=tol, rho=rho)
        
        # apply solution
        x[...,k+1] = g.proximal(t_i[0], step)
    
    # ------ apply interpolation
    else:

        y = interpolator(x[...,k], x_i, t_i, zeta, t0=T_k.operator(x[...,k]))
        
        x[...,k+1] = g.proximal(y, step)

    print_progress(k+1, T.time.num_samples)

# results
errors[a] = [la.norm(x[...,k] - signal[:,[k]]) for k in range(T.time.num_samples)]


#%% RESULTS

import matplotlib.pyplot as plt

plt.rc("text", usetex=True)
plt.rc("font", family="serif")
fontsize = 18

markers = ['o', 's', 'v', '^', '*', 'D', 'H', '<', '>', 'p']
markevery = 100

# ------ tracking error over time
time = num_data*np.arange(1, f.time.num_samples+1)

plt.figure()

for idx_a, a in enumerate(algs):
    
    plt.semilogy(time, errors[a], label=a, marker=markers[idx_a], markevery=markevery)


plt.grid()
plt.legend(fontsize=14)

plt.xlabel(r"Gradient calls (${} t_k$)".format(num_data), fontsize=fontsize)
plt.ylabel("Tracking error", fontsize=fontsize)

plt.tight_layout()

plt.show()