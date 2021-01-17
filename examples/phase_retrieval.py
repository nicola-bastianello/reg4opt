#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Solve an online phase retrieval problem.
"""

import cvxpy as cp
import numpy as np
from numpy import linalg as la
from numpy.random import default_rng
ran = default_rng()

import sys
sys.path.append("..")

from tvopt.utils import orthonormal_matrix
from tvopt import sets

from reg4opt.operators import Operator, DiscreteDynamicOperator
from reg4opt.regression import operator_regression
from reg4opt.utils import print_progress, generate_data
from reg4opt.solvers import anderson_acceleration



class PhaseRetrieval(Operator):
    r"""
    Consider the phase retrieval problem
    
    .. math:: f(x) = \frac{1}{m} \sum_{i = 1}^m \left| \langle a_i, x \rangle^2 - b_i \right|
    
    where :math:`a_i \in \mathbb{R}^n` are the vectors that yield the data
    :math:`b_i = \langle a_i, x^* \rangle^2 + e_i`, for some error :math:`e_i`.
    
    The operator defined by this class is 
    :math:`\mathcal{T} y = \operatorname{prox}_{\alpha f_y}(y)` where 
    
    .. math:: f_y(x) = \frac{1}{m} \sum_{i = 1}^m 
            \left| \langle a_i, y \rangle^2 + 
            2 \langle a_i, y \rangle \langle a_i, x - y \rangle - b_i \right|
    
    is a linearized version of the cost function.
    
    See [#]_ for a treatment of the this algorithm.
    
    References
    ----------
    .. [#] D. Drusvyatskiy and A. S. Lewis, "Error Bounds, Quadratic Growth,
           and Linear Convergence of Proximal Methods," Mathematics of Operations Research, 
           vol. 43, no. 3, pp. 919–948, Aug. 2018.
    """
    
    def __init__(self, A, b):
        # each row in A and element of b represent a data point
        
        super().__init__(sets.R(A.shape[1], 1))
        
        self.m = A.shape[0] # domain dim. and num. data points
        # store data as lists
        self.A = [A[d,].reshape(self.dom.shape).T for d in range(self.m)]
        self.b = [np.array(b[d]).item() for d in range(self.m)]
            
    def operator(self, x, penalty=1):
        
        x = self.dom.check_input(x)
        
        # define CVXPY problem
        y = cp.Variable(self.dom.shape)
        
        cost = cp.sum_squares(y - x) / (2*penalty)
        for d in range(self.m):
            c = (self.A[d].dot(x)).item()           
            cost += cp.abs(c**2 + 2*c*self.A[d] @ (y - x) - self.b[d]) / self.m
        
        problem = cp.Problem(cp.Minimize(cost))
        problem.solve()
        
        return y.value.reshape((-1,1))


def time_varying_phases(n, m, std=1e-1):
    r"""
    Generate time-varing phases.
    
    Generate a sequence of points on the unit sphere in :math:`\mathbb{R}^n`,
    as follows:
        
    * the first phase is defined as :math:`x_0 = e / \| e \|`, with
      :math:`e \sim \mathcal{N}(0, I_n)`;
    
    * subsequent phases are chosen as :math:`x_{k+1} = (x_k + d) / \| x_k + d \|`
      where :math:`d \sim \mathcal{N}(0, \sigma^2 I_n)`.

    Parameters
    ----------
    n : int
        The size of the space.
    m : int
        the number of phases to be generated.
    std : float, optional
        The standard deviation of the normal for choosing the next phase. The 
        default is 1e-1.

    Returns
    -------
    x : ndarray
        The generated phases.
    """
    
    x = np.zeros((n, m))
    
    # first phase
    e = ran.normal(size=(n,1))
    x[:,[0]] = e / la.norm(e)
    
    # subsequent phases
    for l in range(m-1):
        
        y = x[:,[l]] + ran.normal(scale=std, size=(n,1))
        x[:,[l+1]] = y / la.norm(y)
    
    return x


#%% SET-UP

n = 15 # domain dimension
m = 200 # num. data
t_s, t_max = 0.1, 5 # sampling time and time horizon


# ------ signal
s = time_varying_phases(n, int(t_max/t_s)) # generate the phases

# ------ observation noise
noise_scale = 1
noise = ran.laplace(scale=noise_scale, size=(n, int(t_max/t_s)))


# ------ regression cost
# choose condition number
L, mu = 1e2, 1

U = orthonormal_matrix(n)
D = np.diag(np.hstack((L, (L-mu)*ran.random(n-2)+mu, mu)))

A = U.dot(D)

# observations
b = A.dot(s)**2 + noise


# ------ generate operator
T = DiscreteDynamicOperator([PhaseRetrieval(A, b[:,[k]]) for k in range(int(t_max/t_s))], t_s=t_s)
penalty = 0.05


# ------ OpReg parameters
num_data = 10
var = 1e-2 # for choosing training data

bar_zeta = 0.1 # contraction constant of approximate operator

num_iter = 1 # num. of iterations per sampled problem

# PRS parameters
newton_params = {'tol':1e-4, 'num_iter':5, 'b':0.5, 'c':0.1, 'max_iter':2}
rho = 0.01
tol = 1e-4


# initial condition
x0 = 25*ran.standard_normal((n,1))

# Anderson memory slots
m = int(num_data / 2)


#%% TEST THE ALGORITHMS

# -------------------- PROX-LINEAR
print("Prox-linear method ...")

x = np.zeros(T.dom.shape + (T.time.num_samples+1,))
x[...,0] = x0


for k in range(T.time.num_samples):
    
    T_k = T.sample(k*t_s) # sample the cost
    
    # apply prox-linear operator
    y = x[...,k]
    
    for _ in range(num_iter*num_data):
        y = T_k.operator(y, penalty=penalty)
    
    x[...,k+1] = y
    
    print_progress(k+1, T.time.num_samples)

# results
err_pl = [la.norm(x[...,k+1] - s[:,[k]]) for k in range(T.time.num_samples)]


# -------------------- ANDERSON ACCELERATION
print("Anderson acceleration ...")

x = np.zeros(T.dom.shape + (T.time.num_samples+1,))
x[...,0] = x0
x_old = [x0] # memory slots


for k in range(T.time.num_samples):
    
    T_k = T.sample(k*t_s) # sampled operator
    
    for _ in range(num_iter):
        
        # Anderson acceleration
        y = anderson_acceleration({"T":T_k}, m, x_0=x_old, num_iter=6)
        
        # store past iterates
        x_old.append(y)
    
    x[...,k+1] = y
    
    # update list of past iterates
    x_old = [x[...,k+1-i] for i in range(min(k+1, m))][::-1]
    
    
    print_progress(k+1, T.time.num_samples)

# results
err_an = [la.norm(x[...,k+1] - s[:,[k]]) for k in range(T.time.num_samples)]


# -------------------- OPERATOR REGRESSION
print("OpReg ...")

x = np.zeros(T.dom.shape + (T.time.num_samples+1,))
x[...,0] = x0


for k in range(T.time.num_samples):
    
    T_k = T.sample(k*t_s) # sampled operator
    
    # apply operator regression solver
    y = x[...,k]
    
    for _ in range(num_iter):
        
        # generate training data
        x_i, y_i = generate_data(T_k, y, num_data, var=var)
        
        # apply OpReg solver
        t_i, _ = operator_regression(x_i, y_i, bar_zeta, solver="PRS", tol=tol, rho=rho, newton_params=newton_params)
        
        y = t_i[0]
    
    x[...,k+1] = y
    
    print_progress(k+1, T.time.num_samples)

# results
err_or = [la.norm(x[...,k+1] - s[:,[k]]) for k in range(T.time.num_samples)]


#%% RESULTS

import matplotlib.pyplot as plt

plt.rc("text", usetex=True)
plt.rc("font", family="serif")
fontsize = 18


# ------ cumulative tracking error per operator calls
time = num_data*num_iter*np.arange(1, T.time.num_samples+1)

plt.figure()

plt.semilogy(time, np.cumsum(err_pl)/time, label="Prox-linear", marker="o", markevery=10)
plt.semilogy(time, np.cumsum(err_or)/time, label="OpReg", marker="s", markevery=10)
plt.semilogy(time, np.cumsum(err_an)/time, label="Anderson acceleration", marker="v", markevery=10)


plt.grid()
plt.legend(fontsize=12)

plt.xlabel("Operator calls", fontsize=fontsize)
plt.ylabel("Cumulative tracking error", fontsize=fontsize)

plt.show()