#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import cvxpy as cp
import numpy as np
from numpy import linalg as la
ran = np.random.default_rng()

from collections import deque as queue

from tvopt import sets, utils, costs
from tvopt.solvers import stop
from reg4opt.operators import Operator


#%% EXAMPLE COSTS DEFINITIONS

class DynamicLinearRegression(costs.Cost):
    r"""
    Cost for time-varying linear regression.
    
    The cost is defined as
    
        .. math:: f(\pmb{x}; t) = \frac{1}{2} \| \pmb{A} \pmb{x} - \pmb{b}(t) \|^2
    
    where the observation matrix :math:`\pmb{A}` is time-invariant.
    """
    
    def __init__(self, A, b, t_s=1):        
        
        # domains
        time = sets.T(t_s, t_min=0, t_max=b.shape[1]*t_s)
        super().__init__(sets.R(A.shape[1], 1), time)
        
        # store data
        self.A, self.b = A, b
        self.smooth = 2
        
        self._H = A.T.dot(A)
    
    def function(self, x, t):
        
        k = self.time.check_input(t)
        return np.array(0.5*utils.square_norm(self.A.dot(x) - self.b[:,[k]])).item()
    
    def gradient(self, x, t):
        
        k = self.time.check_input(t)
        return self._H.dot(x) - self.A.T.dot(self.b[:,[k]])
    
    def hessian(self, x=None, t=None): # x, t arguments are only for compatibility
        
        return self._H
    
    def proximal(self, x, t, penalty=1):

        k = self.time.check_input(t)
        return la.solve(np.eye(self.dom.size) + penalty*self._H, x + penalty*self.b[:,[k]])


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
           and Linear Convergence of Proximal Methods," Mathematics of 
           Operations Research, vol. 43, no. 3, pp. 919–948, Aug. 2018.
    """
    
    def __init__(self, A, b, penalty=1):
        # each row in A and element of b represent a data point
        
        super().__init__(sets.R(A.shape[1], 1))
        
        self.m = A.shape[0] # domain dim. and num. data points
        # store data as lists
        self.A = [A[d,].reshape(self.dom.shape).T for d in range(self.m)]
        self.b = [np.array(b[d]).item() for d in range(self.m)]
        # store default penalty
        self.penalty = penalty
            
    def operator(self, x, penalty=None):
        
        x = self.dom.check_input(x)
        penalty = penalty if penalty is not None else self.penalty
        
        # define CVXPY problem
        y = cp.Variable(self.dom.shape)
        
        cost = cp.sum_squares(y - x) / (2*penalty)
        for d in range(self.m):
            c = (self.A[d].dot(x)).item()           
            cost += cp.abs(c**2 + 2*c*self.A[d] @ (y - x) - self.b[d]) / self.m
        
        problem = cp.Problem(cp.Minimize(cost))
        try:
            problem.solve()
        except:
            problem.solve(solver=cp.ECOS)
        
        return y.value.reshape((-1,1))

def dynamic_phases(n, num_pieces=15, len_pieces=10):
    r"""
    Generate a dynamic phase.
    
    This function generates a signal representing a piece-wise constant
    phase, with `num_pieces` constant parts, each time remaining constant
    for `len_pieces` steps.
    
    The phases are generated sampling from a normal distribution and 
    normalized to have unitary norm.
    """
    
    x = np.zeros((n, num_pieces*len_pieces))
    
    for l in range(num_pieces):
        
        # generate phase
        e = ran.normal(size=(n,1))
        e = e / la.norm(e)
        
        # store piecewise constant signal
        for i in range(l*len_pieces, (l+1)*len_pieces): x[:,[i]] = e

    return x

#%% SOLVERS

def fista(problem, step, x_0=0, num_iter=100, tol=None):
    r"""
    Fast iterative shrinkage.thresholding algorithm (FISTA).
    
    The function implements FISTA for solving the composite problem
    :math:`\min_x f(x) + g(x)`, with :math:`f` smooth. The algorithm is
    defined as follows, for :math:`\ell \in \mathbb{N}`:
    
    .. math:: \begin{align}
                  x^{\ell+1} &= \operatorname{prox}_{\alpha g}(y^\ell - \alpha \nabla f(y^\ell)) \\
                  t^{\ell+1} &= \frac{1 + \sqrt{4 (t^\ell)^2 + 1}}{2} \\
                  y^{\ell+1} &= x^{\ell+1} + \frac{t^\ell - 1}{t^{\ell+1}} (x^{\ell+1} - x^\ell)
              \end{align}
    
    where :math:`\alpha` is an appropriate step-size, and :math:`y^1 = x^0`,
    :math:`t^1 = 1`. See [#]_ for details and convergence.

    Parameters
    ----------
    problem : dict
        A dictionary containing the problem data, in particular the cost
        functions "f" and "g".
    step : float
        The step-size for the method.
    x_0 : array_like, optional
        The initial condition. This can be either an ndarray of suitable size,
        or a scalar. If it is a scalar then the same initial value is used for
        all components of :math:`x`.
    num_iter : int, optional
        The number of iterations to be performed.
    tol : float, optional
        If given, this argument specifies the tolerance :math:`t` in the 
        stopping condition :math:`\| x^{\ell+1} - x^\ell \| \leq t`.

    Returns
    -------
    x : ndarray
        The (approximate) solution.
    
    References
    ----------
    .. [#] A. Beck and M. Teboulle, "A Fast Iterative Shrinkage-Thresholding 
           Algorithm for Linear Inverse Problems," SIAM Journal on Imaging 
           Sciences, vol. 2, no. 1, pp. 183–202, Jan. 2009.
    """
    
    # unpack the problem
    f, g = problem["f"], problem["g"]
    
    # initialization
    x = np.zeros(f.dom.shape)
    x[...] = x_0
    
    y = x_0
    t = 1 # for inertial coefficient
    
    
    for l in range(num_iter):
        
        # store past iterate
        x_old = x
        
        # proximal gradient step
        x = g.proximal(y - step*f.gradient(y), step)
        
        if stop(x, x_old, tol): break
    
        # inertial coefficient
        t_old = t
        t = (1 + math.sqrt(t**2 + 1)) / 2
        # inertial step
        y = x + ((t_old-1)/t)*(x - x_old)
    
    return x

def backtracking_fista(problem, b, x_0=0, num_iter=100, tol=None):
    r"""

    """
    
    # unpack the problem
    f, g = problem["f"], problem["g"]
    
    # initialization
    x = np.zeros(f.dom.shape)
    x[...] = x_0
    
    y = x_0
    t = 1 # for inertial coefficient
    
    
    for l in range(num_iter):
        
        # store past iterate
        x_old = x
        
        # backtracking line search
        a = 1
        s = False
        while not s:
            
            p_l = g.proximal(y - a*f.gradient(y), a)
            
            if f.function(p_l) <= \
               f.function(y) + (p_l - y).T.dot(f.gradient(y)) \
               + (1/(2*a))*utils.square_norm(p_l - y):
                s = True
            else:
                a = b*a
        
        # proximal gradient step
        x = p_l
        
        if stop(x, x_old, tol): break
    
        # inertial coefficient
        t_old = t
        t = (1 + math.sqrt(t**2 + 1)) / 2
        # inertial step
        y = x + ((t_old-1)/t)*(x - x_old)
    
    return x

def guarded_anderson_acceleration(problem, m, step, x_0=None, num_iter=100, tol=None):
    r"""
    Anderson acceleration method.
    
    This function implements the Anderson acceleration (AA) method defined as 
    follows. Let :math:`\mathcal{T} : \mathbb{R}^n \to \mathbb{R}^n` be the 
    operator that we want to accelerate, and let `m` be the number of past
    iterates to be used in accelerating. The AA at each iteration 
    :math:`\ell \in \mathbb{N}` chooses the coefficients 
    :math:`\alpha^\ell = [\alpha_1^\ell, \ldots, \alpha_m^\ell]^\top` by solving:
        
     .. math:: \alpha^\ell = \operatorname{arg\,min}_{\alpha^\top \pmb{1} = 1} \| R^\ell \alpha^\ell \|
     
     where
     
     .. math:: R^\ell = \left[ (\mathcal{T}-\mathcal{I}) x^{\ell}, 
                               (\mathcal{T}-\mathcal{I}) x^{\ell-1}, \ldots, 
                               (\mathcal{T}-\mathcal{I}) x^{\ell-m+1} \right].
    
    After selecting the coefficients, the next iterate of the algorithm is 
    characterized by:
        
    .. math:: x^{\ell+1} = \sum_{i = 1}^m \alpha_i \mathcal{T} x^{\ell-i+1}.
    
    See [#]_ and references therein for more details.    

    Parameters
    ----------
    problem : dict
        A dictionary containing the problem data, in particular the operator
        with keyword "T".
    m : int
        The number of past iterates to be used in the acceleration scheme.
    step : float
        TODO The step-size
    x_0 : array_like, optional
        The initial condition. This can be either an ndarray of suitable size,
        or a scalar. If it is a scalar then the same initial value is used for
        all components of :math:`x`.
    num_iter : int, optional
        The number of iterations to be performed.
    tol : float, optional
        If given, this argument specifies the tolerance :math:`t` in the 
        stopping condition :math:`\| x^{\ell+1} - x^\ell \| \leq t`.

    Returns
    -------
    x : ndarray
        The (approximate) solution.
    
    References
    ----------
    .. [#] V. Mai and M. Johansson, "Anderson Acceleration of Proximal Gradient 
           Methods," in Proceedings of the 37th International Conference on 
           Machine Learning, Virtual, Jul. 2020, vol. 119, pp. 6620–6629.
    
    Notes
    -----
    The function uses a double ended queue, as implemented by 
    `collections.deque`, to store the `m` last iterates, so that the memory
    usage is as efficient as possible.
    
    In order to avoid singularity of the residual matrix :math:`R^\ell`, a
    Tikhonov regularization with weight :math:`10^{-10}` is added to the 
    problem that computes :math:`\alpha^\ell`, see [1]_.
    """
    
    f, g = problem["f"], problem["g"] # unpack the problem
    
    # initial condition
    if x_0 is None: x = np.zeros(f.dom.shape)
    else: x = x_0
    
    # queues for past information
    y = queue(x, maxlen=m)
    z = queue(maxlen=m)
    
    
    # ------ intial step
    y.append(x - step*f.gradient(x))
    x = g.proximal(y[-1], step)
    z.append(y[-1])
    
    
    # ------ following steps
    for l in range(1, num_iter):
        
        # num. of past iterate available
        m_l = min(m, l)
        
        # gradient step
        z.append(x - step*f.gradient(x))
        
        # compute extrapolation coefficients
        R = np.hstack([z[i] - y[i] for i in range(m_l)])
        
        if m_l == 1:
            a = [1]
        else:
            a = la.solve(R.T.dot(R) + 1e-10*np.eye(m_l), np.ones((m_l,1)))
            a = a / np.dot(np.ones(a.shape).T, a)
    
        # guarding procedure
        y_ext = sum([a[i]*z[i] for i in range(m_l)])
        x_test = g.proximal(y_ext, step)
        
        x_old = x
        if f.function(x_test) <= f.function(x) - (step/2)*utils.square_norm(f.gradient(x)):
            x = x_test
            y.append(y_ext)
        else:
            x = g.proximal(z[-1], step)
            y.append(z[-1])
        
        if stop(x, x_old, tol): break
        
    return x