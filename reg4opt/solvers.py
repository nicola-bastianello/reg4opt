#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions.
"""

import numpy as np
from numpy import linalg as la

from numpy.random import default_rng
ran = default_rng()

from tvopt.solvers import stop

from collections import deque as queue



def nesterov_gradient(problem, step, inertia, x_0=0, num_iter=100, tol=None):
    r"""
    """
    
    f = problem["f"]
    x = np.zeros(f.dom.shape)
    x[...] = x_0
    x_old = x_0
    y = x_0
    
    
    for l in range(num_iter):
        
        x_old_old = x_old
        x_old = x
        
        # gradient step
        x = x - step*f.gradient(y) + inertia*(x - x_old_old)
        
        if stop(x, x_old, tol): break
    
        y = x + inertia*(x - x_old)
    
    return x


def anderson_acceleration(problem, m, x_0=None, num_iter=100, tol=None):
    r"""
    """
    
    T = problem["T"]
    
    # create a queue for the iterates history (rightmost element is newest)
    if x_0 is None: x_0 = [np.zeros(T.dom.shape)]
    if not isinstance(x_0, list): x_0 = [x_0]
    x_old = queue(x_0, maxlen=m)

    
    # first step
    x = T.operator(x_old[-1])
    x_old.append(x)
    
    
    # remaining steps
    for l in range(1,num_iter):
        
        m_l = min(m, l)
        
        # compute residuals
        R = np.hstack([np.reshape(T.operator(x_old[-i])-x_old[-i], (-1,1)) for i in range(1, m_l+1)])

        # compute extrapolation coefficients
        if m_l == 1:
            a = [1]
        else:
            a = la.solve(R.T.dot(R), np.ones((m_l,1)))
            a = a / np.dot(np.ones(a.shape).T, a)
        
        # compute next iterate
        x = np.sum([a[i-1]*T.operator(x_old[-i]) for i in range(1, m_l+1)], axis=0)
        
        if stop(x, x_old[-1], tol): break
        x_old.append(x)
        
    # return list(x_old)
    return x



