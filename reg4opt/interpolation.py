#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from numpy import linalg as la

from numpy.random import default_rng
ran = default_rng()

from tvopt import sets
from reg4opt import utils


#%% INTERPOLATION

def interpolator(x, x_i, t_i, zeta, t0=None):
    """
    Operator interpolation.

    Parameters
    ----------
    x : array_like
        The point where the interpolated operator should be evaluated.
    x_i : list
        The points of the regression problem.
    t_i : list
        The solution of the regression problem.
    zeta : float
        The contraction constant.
    t0 : array_like, optional
        The initial condition for the projection algorithm, defaults to a 
        vector of zeros.

    Returns
    -------
    ndarray
        The interpolated operator evaluated at `x`.
    """
    
    d = len(x_i) # num. of data points
    
    # ------ check if `x` is in `x_i`
    for i in range(d):
        
        if np.allclose(x, x_i[i]): return t_i[i]

    
    # ------ otherwise, perform interpolation
    # 1-dimensional case
    if np.size(x) == 1:
        
        # compute lower and upper bounds of the intersection
        low, up = -np.inf, np.inf
        
        for i in range(d):
            
            r = zeta*abs(x - x_i[i]) # radius
            low, up = max(low, float(t_i[i] - r)), min(up, float(t_i[i] + r))
                
        return (up + low) / 2
        
    # multi-dimensional case
    else:
        # generate the ball sets
        s = [sets.Ball(t_i[i], zeta*utils.norm(x - x_i[i])) for i in range(d)]
        
        # choose the initial condition
        if t0 is None: t0 = ran.standard_normal(np.shape(x))
        
        return alternating_projections(s, t0)


#%% PROJECTION METHODS
    
# ------ FEASIBILITY

def alternating_projections(s, x=None, tol=1e-10, num_iters=1000):
    """
    Method of alternating projections.
    
    method of alternating 
    projections (MAP) (see [1]__ and references therein).

    Parameters
    ----------


    Returns
    -------
    x : ndarray
        The (possibly approximate) projection onto the intersection.
    
    References
    ----------
    .. [1] H. Bauschke and V. Koch, "Projection Methods: Swiss Army Knives for
           Solving Feasibility and Best Approximation Problems with 
           Halfspaces," in Contemporary Mathematics, vol. 636, S. Reich 
           and A. Zaslavski, Eds. Providence, Rhode Island: 
           American Mathematical Society, 2015, pp. 1–40.
    """
    
    x = ran.standard_normal(s[0].shape) if x is None else x # initial condition
    x_old = np.copy(x)
    
        
    for l in range(num_iters):
    
        for i in range(len(s)):
            
            x = s[i].projection(x)
            
        # check stopping condition
        if utils.norm(x - x_old) <= tol: break
        else: x_old = np.copy(x)
    
    return x

def parallel_projections(s, x=None, tol=1e-10, num_iters=1000):
    
    x = ran.standard_normal(s[0].shape) if x is None else x # initial condition
    x_old = np.copy(x)
    
    
    for l in range(num_iters):
        
        y = [s[i].projection(x) for i in range(len(s))]
    
        x = np.mean(y, axis=0)
                    
        # check stopping condition
        if utils.norm(x - x_old) <= tol: break
        else: x_old = np.copy(x)
    
    return x

def peaceman_rachford(s, x=None, a=0.95, tol=1e-10, num_iters=1000):
    
    d = len(s)
    
    z0 = ran.standard_normal(s[0].shape) if x is None else x # initial condition    
    z, x = np.tile(z0, d), np.tile(z0, d)
    x_bar_old = z0
    
    
    for l in range(num_iters):
        
        # update local estimates
        for i in range(d):
            
            x[:,[i]] = s[i].projection(z[:,[i]])
        
        # compute average
        x_bar = np.mean(x, axis=1, keepdims=True)
        
        # update z
        for i in range(d):
            
            z[:,[i]] = (1-a)*z[:,[i]] + a*(2*x_bar - x[:,[i]])
        
        # check stopping condition
        if utils.norm(x_bar - x_bar_old) <= tol: break
        else: x_bar_old = np.copy(x_bar)
    
    return x_bar


# ------ BEST APPROXIMATION

def dykstra(s, v, tol=1e-10, num_iters=1000):
    
    d, n = len(s), np.size(v) # num. of balls and domain dimension
    
    # initialize variables
    x, p, q = np.tile(v, (1, d)), np.zeros((n, d)), np.zeros((n, d))
    x_old = np.copy(x)
    
        
    for l in range(num_iters):
        
        # average
        x_bar = np.mean(x + p, axis=1, keepdims=True)
            
        # p, x and q updates
        for i in range(d):
            
            p[:,[i]] = x[:,[i]] + p[:,[i]] - x_bar
            
            x[:,[i]] = s[i].projection(x_bar + q[:,[i]])
            
            q[:,[i]] = x_bar + q[:,[i]] - x[:,[i]]
            
        # check stopping condition
        if utils.norm(x - x_old) <= tol: break
        else: x_old = np.copy(x)
    
    return x

def halpern(s, v, tol=1e-10, num_iters=1000):
    
    x, x_old = v, v
    
    
    for l in range(1,num_iters):
        
        # perform alternating projections
        for i in range(len(s)):
            
            x = s[i].projection(x)
        
        # update
        x = v / (l+1) + (l/(l+1))*x
            
        # check stopping condition
        if utils.norm(x - x_old) <= tol: break
        else: x_old = np.copy(x)
    
    return x

def haugazeau(s, v, tol=1e-10, num_iters=1000):
    
    x, x_old = v, v
    
    
    for l in range(num_iters):
        
        for i in range(len(s)):
            
            x = _haugazeau_operator(v, x, s[i].projection(x))
        
        # check stopping condition
        if utils.norm(x - x_old) <= tol: break
        else: x_old = np.copy(x)
    
    return x

def _haugazeau_operator(x, y, z):
    
    a = (x - y).T.dot(y - z)
    b = la.norm(x - y)**2
    c = la.norm(y - z)**2
    d = b*c - a**2
    
    if d == 0 and a >= 0:
        return z
    elif d > 0 and a*c >= d:
        return x + (1 + a/c)*(z - y)
    elif d > 0 and a*c < d:
        return y + (c/d)*(a*(x - y) + b*(z - y))
    else:
        return None