#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Interpolation of Lipschitz continuous operators.
"""

import numpy as np
from numpy import linalg as la

from numpy.random import default_rng
ran = default_rng()

from tvopt import sets
from reg4opt import utils


#%% INTERPOLATION

def interpolator(x, x_i, t_i, zeta, t0=None, solver=None, **solver_params):
    r"""
    Operator interpolation.
    
    This function interpolates the :math:`\zeta`-Lipschitz continuous operator
    :math:`\mathcal{T} : \mathbb{R}^n \to \mathbb{R}^n` to the point :math:`x`,
    preserving the Lipschitz continuity.
    
    In particular, given the operator evaluations
    
    .. math:: t_i = \mathcal{T} x_i, \quad i = 1, \ldots, D
    
    the function computes :math:`\hat{\mathcal{T}} x` as
    
    .. math:: \hat{\mathcal{T}} x = \begin{cases}
                  t_i & \text{if} \ x = x_i \\
                  \hat{t} \in \bigcap_{i \in [D]} \mathbb{B}_{\zeta \|x - x_i\|}(t_i) & \text{otherwise}
              \end{cases}
    
    where :math:`\mathbb{B}_{\zeta \|x - x_i\|}(t_i)` is a ball of center 
    :math:`t_i` and radius :math:`\zeta \|x - x_i\|`. We see then that 
    interpolating requires finding a point in the intersection of balls. To 
    this end, an efficient approach is to use the method of alternating 
    projections (MAP) (see `alternating_projections`), but other solvers are
    available.

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
    solver : str
        The method that should be used to find a point in the intersection of
        balls. By default the MAP is used, due to its efficiency in this 
        scenario.
    solver_params : tuple
        Parameters for the solver, for example the maximum number of iterations
        that it can perform.

    Returns
    -------
    ndarray
        The interpolated operator evaluated at `x`.
    
    See Also
    --------
    alternating_projections : The method of alternating projections (MAP).
    
    Notes
    -----
    If the operator is defined in :math:`\mathbb{R}` (that is, :math:`n = 1`),
    then the balls become closed intervals on the line and the intersection
    can be computed in closed form.
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
        
        if solver is None:
            solver = alternating_projections

        return solver(s, t0, **solver_params)


#%% PROJECTION METHODS
    
# ------ FEASIBILITY

def alternating_projections(s, x=None, tol=1e-10, num_iters=1000):
    r"""
    Method of alternating projections.
    
    Given an initial point :math:`x^0`, this method computes for 
    :math:`\ell \in \mathbb{N}`:
    
    .. math:: x^{\ell+1} = \operatorname{proj}_n \circ \cdots \circ \operatorname{proj}_1 x^\ell
    
    where :math:`\operatorname{proj}_i` is the projection operator onto the
    :math:`i`-th set in the list `s`. See [#]_ for details and convergence.

    Parameters
    ----------
    s : list
        A list of `Set` objects defining the problem.
    x : array_like, optional
        The initial point :math:`x^0`, defaults to a randomly picked one if
        not given.
    tol : float, optional
        The algorithm is stopped if the fixed point residual 
        :math:`\|x^{\ell+1} - x^\ell\|` is smaller than `tol`.
    num_iters : int, optional
        The maximum number of iterations that the algorithm can perform.

    Returns
    -------
    x : ndarray
        A point in the intersection of the given sets.
    
    References
    ----------
    .. [#] H. Bauschke and V. Koch, "Projection Methods: Swiss Army Knives for
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
    r"""
    Method of parallel projections.
    
    Given an initial point :math:`x^0`, this method computes for 
    :math:`\ell \in \mathbb{N}`:
    
    .. math:: x^{\ell+1} = \frac{1}{n} \sum_{i = 1}^n \operatorname{proj}_i x^\ell
    
    where :math:`\operatorname{proj}_i` is the projection operator onto the
    :math:`i`-th set in the list `s`. See [#]_ for details and convergence.

    Parameters
    ----------
    s : list
        A list of `Set` objects defining the problem.
    x : array_like, optional
        The initial point :math:`x^0`, defaults to a randomly picked one if
        not given.
    tol : float, optional
        The algorithm is stopped if the fixed point residual 
        :math:`\|x^{\ell+1} - x^\ell\|` is smaller than `tol`.
    num_iters : int, optional
        The maximum number of iterations that the algorithm can perform.

    Returns
    -------
    x : ndarray
        A point in the intersection of the given sets.
    
    References
    ----------
    .. [#] H. Bauschke and V. Koch, "Projection Methods: Swiss Army Knives for
           Solving Feasibility and Best Approximation Problems with 
           Halfspaces," in Contemporary Mathematics, vol. 636, S. Reich 
           and A. Zaslavski, Eds. Providence, Rhode Island: 
           American Mathematical Society, 2015, pp. 1–40.
    """
    
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
    r"""
    Peaceman-Rachford splitting (PRS).
    
    Given an initial point :math:`x^0`, this method computes for 
    :math:`\ell \in \mathbb{N}`:
    
    .. math:: \begin{align}
                  x_i^{\ell+1} &= \operatorname{proj}_i z_i^\ell \\
                  \bar{x}^{\ell+1} &= \frac{1}{n} \sum_{i = 1}^n x_i^{\ell+1} \\
                  z_i^{\ell+1} &= (1-a) z_i^\ell + a (2 \bar{x}^{\ell+1} - x_i^{\ell+1})
              \end{align}
    
    where :math:`\operatorname{proj}_i` is the projection operator onto the
    :math:`i`-th set in the list `s`, and :math:`a \in (0,1]`. See [#]_ for 
    details and convergence.

    Parameters
    ----------
    s : list
        A list of `Set` objects defining the problem.
    x : array_like, optional
        The initial point :math:`x^0`, defaults to a randomly picked one if
        not given.
    a : float, optional
        The relaxation constant of the Peaceman-Rachford, defaults to 0.95.
    tol : float, optional
        The algorithm is stopped if the fixed point residual 
        :math:`\|x^{\ell+1} - x^\ell\|` is smaller than `tol`.
    num_iters : int, optional
        The maximum number of iterations that the algorithm can perform.

    Returns
    -------
    x : ndarray
        A point in the intersection of the given sets.
    
    References
    ----------
    .. [#] H. Bauschke and V. Koch, "Projection Methods: Swiss Army Knives for
           Solving Feasibility and Best Approximation Problems with 
           Halfspaces," in Contemporary Mathematics, vol. 636, S. Reich 
           and A. Zaslavski, Eds. Providence, Rhode Island: 
           American Mathematical Society, 2015, pp. 1–40.
    """
    
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
    r"""
    Dykstra projection method.
    
    Given an initial point :math:`x^0 = v`, this method computes for 
    :math:`\ell \in \mathbb{N}`:
    
    .. math:: \begin{align}
                  \bar{x}^{\ell+1} &= \frac{1}{n} \sum_{i = 1}^n (x_i^\ell + p_i^\ell) \\
                  p_i^{\ell+1} &= x_i^\ell + p_i^\ell - \bar{x}^{\ell+1} \\
                  x_i^{\ell+1} &= \operatorname{proj}_i(\bar{x}^{\ell+1} + q_i^\ell) \\
                  q_i^{\ell+1} &= \bar{x}^{\ell+1} + q_i^\ell - x_i^{\ell+1}
              \end{align}
    
    where :math:`\operatorname{proj}_i` is the projection operator onto the
    :math:`i`-th set in the list `s`. See [#]_ and [#]_ for details and 
    convergence.

    Parameters
    ----------
    s : list
        A list of `Set` objects defining the problem.
    v : array_like
        The point to be projected.
    tol : float, optional
        The algorithm is stopped if the fixed point residual 
        :math:`\|x^{\ell+1} - x^\ell\|` is smaller than `tol`.
    num_iters : int, optional
        The maximum number of iterations that the algorithm can perform.

    Returns
    -------
    x : ndarray
        An approximate projection of `v` onto the intersection of the sets.
    
    References
    ----------
    .. [#] H. Bauschke and V. Koch, "Projection Methods: Swiss Army Knives for
           Solving Feasibility and Best Approximation Problems with 
           Halfspaces," in Contemporary Mathematics, vol. 636, S. Reich 
           and A. Zaslavski, Eds. Providence, Rhode Island: 
           American Mathematical Society, 2015, pp. 1–40.
    .. [#] H. H. Bauschke, R. S. Burachik, D. B. Herman, and C. Y. Kaya, "On
           Dykstra's algorithm: finite convergence, stalling, and the method of
           alternating projections," Optimization Letters, vol. 14, no. 8, 
           pp. 1975–1987, Nov. 2020.
    """
    
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
    r"""
    Halpern projection method.
    
    Given an initial point :math:`x^0 = v`, this method computes for 
    :math:`\ell \in \mathbb{N}`:
    
    .. math:: \begin{align}
                  y^{\ell+1} &= \operatorname{proj}_n \circ \cdots \circ \operatorname{proj}_1 x^\ell \\
                  x^{\ell+1} &= \frac{v}{l+1} + \frac{l}{l+1} y^{\ell+1}
              \end{align}
    
    where :math:`\operatorname{proj}_i` is the projection operator onto the
    :math:`i`-th set in the list `s`. See [#]_ and 
    [#]_ for details and convergence.

    Parameters
    ----------
    s : list
        A list of `Set` objects defining the problem.
    v : array_like
        The point to be projected.
    tol : float, optional
        The algorithm is stopped if the fixed point residual 
        :math:`\|x^{\ell+1} - x^\ell\|` is smaller than `tol`.
    num_iters : int, optional
        The maximum number of iterations that the algorithm can perform.

    Returns
    -------
    x : ndarray
        An approximate projection of `v` onto the intersection of the sets.
    
    References
    ----------
    .. [#] H. Bauschke and V. Koch, "Projection Methods: Swiss Army Knives for
           Solving Feasibility and Best Approximation Problems with 
           Halfspaces," in Contemporary Mathematics, vol. 636, S. Reich 
           and A. Zaslavski, Eds. Providence, Rhode Island: 
           American Mathematical Society, 2015, pp. 1–40.
    .. [#] H. H. Bauschke and P. L. Combettes, Convex analysis and monotone 
           operator theory in Hilbert spaces, 2nd ed. Cham: Springer, 2017.
    """
    
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
    r"""
    Haugazeau projection method.
    
    Given an initial point :math:`x^0 = v`, this method computes for 
    :math:`\ell \in \mathbb{N}`:
    
    .. math:: x^{\ell+1} = Q(v, x^\ell, \operatorname{proj}_i x^\ell)
    
    choosing :math:`i = 1, \ldots, n` sequentially, and where 
    :math:`\operatorname{proj}_i` is the projection operator onto the
    :math:`i`-th set in the list `s`, and
    
    .. math:: Q(x, y, z) = \begin{cases}
                  z & \text{if} \ d = 0, a \geq 0 \\
                  x + (1 + a/c) (z - y) & \text{if} \ d > 0, a c \geq d \\
                  y + (c/d) (a(x - y) + b(z - y)) & \text{if} \ d > 0, a c < d \\
                  \text{undefined otherwise}
              \end{cases}
    
    where :math:`a = (x - y)^\top (y - z)`, :math:`b = \|x - y\|^2`, 
    :math:`c = \|y - z\|^2`, and :math:`d = b c - a^2`. See [#]_ for details 
    and convergence.

    Parameters
    ----------
    s : list
        A list of `Set` objects defining the problem.
    v : array_like
        The point to be projected.
    tol : float, optional
        The algorithm is stopped if the fixed point residual 
        :math:`\|x^{\ell+1} - x^\ell\|` is smaller than `tol`.
    num_iters : int, optional
        The maximum number of iterations that the algorithm can perform.

    Returns
    -------
    x : ndarray
        An approximate projection of `v` onto the intersection of the sets.
    
    References
    ----------
    .. [#] H. Bauschke and V. Koch, "Projection Methods: Swiss Army Knives for
           Solving Feasibility and Best Approximation Problems with 
           Halfspaces," in Contemporary Mathematics, vol. 636, S. Reich 
           and A. Zaslavski, Eds. Providence, Rhode Island: 
           American Mathematical Society, 2015, pp. 1–40.
    """
    
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