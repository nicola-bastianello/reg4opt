#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Solvers.
"""

import math
import numpy as np
from numpy import linalg as la

from numpy.random import default_rng
ran = default_rng()

from tvopt.solvers import stop

from collections import deque as queue


#%% SOLVERS FOR SMOOTH PROBLEMS

def nesterov_gradient(problem, step, inertia, x_0=0, num_iter=100, tol=None):
    r"""
    Nesterov gradient method.
    
    This function implements the Nesterov accelerated gradient described by the
    following equations, for :math:`\ell \in \mathbb{N}`:
    
    .. math:: \begin{align}
                  x^{\ell+1} &= x^\ell - \alpha \nabla f(y^\ell) + \beta (x^\ell - x^{\ell-1}) \\
                  y^{\ell+1} &= x^{\ell+1} + \beta (x^{\ell+1} - x^{\ell})
              \end{align}
    
    where :math:`\alpha` and :math:`\beta` are the step-size and inertial
    coefficient, respectively, and where :math:`y^0 = x^0`.

    Parameters
    ----------
    problem : dict
        A dictionary containing the problem data, in particular the cost
        function "f".
    step : float
        The step-size for the method.
    inertia : float
        The inertial coefficient for the method.
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
    .. [#] Y. Nesterov, Lectures on Convex Optimization, vol. 137. Cham: 
           Springer International Publishing, 2018.
    """
    
    # unpack the problem
    f = problem["f"]
    
    # initialization
    x = np.zeros(f.dom.shape)
    x[...] = x_0
    x_old = x_0
    y = x_0
    
    
    for l in range(num_iter):
        
        # store past iterates (the two previous ones)
        x_old_old = x_old
        x_old = x
        
        # gradient step
        x = x - step*f.gradient(y) + inertia*(x - x_old_old)
        
        if stop(x, x_old, tol): break
    
        # inertial step
        y = x + inertia*(x - x_old)
    
    return x


#%% ACCELERATION SCHEMES FOR OPERATORS

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

def anderson_acceleration(problem, m, x_0=None, num_iter=100, tol=None):
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
    """
    
    T = problem["T"] # unpack the operator
    
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
        
    return x