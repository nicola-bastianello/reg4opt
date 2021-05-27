#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Solvers for operator and convex regression.
"""

import numpy as np

from reg4opt import utils 


#%% REGRESSION METHODS

def operator_regression(x, y, zeta, **kwargs):
    r"""
    Operator regression.
    
    The function solves an operator regression problem
    
    .. math:: \begin{align}
                  &\min_{t_i, \ i \in [D]} \frac{1}{D} \sum_{i = 1}^D \| t_i - y_i \|^2 \\
                  &\text{s.t.} \ \| t_i - t_j \|^2 \leq \zeta^2 \| y_i - y_j \|^2, \quad 1 \leq i < j \leq D
              \end{align}
    
    where :math:`y_i = \mathcal{T} x_i`, :math:`i = 1, \ldots, D`, are 
    evaluations of the operator to be approximated, and :math:`\zeta \in (0, 1]`
    is the Lipschitz continuity modulus for the approximated operator.
    
    In case :math:`D = 2`, the function returns a closed form solution, see
    `closed_form_solution_or`. If :math:`D > 2` the solution is (approximately)
    computed using a solver based on the Peaceman-Rachford splitting, 
    see `prs_solver_or`.
    
    Notice that the training data `x` and `y` should be lists of column numpy
    arrays (for the sake of efficiency, no check is applied to them).

    Parameters
    ----------
    x : list
        The points :math:`x_i` where the original operator is evaluated.
    y : list
        The evaluations of the operator :math:`y_i = \mathcal{T} x_i`.
    zeta : float
        The Lipschitz modulus for the reconstructed operator.
    **kwargs : tuple
        Optional arguments for the PRS solver.

    Returns
    -------
    list
        The approximate operator evaluated in the points :math:`x_i`.
    float
        The Lipschitz modulus (either the given one or the one computed via
        autotuning).
    """
        
    if len(x) == 2:
        return closed_form_solution_or(x, y, zeta)
    else:
        return prs_solver_or(x, y, zeta, **kwargs)

        
def convex_regression(x, y, mu, L, w=None, **kwargs):
    r"""
    Convex regression.
    
    The function solves a convex regression problem
    
    
    .. math:: \begin{align}
                  &\min_{f_i, g_i, \ i \in [D]} \frac{1}{D} \sum_{i = 1}^D (f_i - y_i)^2 \\
                  &\text{s.t.} \ f_i - f_j - \langle g_j, x_i - x_j \rangle \geq \\
                  &\qquad \frac{1}{2(1 - \mu / L)} \left( \frac{1}{L} \| g_i - g_j \|^2 
                  + \mu \| x_i - x_j \|^2 - \frac{2\mu}{L} \langle g_j - g_i, x_j - x_i \rangle \right), \quad 1 \leq i < j \leq D
              \end{align}
    
    where :math:`y_i = f(x_i)`, :math:`i = 1, \ldots, D`, are 
    evaluations of the function to be approximated, and :math:`0 < \mu < L`
    are the strong convexity and smoothness moduli for the target function.
    
    The function can also incorporate gradient information, in which case the
    problem becomes:
    
    .. math:: \begin{align}
                  &\min_{f_i, g_i, \ i \in [D]} \frac{1}{D} \sum_{i = 1}^D (f_i - y_i)^2 + \| g_i - w_i \|^2 \\
                  &\text{s.t.} \ f_i - f_j - \langle g_j, x_i - x_j \rangle \geq \\
                  &\qquad \frac{1}{2(1 - \mu / L)} \left( \frac{1}{L} \| g_i - g_j \|^2 
                  + \mu \| x_i - x_j \|^2 - \frac{2\mu}{L} \langle g_j - g_i, x_j - x_i \rangle \right), \quad 1 \leq i < j \leq D
              \end{align} 
              
    with :math:`w_i = \nabla f_i(x_i)`.
    
    In case :math:`D = 2` (and gradient information is given), the function 
    returns a closed form solution, see `closed_form_solution_cr`. Otherwise
    the solution is (approximately) computed using a solver based on the 
    Peaceman-Rachford splitting, see `prs_solver_cr`.
    
    Notice that the training data `x` and `y` (and the optional `w`) should be
    lists of column numpy arrays (for the sake of efficiency, no check is 
    applied to them).

    Parameters
    ----------
    x : list
        The points :math:`x_i` where the original function is evaluated.
    y : list
        The evaluations of the function :math:`y_i = f(x_i)`.
    mu : float
        The strong convexity modulus of the target function.
    L : float
        The smoothness modulus of the target function.
    w : list, optional
        Optional gradient evaluations to add data to the regression problem.
    **kwargs : tuple
        Arguments that should be passed to the solver.

    Returns
    -------
    list
        The approximate function evaluated in the points :math:`x_i`.
    list
        The gradient of the approximate function evaluated in the points 
        :math:`x_i`.
    """
    
    if len(x) == 2 and w is not None:
        return closed_form_solution_cr(x, y, w, mu, L)
    else:
        return prs_solver_cr(x, y, mu, L, w=w, **kwargs)


#%% OPERATOR REGRESSION

# ------------ closed form solution (w/ 2 training points)
def closed_form_solution_or(x, y, zeta):
    r"""
    Closed form solution of operator regression with 2 training points.

    Parameters
    ----------
    x : list
        The points :math:`x_i` where the original operator is evaluated.
    y : list
        The evaluations of the operator :math:`y_i = \mathcal{T} x_i`.
    zeta : float
        The Lipschitz modulus for the reconstructed operator.

    Returns
    -------
    list
        The approximate operator evaluated in the points :math:`x_i`.
    """
    
    # optimal Lagrange multiplier    
    ll = max(0, 0.5*(utils.norm(y[0] - y[1])/(zeta*utils.norm(x[0] - x[1])) - 1))
    
    # return primal solution
    return [((1+ll)*y[0] + ll*y[1]) / (1 + 2*ll), ((1+ll)*y[1] + ll*y[0]) / (1 + 2*ll)]


# ------------ PRS solver
def prs_solver_or(x, y, zeta, tol=1e-2, num_iter=1e3, rho=1):
    r"""
    Operator regression solver using PRS.
    
    This function implements a tailored solver for operator regression based
    on the Peaceman-Rachford splitting.

    Parameters
    ----------
    x : list
        The points :math:`x_i` where the original operator is evaluated.
    y : list
        The evaluations of the operator :math:`y_i = \mathcal{T} x_i`.
    zeta : float
        The Lipschitz modulus for the reconstructed operator.
    tol : float, optional
        The solver stops if the fixed point residual 
        :math:`\| x^{\ell+1} - x^\ell \|` is below this tolerance.
    num_iter : int, optional
        The maximum number of iterations that the solver can perform.
    rho : float, optional
        The penalty parameter for PRS, must be a positive scalar.

    Returns
    -------
    list
        The approximate operator evaluated in the points :math:`x_i`.
    """
    
    # -------- problem information
    d, n = len(x), np.size(x[0]) # num. of data points and size of domain
    
    # list of ordered data points pairs
    E = [(i,j) for i in range(d-1) for j in range(i+1,d)]
    
    
    # -------- initialization
    c = 1/(d-1) + 1/rho
    q_0 = {e : zeta*utils.norm(x[e[0]] - x[e[1]]) for e in E}
    
    # initialize the variables
    xi = {e : np.zeros((2*n,1)) for e in E}
    z = {e : np.zeros((2*n,1)) for e in E}
    z_old = np.zeros((2*n,1)) # memory slot to compute FPR

    # -------- solve the regression
    for k in range(int(num_iter)):
        
        if k == 0:
            v = [np.zeros((n,1)) for _ in range(d)]
        else:
            for i in range(d): v[i][...] = 0
        
        # perform the local updates and (recursively) project onto the 
        # consensus constraints
        for e in E:
            
            w_1, w_2 = (y[e[0]]/(d-1) + z[e][:n,]/rho)/c, (y[e[1]]/(d-1) + z[e][n:,]/rho)/c
            ll = max(0, 0.5*(utils.norm(w_1 - w_2)/q_0[e] - 1))
            
            xi[e][:n,] = ((1+ll)*w_1 + ll*w_2) / (1 + 2*ll)
            xi[e][n:,] = ((1+ll)*w_2 + ll*w_1) / (1 + 2*ll)

            t = (2*xi[e] - z[e]) / (d-1)
            v[e[0]] += t[:n,]
            v[e[1]] += t[n:,]
                    
        # auxiliary variables update, and compute the fixed point residual 
        # (distance between consecutive iterates)
        fpr = 0 # fixed point residual for stopping condition
        
        for e in E:
            
            # store the previous iterate and update
            z_old = z[e]
            z[e] = z[e] + 2*(np.vstack((v[e[0]], v[e[1]])) - xi[e])
            # compute the fixed point residual
            fpr = np.maximum(fpr, utils.square_norm(z[e] - z_old))
        
        # check stopping condition
        if fpr <= tol: break
    
    return [xi[(i,i+1)][:n,] for i in range(d-1)]  + [xi[(d-2,d-1)][n:,]]


#%% CONVEX REGRESSION

# ------------ closed form solution (w/ 2 training points)
def closed_form_solution_cr(x, y, w, mu, L):
    r"""
    Closed form solution of convex regression with 2 training points.

    Parameters
    ----------
    x : list
        The points :math:`x_i` where the original function is evaluated.
    y : list
        The evaluations of the function :math:`y_i = f(x_i)`.
    w : list
        The valuations of the gradient :math:`w_i = \nabla f(x_i)`.
    mu : float
        The strong convexity modulus of the target function.
    L : float
        The smoothness modulus of the target function.

    Returns
    -------
    list
        The approximate function evaluated in the points :math:`x_i`.
    list
        The gradient of the approximate function evaluated in the points 
        :math:`x_i`.
    """
    
    x_diff = x[0] - x[1]
    
    l_plus = 0.5*(-1 + utils.norm((L+mu)*x_diff - 2*(w[0]-w[1])) / ((L-mu)*utils.norm(x_diff)))
    l_minus = (x_diff.T.dot(w[0]+w[1]) - 2*(y[0]-y[1])) / ((L-mu)*(utils.square_norm(x_diff)+4))
    
    l_1, l_2 = max(0, 0.5*(l_plus+l_minus)), max(0, 0.5*(l_plus-l_minus))
    
    # temporary vector
    t = l_plus*(w[0]+w[1]) - (L-mu)*l_plus*l_minus*x_diff
    
    return [y[0] + (L-mu)*l_minus, y[1] - (L-mu)*l_minus], \
           [(w[0] + t + (mu*l_1+L*l_2)*x_diff)/(1+2*l_plus), (w[1] + t - (L*l_1+mu*l_2)*x_diff)/(1+2*l_plus)]


# ------------ PRS solver
def prs_solver_cr(x, y, mu, L, w=None, tol=1e-2, num_iter=1e3, rho=1):
    r"""
    Convex regression solver using PRS.
    
    This function implements a tailored solver for convex regression based
    on the Peaceman-Rachford splitting. The solver can optionally exploit
    gradient information in the cost.

    Parameters
    ----------
    x : list
        The points :math:`x_i` where the original function is evaluated.
    y : list
        The evaluations of the function :math:`y_i = f(x_i)`.
    mu : float
        The strong convexity modulus of the target function.
    L : float
        The smoothness modulus of the target function.
    w : list, optional
        Optional evaluations of the gradient :math:`w_i = \nabla f(x_i)`.
    tol : float, optional
        The solver stops if the fixed point residual 
        :math:`\| x^{\ell+1} - x^\ell \|` is below this tolerance.
    num_iter : int, optional
        The maximum number of iterations that the solver can perform.
    rho : float, optional
        The penalty parameter for PRS, must be a positive scalar.

    Returns
    -------
    list
        The approximate function evaluated in the points :math:`x_i`.
    list
        The gradient of the approximate function evaluated in the points 
        :math:`x_i`.
    """

    
    # -------- problem information
    d, n = len(x), np.size(x[0]) # num. of data points and size of domain
    
    # list of ordered data points pairs
    E = [(i,j) for i in range(d-1) for j in range(i+1,d)]
    
    
    # -------- generate cost and constraints data
    c = 1 + rho/(d-1)
    
    q, x_diff = {}, {}
    for e in E:
        
        if w is not None:
            q[e] = np.vstack((y[e[0]], y[e[1]], w[e[0]], w[e[1]])) / (1 + (d-1)/rho)
        else:
            q[e] = np.vstack((y[e[0]], y[e[1]], np.zeros((2*n,1)))) / (1 + (d-1)/rho)
        
        x_diff[e] = x[e[0]] - x[e[1]]

    
    # initialize the variables
    xi = {e : np.zeros((2*(n+1),1)) for e in E}
    z = {e : np.zeros((2*(n+1),1)) for e in E}
    z_old = np.zeros((2*(n+1),1)) # memory slot to compute FPR

    # -------- solve the regression
    for k in range(int(num_iter)):
        
        if k == 0:
            v = [np.zeros((n+1,1)) for _ in range(d)]
        else:
            for i in range(d): v[i][...] = 0
        
        
        # solve the local updates
        for e in E:
            
            y = q[e] + z[e] / c # temporary vector for the cost
            
            # compute Lagrange multipliers           
            l_plus = 0.5*(-1 + utils.norm((L+mu)*x_diff[e] - 2*(y[2:n+2,]-y[n+2:,])) / ((L-mu)*utils.norm(x_diff[e])))
            l_minus = (x_diff[e].T.dot(y[2:n+2,]+y[n+2:,]) - 2*(y[0]-y[1])) / ((L-mu)*(utils.square_norm(x_diff[e])+4))
            
            l_1, l_2 = max(0, 0.5*(l_plus+l_minus)), max(0, 0.5*(l_plus-l_minus))
            
            
            # compute the primal solution
            xi[e][:2,] = y[:2,] - (L-mu)*l_minus*np.array([[-1],[1]])
            
            t = l_plus*(y[2:n+2,]+y[n+2:,]) - (L-mu)*l_plus*l_minus*x_diff[e]
            xi[e][2:,] = (1/(1+2*l_plus))*(y[2:,] + np.vstack((t + (mu*l_1+L*l_2)*x_diff[e], t - (L*l_1+mu*l_2)*x_diff[e])))
            
            
            # project onto consensus space
            t = (2*xi[e] - z[e]) / (d-1)
            v[e[0]][0] += t[0]
            v[e[0]][1:,] += t[2:n+2,]
            v[e[1]][0] += t[1]
            v[e[1]][1:,] += t[n+2:,]
            
            
        # auxiliary variables update, and compute the fixed point residual 
        # (distance between consecutive iterates)
        fpr = 0 # fixed point residual for stopping condition
        
        for e in E:
            
            # store the previous iterate and update
            z_old = z[e]
            z[e] = z[e] + 2*(np.vstack((v[e[0]][0], v[e[1]][0], v[e[0]][1:,], v[e[1]][1:,])) - xi[e])
            # compute the fixed point residual
            fpr = np.maximum(fpr, utils.square_norm(z[e] - z_old))
        
        # check stopping condition
        if fpr <= tol: break

    return [xi[(i,i+1)][0] for i in range(d-1)]  + [xi[(d-2,d-1)][0]], \
           [xi[(i,i+1)][2:n+2,] for i in range(d-1)]  + [xi[(d-2,d-1)][2:n+2,]]