#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convex and operator regression methods.
"""

import math
import numpy as np
import cvxpy as cp

from numpy.random import default_rng
ran = default_rng()

from reg4opt import utils 

#TODO
# - document and comment (say that the data should be column numpy array)


#%% OPERATOR REGRESSION

# def operator_regression(x, y, c, autotune=False, solver="CVXPY", **solver_params):
    
#     solver = solver.strip().lower()
    
#     if solver == "cvxpy" or solver == "c" or solver == "cp":
#         return cvxpy_solver(x, y, c, autotune, **solver_params)
    
#     elif solver == "prs" or solver == "p":
#         if not autotune:
#             return prs_solver(x, y, c, **solver_params), c
#         else:
#             return prs_solver_autotuning(x, y, c, **solver_params)

def operator_regression(x, y, c, autotune=False, solver="CVXPY", **solver_params):
    
    solver = solver.strip().lower()
    
    if solver == "cvxpy" or solver == "c" or solver == "cp":
        return cvxpy_solver(x, y, c, autotune, **solver_params)
    
    elif solver == "prs" or solver == "p":
        if not autotune:
            return prs_solver(x, y, c, **solver_params), c
        else:
            return prs_solver_autotuning(x, y, c, **solver_params)
    
    elif solver == "prs_2" or solver == "p2":
        return prs_solver_2(x, y, c, **solver_params), c


#%% CVXPY SOLVER

def cvxpy_solver(x, y, c, autotune=False, cp_solver=None, **solver_params):

    # -------- problem information
    d, n = len(x), np.size(x[0]) # num. data points and domain dimension
    
    # generate costs and constraints
    t = cp.Variable(d*n)
    
    # define the cost
    cost = 0.5*cp.sum_squares(t - np.stack(y).flatten())
    
    # add autotuning if required
    if autotune:
        zeta = cp.Variable()        
        cost += c*(zeta)**2
    else:
        zeta = c**2
    
    
    # define the constraints   
    constr = []    
    for i in range(d-1):
        for j in range(i+1,d):
            
            constr.append(cp.sum_squares(t[i] - t[j]) <= zeta * cp.sum_squares(x[i] - x[j]))
    
    # add constraints for autotuning
    if autotune:
        constr.append(zeta <= 1)
        constr.append(zeta >= 0)
            

    # create and solve problem
    cp_solver = cp_solver if cp_solver is not None else cp.ECOS
    problem = cp.Problem(cp.Minimize(cost), constr)
    problem.solve(cp_solver, **solver_params)
    
    # return results
    if autotune: zeta = math.sqrt(zeta.value)
    else: zeta = c
    
    return [t.value[i*n:(i+1)*n].reshape((-1,1)) for i in range(d)], zeta


#%% PRS SOLVERS

# ------------ PRS solver
def prs_solver(x, y, zeta, tol=1e-2, num_iter=1e3, rho=1, newton_params=None):
    """
    """
    
    if newton_params is None:
        newton_params = {'tol':1e-4, 'num_iter':25, 'b':0.8, 'c':0.5, 'max_iter':10}
    
    # -------- problem information
    d, n = len(x), np.size(x[0]) # num. of data points and size of domain
    
    # list of ordered data points pairs
    E = [(i,j) for i in range(d-1) for j in range(i+1,d)]
    
    
    # -------- generate cost and constraints data
    # cost's quadratic term
    p_0 = 1/(d-1) + 1/rho
    P = np.kron(np.array([[1, -1], [-1, 1]]), np.eye(n))
    
    q_0, r = {}, {}
    for e in E:
        
        q_0[e] = -np.vstack((y[e[0]], y[e[1]])) / (d-1)
        r[e] = -0.5*(zeta**2)*utils.square_norm(x[e[0]] - x[e[1]])
        
    
    # initialize the variables
    xi = {e : np.zeros((2*n,1)) for e in E}
    z = {e : np.zeros((2*n,1)) for e in E}
    z_old = np.zeros((2*n,1)) # memory slot to compute FPR

    # -------- solve the regression
    for k in range(int(num_iter)):
        
        v = {i : np.zeros((n,1)) for i in range(d)}
        
        # perform the local updates and (recursively) project onto the 
        # consensus constraints
        for e in E:
            
            xi[e], _ = qcqp(p_0, q_0[e] - z[e]/rho, P, r[e], **newton_params)

            v[e[0]] += (2*xi[e] - z[e])[:n,] / (d-1)
            v[e[1]] += (2*xi[e] - z[e])[n:,] / (d-1)
                    
        # auxiliary variables update, and compute the fixed point residual 
        # (distance between consecutive iterates)
        fpr = 0 # fixed point residual for stopping condition
        
        for e in E:
            
            # store the previous iterate and update
            z_old = z[e]
            z[e] = z[e] + np.vstack((v[e[0]], v[e[1]])) - xi[e]
            # compute the fixed point residual
            fpr = np.maximum(fpr, utils.square_norm(z[e] - z_old))
        
        # check stopping condition
        if fpr <= tol: break
    
    t = [xi[(i,i+1)][:n,] for i in range(d-1)] 
    t.append(xi[(d-2,d-1)][n:,])
    
    return t

def _P_inverse(p_0, p_1, n):
    
    P_inv = np.zeros((2*n,2*n))
    
    det = p_0*(p_0+2*p_1) # determinant of P_0 + l P_1
    # elements of inverse
    p_inv_1 = (p_0+p_1)/det
    p_inv_2 = p_1/det

    for i in range(n):
        P_inv[i,i] = p_inv_1
        P_inv[i+n,i] = p_inv_2

    P_inv[n:,n:] = P_inv[:n,:n]
    P_inv[:n,n:] = P_inv[n:,:n]
        
    return P_inv

def qcqp(p_0, q_0, P, r,
                l=0, tol=1e-4, num_iter=25, b=0.8, c=0.5, max_iter=10):
    
    # p_0 is the diagonal element of P_0
    
    n = q_0.size//2 # domain dimension
    
    # compute P^-1(l) and P^-1(l) q(l)
    P_inv = _P_inverse(p_0, l, n)
    v_1 = P_inv.dot(q_0)
    
    # compute current cost
    f_l = -0.5*q_0.T.dot(v_1) + l*r
    
    # initialize gradient and Hessian
    g, h = 0, 0


    for k in range(int(num_iter)):
        
        # compute gradient and Hessian
        g = 0.5*v_1.T.dot(P.dot(v_1)) + r
        h = - v_1.T.dot(P.dot(P_inv.dot(P.dot(v_1))))
        
        # store previous iterate and cost
        l_old, f_l_old = l, f_l
        
        # backtracking line search
        dd = g / h # ascent direction
        step = 1 # initial step-size
        
        stop, t = False, 0 # backtracking stopping conditions
        
        while not stop and t < max_iter:
            
            # compute candidate solution with projected Newton
            l = np.maximum(l_old + step*dd, 0)
            
            # compute P^-1(l) and P^-1(l) q(l)
            P_inv = _P_inverse(p_0, l, n)
            v_1 = P_inv.dot(q_0)
            
            # compute current cost
            f_l = -0.5*q_0.T.dot(v_1) + l*r
            
            # check stopping condition
            if f_l < f_l_old + c*step*g*dd:
                step = b*step
                t += 1
            else:
                stop = True
        
        # if backtracking fails, apply projected gradient instead
        if not stop:
            
            # choose the step-size
            step = - 1 / h
            
            # projected gradient step
            l = np.maximum(l_old + step*g, 0)

            # compute P^-1(l) and P^-1(l) q(l)
            P_inv = _P_inverse(p_0, l, n)
            v_1 = P_inv.dot(q_0)
            
            # compute current cost
            f_l = -0.5*q_0.T.dot(v_1) + l*r

        # check stopping condition
        if abs(l-l_old) <= tol: break

    return - v_1, l



def prs_solver_2(x, y, zeta, tol=1e-2, num_iter=1e3, rho=1, newton_params=None):
    """
    """
    
    if newton_params is None:
        newton_params = {'tol':1e-4, 'num_iter':25, 'b':0.8, 'c':0.5, 'max_iter':10}
    
    # -------- problem information
    d, n = len(x), np.size(x[0]) # num. of data points and size of domain
    
    # list of ordered data points pairs
    E = [(i,j) for i in range(d-1) for j in range(i+1,d)]
    
    
    # -------- generate cost and constraints data
    # cost's quadratic term
    p_0 = 1/(d-1) + 1/rho
    P = np.kron(np.array([[1, -1], [-1, 1]]), np.eye(n))
    
    q_0, r = {}, {}
    for e in E:
        
        q_0[e] = -np.vstack((y[e[0]], y[e[1]])) / (d-1)
        r[e] = -0.5*(zeta**2)*utils.square_norm(x[e[0]] - x[e[1]])
        
    
    # initialize the variables
    xi = {e : np.zeros((2*n,1)) for e in E}
    z = {e : np.zeros((2*n,1)) for e in E}
    z_old = np.zeros((2*n,1)) # memory slot to compute FPR
    

    # -------- solve the regression
    for k in range(int(num_iter)):
        
        v = {i : np.zeros((n,1)) for i in range(d)}
        
        # perform the local updates and (recursively) project onto the 
        # consensus constraints
        for e in E:
            
            xi[e], _ = qcqp_2(p_0, q_0[e] - z[e]/rho, P, r[e], **newton_params)

            v[e[0]] += (2*xi[e] - z[e])[:n,] / (d-1)
            v[e[1]] += (2*xi[e] - z[e])[n:,] / (d-1)
                    
        # auxiliary variables update, and compute the fixed point residual 
        # (distance between consecutive iterates)
        fpr = 0 # fixed point residual for stopping condition
        
        for e in E:
            
            # store the previous iterate and update
            z_old = z[e]
            z[e] = z[e] + np.vstack((v[e[0]], v[e[1]])) - xi[e]
            # compute the fixed point residual
            fpr = np.maximum(fpr, utils.square_norm(z[e] - z_old))
        
        # check stopping condition
        if fpr <= tol: break
    
    t = [xi[(i,i+1)][:n,] for i in range(d-1)] 
    t.append(xi[(d-2,d-1)][n:,])
    
    return t


def qcqp_2(p_0, q_0, P, r,
                l=-1, tol=1e-4, num_iter=25, b=0.8, c=0.5, max_iter=10):
    
    # p_0 is the diagonal element of P_0
    
    n = q_0.size//2 # domain dimension
    
    # compute P^-1(l) and P^-1(l) q(l)
    P_inv = _P_inverse(p_0, l, n)
    v_1 = P_inv.dot(q_0)
    
    # compute current cost
    f_l = -0.5*q_0.T.dot(v_1) + l*r
    
    # initialize gradient and Hessian
    g, h = 0, 0


    for k in range(int(num_iter)):
        
        # compute gradient and Hessian
        g = 0.5*v_1.T.dot(P.dot(v_1)) + r
        h = - v_1.T.dot(P.dot(P_inv.dot(P.dot(v_1))))
        
        # store previous iterate and cost
        l_old, f_l_old = l, f_l
        
        # backtracking line search
        dd = g / h # ascent direction
        step = 1 # initial step-size
        
        stop, t = False, 0 # backtracking stopping conditions
        
        while not stop and t < max_iter:
            
            # compute candidate solution with projected Newton
            l = np.maximum(l_old + step*dd, 0)
            
            # compute P^-1(l) and P^-1(l) q(l)
            P_inv = _P_inverse(p_0, l, n)
            v_1 = P_inv.dot(q_0)
            
            # compute current cost
            f_l = -0.5*q_0.T.dot(v_1) + l*r
            
            # check stopping condition
            if f_l < f_l_old + c*step*g*dd:
                step = b*step
                t += 1
            else:
                stop = True
        
        # if backtracking fails, apply projected gradient instead
        if not stop:
            
            # choose the step-size
            step = - 1 / h
            
            # projected gradient step
            l = np.maximum(l_old + step*g, 0)

            # compute P^-1(l) and P^-1(l) q(l)
            P_inv = _P_inverse(p_0, l, n)
            v_1 = P_inv.dot(q_0)
            
            # compute current cost
            f_l = -0.5*q_0.T.dot(v_1) + l*r

        # check stopping condition
        if abs(l-l_old) <= tol: break

    return - v_1, l


# ------------ PRS solver w/ autotuning
def prs_solver_autotuning(x, y, weight, tol=1e-2, num_iter=1e3, rho=1, eps=1e-10, newton_params=None):
    """
    """
    
    if newton_params is None:
        newton_params = {'tol':1e-4, 'num_iter':25, 'b':0.8, 'c':0.5, 'max_iter':10}
    
    # -------- problem information
    d, n = len(x), np.size(x[0]) # num. of data points and size of domain
    
    # list of ordered data points pairs
    E = [(i,j) for i in range(d-1) for j in range(i+1,d)]
    
    
    # -------- generate cost and constraints data
    # cost's quadratic term
    p_0_1, p_0_2 = 1/(d-1) + 1/rho, 2*weight/(d*(d-1)) + 1/rho
    P = np.zeros((2*n+1, 2*n+1))
    P[:-1,:-1] = np.kron(np.array([[1, -1], [-1, 1]]), np.eye(n))
    
    q_0, q = {}, {}
    for e in E:
        
        q_0[e] = -np.vstack((y[e[0]], y[e[1]], 0)) / (d-1)
        q[e] = np.vstack((np.zeros((2*n,1)), -0.5*utils.square_norm(x[e[0]] - x[e[1]])))
        
    
    # initialize the variables
    xi = {e : np.zeros((2*n+1,1)) for e in E}
    z = {e : np.zeros((2*n+1,1)) for e in E}
    z_old = np.zeros((2*n+1,1)) # memory slot to compute FPR

    # -------- solve the regression
    for k in range(int(num_iter)):
        
        v = {i : np.zeros((n,1)) for i in range(d)}
        w = 0
        
        # perform the local updates and (recursively) project onto the 
        # consensus constraints
        for e in E:
            
            xi[e], _ = qcqp_autotuning(p_0_1, p_0_2, q_0[e] - z[e]/rho, P, q[e], **newton_params)

            v[e[0]] += (2*xi[e] - z[e])[:n,] / (d-1)
            v[e[1]] += (2*xi[e] - z[e])[n:2*n,] / (d-1)
            
            # project onto (0,1) and then average the contraction constants
            w += np.minimum(np.maximum((2*xi[e] - z[e])[-1] / (d-1), eps), 1-eps) / (d*(d-1))
                    
        # auxiliary variables update, and compute the fixed point residual 
        # (distance between consecutive iterates)
        fpr = 0 # fixed point residual for stopping condition
        
        for e in E:
            
            # store the previous iterate and update
            z_old = z[e]
            z[e] = z[e] + np.vstack((v[e[0]], v[e[1]], w)) - xi[e]
            # compute the fixed point residual
            fpr = np.maximum(fpr, utils.square_norm(z[e] - z_old))
        
        # check stopping condition
        if fpr <= tol: break
    
    t = [xi[(i,i+1)][:n,] for i in range(d-1)] 
    t.append(xi[(d-2,d-1)][n:2*n,])
    
    return t, xi[(0,1)][-1]

def _P_inverse_autotuning(p_0_1, p_0_2, p_1, n):
    
    P_inv = np.zeros((2*n+1,2*n+1))
    
    det = p_0_1*(p_0_1+2*p_1) # determinant of P_0 + l P_1
    # elements of inverse
    p_inv_1 = (p_0_1+p_1)/det
    p_inv_2 = p_1/det

    for i in range(n):
        P_inv[i,i] = p_inv_1
        P_inv[i+n,i] = p_inv_2

    P_inv[n:2*n,n:2*n] = P_inv[:n,:n]
    P_inv[:n,n:2*n] = P_inv[n:2*n,:n]
    
    P_inv[-1,-1] = 1/p_0_2
        
    return P_inv

def qcqp_autotuning(p_0_1, p_0_2, q_0, P, q,
                l=0, tol=1e-4, num_iter=25, b=0.8, c=0.5, max_iter=10):
    
    # p_0 is the diagonal element of P_0
    
    n = q_0.size//2 # domain dimension
    
    # compute q(l), P^-1(l) and P^-1(l) q(l)
    q_l = q_0 + l*q
    P_inv = _P_inverse_autotuning(p_0_1, p_0_2, l, n)
    v_1 = P_inv.dot(q_l)
    
    # compute current cost
    f_l = -0.5*q_l.T.dot(v_1)
    
    # initialize gradient and Hessian
    g, h = 0, 0


    for k in range(int(num_iter)):
        
        v_2 = P.dot(v_1)
        v_3 = P_inv.dot(q)
        
        # compute gradient and Hessian
        g = - q.T.dot(v_1) + 0.5*v_1.T.dot(v_2)
        h = - q.T.dot(v_3) + 2*v_3.T.dot(v_2) \
            - v_1.T.dot(P.dot(P_inv.dot(v_2)))
        
        # store previous iterate and cost
        l_old, f_l_old = l, f_l
        
        # backtracking line search
        dd = g / h # ascent direction
        step = 1 # initial step-size
        
        stop, t = False, 0 # backtracking stopping conditions
        
        while not stop and t < max_iter:
            
            # compute candidate solution with projected Newton
            l = np.maximum(l_old + step*dd, 0)
            
            # compute q(l), P^-1(l) and P^-1(l) q(l)
            q_l = q_0 + l*q
            P_inv = _P_inverse_autotuning(p_0_1, p_0_2, l, n)
            v_1 = P_inv.dot(q_l)
            
            # compute current cost
            f_l = -0.5*q_l.T.dot(v_1)
            
            # check stopping condition
            if f_l < f_l_old + c*step*g*dd:
                step = b*step
                t += 1
            else:
                stop = True
        
        # if backtracking fails, apply projected gradient instead
        if not stop:
            
            # choose the step-size
            step = - 1 / h
            
            # projected gradient step
            l = np.maximum(l_old + step*g, 0)

            # compute q(l), P^-1(l) and P^-1(l) q(l)
            q_l = q_0 + l*q
            P_inv = _P_inverse_autotuning(p_0_1, p_0_2, l, n)
            v_1 = P_inv.dot(q_l)
            
            # compute current cost
            f_l = -0.5*q_l.T.dot(v_1)

        # check stopping condition
        if abs(l-l_old) <= tol: break

    return - v_1, l


#%% UTILITY FUNCTIONS

def _fill_diagonal(A, a, n):
    
    for i in range(n): A[i,i] = a

def _inverse(m):
    """
    Inverse of a 2-by-2 matrix.

    """    
    
    return np.array([[m[1,1], -m[1,0]], [-m[0,1], m[1,1]]]) \
           / (m[0,0]*m[1,1] - m[0,1]*m[1,0])