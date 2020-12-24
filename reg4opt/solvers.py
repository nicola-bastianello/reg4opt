#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

from numpy. random import default_rng
ran = default_rng()

from reg4opt.regression import operator_regression, operator_regression_prs
from reg4opt.interpolation import interpolator

#TODO
# - make num_data, var, num_iter kwarguments
# - allow to choose the type of solver (cvxpy or prs), and if there is autotuning (opreg)


#%% OPERATOR REGRESSION-BASED SOLVERS

def opreg_solver(T, x, zeta, num_data, var, num_iter, **solver_args):
        
    for _ in range(num_iter):
        
        # training data
        x_i = [x] + [x + math.sqrt(var)*ran.standard_normal(x.shape) for _ in range(num_data-1)]
        y_i = [T.operator(z) for z in x_i]
        
        # solve OpReg
        t_i, _ = operator_regression(x_i, y_i, zeta, **solver_args)
        
        # apply solution
        x = t_i[0]
    
    return x

def interp_opreg_solver(T, x, zeta, num_data, var, num_iter):
    
    # ------ first step: solve OpReg
    # training data
    x_i = [x] + [x + math.sqrt(var)*ran.standard_normal(x.shape) for _ in range(num_data-1)]
    y_i = [T.operator(z) for z in x_i]
    
    # solve OpReg
    t_i, _ = operator_regression(x_i, y_i, zeta)
    
    # apply solution
    x = t_i[0]
    
    # ------ interpolation step        
    for _ in range(num_iter-1):
        
        x = interpolator(x, x_i, t_i, zeta)
    
    return x

    
    
#TODO function that allows to choose solver (cvxpy or PRS)
# def opreg_solver(T, x, zeta=0.5, num_data=5, var_data=1, num_iter=5, solver="c", **solver_params):
    
#     solver = solver.strip().lower()
    
#     for l in range(num_iter):
        
#         # training data
#         x_i = [x] + [x + math.sqrt(var)*ran.standard_normal((n, 1)) for _ in range(num_data-1)]
#         y_i = [T.operator(z) for z in x_i]
        
#         # solve OpReg
#         if solver == "cvxpy" or solver == "c":
            
#             t_i, _ = operator_regression(x_i, y_i, zeta)
        
#         elif solver == "prs" or solver == "p":
            
#             t_i, _ = operator_regression_prs(x_i, y_i, **solver_params)
        
        
#         # apply solution
        
#     return x




def opreg_solver_prs(T, x, zeta, num_data, var, num_iter, **solver_args):
        
    for _ in range(num_iter):
        
        # training data
        x_i = [x] + [x + math.sqrt(var)*ran.standard_normal(x.shape) for _ in range(num_data-1)]
        y_i = [T.operator(z) for z in x_i]
        
        # solve OpReg
        t_i = operator_regression_prs(x_i, y_i, zeta, **solver_args)
        
        # apply solution
        x = t_i[0]
    
    return x