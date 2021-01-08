#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utility functions.
"""

import sys
import math
import numpy as np
from numpy import linalg as la

from numpy.random import default_rng
ran = default_rng()

from tvopt import sets


#%% DATA GENERATION FOR OPERATOR REGRESSION

def generate_data(T, x, num_data, method="normal", **kwargs):
    """
    Generate the training data for an operator regression problem.
    
    The function encapsulates different method of generating the training data
    for an operator regression problem. The choices are
    
    * normal: the data are chosen as :math:`x + d_i`, where
      :math:`d_i` are random vectors with normal distribution;
    * fireworks: the data are chosen as :math:`x + d_i`, where we have
      :math:`d_i = a e_j`, with :math:`a` a r.v. with normal distribution and
      :math:`e_j` a randomly selected vector of the standard basis;
    * uniform: the data are chosen as :math:`x + d_i`, where
      :math:`d_i` are random vectors with uniform distribution.

    Parameters
    ----------
    T : operators.Operator
        The operator source of the training data.
    x : ndarray
        The center of the training data, that is, all other training points are
        chosen as perturbations of `x`.
    num_data : int
        The number of training points, including `x`, which means that 
        `num_data`-1 points are randomly selected.
    method : str, optional
        The method to choose the training points, defaults to "normal".
    **kwargs : tuple
        The arguments of `method`.

    Returns
    -------
    list
        The points :math:`x_i` selected by the method.
    list
        The operator `T` evaluated in the chosen points.
    
    See Also
    --------
    generate_data_n : Data generation with normal distribution.
    generate_data_fw : Data generation with fireworks method.
    generate_data_u : Data generation with unform distribution.
    """
    
    if method == "normal" or method == "n":
        return generate_data_n(T, x, num_data, **kwargs)
    elif method == "fireworks" or method == "fw" or method == "f":
        return generate_data_fw(T, x, num_data, **kwargs)
    elif method == "uniform" or method == "u":
        return generate_data_fw(T, x, num_data, **kwargs)
    else:
        return generate_data_n(T, x, num_data, **kwargs)

def generate_data_n(T, x, num_data, var):
    """
    Data generation using perturbation that are normally distributed with
    variance `var`. See `generate_data` for details.
    """

    x_i = [x] + [x + math.sqrt(var)*ran.standard_normal(x.shape) for _ in range(num_data-1)]
    
    return x_i, [T.operator(z) for z in x_i]

def generate_data_fw(T, x, num_data, var):
    """
    Data generation using perturbation chosen according to the fireworks
    method. See `generate_data` for details.
    """
    
    # pick directions and magnitudes
    d = ran.choice(range(x.size), size=num_data-1, replace=False)
    m = math.sqrt(var)*ran.standard_normal(num_data-1)
    
    # create the points
    x_i = [x] + [x + m[i]*_standard_basis(d[i], x.size) for i in range(num_data-1)]
    
    return x_i, [T.operator(z) for z in x_i]

def generate_data_u(T, x, num_data, a):
    """
    Data generation using perturbation that are uniformly distributed with
    range -`a` to `a`. See `generate_data` for details.
    """
    
    x_i = [x] + [x + 2*a*ran.random(x.shape)-a for _ in range(num_data-1)]
    
    return x_i, [T.operator(z) for z in x_i]


#%% LINEAR ALGEBRA

def square_norm(x):
    """
    Compute the square norm of the given vector.

    Parameters
    ----------
    x : array_like
        The vector array.

    Returns
    -------
    ndarray
        The square norm.
    
    Notes
    -----
    The function reshapes `x` to a column vector, so it does not correctly
    handle n-dimensional arrays. For n-dim arrays use `numpy.linalg.norm`.
    """
    
    x = np.reshape(x, (-1,1))
    
    return x.T.dot(x)


def norm(x):
    """
    Compute the norm of the given vector.

    Parameters
    ----------
    x : array_like
        The vector array.

    Returns
    -------
    ndarray
        The square norm.
    
    See Also
    --------
    square_norm : Square norm
    
    Notes
    -----
    The function reshapes `x` to a column vector, so it does not correctly
    handle n-dimensional arrays. For n-dim arrays use `numpy.linalg.norm`.
    """
    
    return math.sqrt(square_norm(x))

def norm_1(x):
    r"""
    Compute the :math:`\ell_1` norm of the given vector.

    Parameters
    ----------
    x : array_like
        The vector array.

    Returns
    -------
    ndarray
        The :math:`\ell_1` norm.
    
    See Also
    --------
    norm : Euclidean norm
    
    Notes
    -----
    The function treats `x` as a one-dimensional vector (row or column). For 
    n-dim arrays use `numpy.linalg.norm`.
    """

    return np.sum(np.abs(x))

def _standard_basis(i, n):
    r"""
    Return the `i`-th vector of the standard basis in :math:`\mathbb{R}^n`.
    """
    
    x = np.zeros((n,1))
    x[i] = 1
    
    return x


#%% VARIOUS

def print_progress(i, num_iter, bar_length=80, decimals=2):
    """
    Print the progresso to command line.
    
    Parameters
    ----------
    i : int
        Current iteration.
    num_iter : int
        Total number of iterations.
    bar_length : int, optional
        Length of progress bar.
    decimals : int, optional
        Decimal places of the progress percent.
    
    Notes
    -----
    Adapted from `here 
    <https://gist.github.com/aubricus/f91fb55dc6ba5557fbab06119420dd6a>`_.
    """
    
    # completion percent
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (i / num_iter))
    
    # progress bar
    filled_length = int(round(bar_length * i / num_iter))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    
    # print progress
    sys.stdout.write('\rCompleted |%s| %s%s' % (bar, percents, '%')),
    if i == num_iter:
        sys.stdout.write('\n')
    sys.stdout.flush()

def random_intersecting_balls(n, d, max_radius=100):
    r"""
    Randomly generate a set of intersecting ball sets.
    
    This utility function generates a set of `d` balls in :math:`\mathbb{R}^n`
    for the purpose of testing projection methods. The centers of the balls are
    uniformly chosen points in the unit sphere, and the radius is randomly 
    picked between 0 (excluded) and `max_radius`.
    
    As a consequence, the origin of :math:`\mathbb{R}^n` is (one of the) points
    in the intersection of the balls.

    Parameters
    ----------
    n : int
        The dimension of the space.
    d : int
        The number of balls to be generated.
    max_radius : float, optional
        The maximum radius of the generated balls. The default is 100.

    Returns
    -------
    s : list
        A list of tvopt.sets.Ball objects representing the sets.

    Notes
    -----
    Adapted from `here 
    <https://stackoverflow.com/a/54544972>`_.
    """
    
    s = []
    
    for i in range(d):
        
        # generate radius
        r = max_radius*ran.random()
        
        # generate center
        c = ran.standard_normal((n, 1)) # random direction
        c = c / la.norm(c) # normalize the random direction
        a = ran.random()**(1/n) # random radius
        
        # create set
        s.append(sets.Ball(a*r*c, r))
    
    return s