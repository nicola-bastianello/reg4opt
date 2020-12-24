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

#%% PROJECTIONS

def proj_nonnegative(x):
    """
    Projection onto the non-negative orthant.

    Parameters
    ----------
    x : array_like
        The vector to be projected.

    Returns
    -------
    ndarray
        The projection of `x` onto the non-negative orthant.
    """
    
    return np.maximum(x, np.zeros(x.shape))


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
    # this function generates d random balls in an n-dim space
    # the radius is randomly picked from (0, max_radius)
    # the center is randomly chosen taking a point in the unit sphere
    # a list of sets.Ball objects is returned
    
    # the balls are guaranteed to intersect in the origin at least
    # https://stackoverflow.com/a/54544972
    
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