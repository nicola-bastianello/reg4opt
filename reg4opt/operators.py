#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Operator template and examples.
"""


import numpy as np
from numpy import linalg as la


#%% FUNCTION TEMPLATE

class Operator():
    r"""
    Template class for operators.
    
    This class serves as a template for operators:
    
        .. math:: \mathcal{T} : \mathbb{D}_1 \to \mathbb{D}_2
    
    where :math:`\mathbb{D}_i \subset \mathbb{R}^{n_{i,1} \times n_{i,2} \times \cdots}`
    for some given dimensions :math:`n_{i,1}, n_{i,2}, \ldots`, :math:`i = 1, 2`.
    
    
    TODO: operations: sum of operators, product by scalar, composition, averaging
    """
    
    def __init__(self, dom, rng=None, time=None):
        """
        Class constructor.

        Parameters
        ----------
        dom : tvopt.sets.Set
            The operator's domain.
        rng : tvopt.sets.Set, optional
            The operator's range, which coincides with the domain if None.
        time : tvopt.sets.T, optional
            The operator's time domain, if None the operator is static.
        """
        
        self.dom, self.time = dom, time
        self.is_dynamic = time is not None
        self.rng = rng if rng is not None else self.dom
        
    def operator(self, x, *args, **kwargs):
        """
        An evaluation of the operator.

        Parameters
        ----------
        x : array_like
            The x where the operator should be evaluated.
        *args
            The time at which the operator should be evaluated. Not required if
            the operator is static.
        **kwargs
            Any other required argument.
        """
        
        raise NotImplementedError()
    
    def fpr(self, x, *args, **kwargs):
        """
        Evaluate the fixed point residual (FPR).

        Parameters
        ----------
        x : array_like
            The point whose FPR should be evaluated.
        *args
            The time at which the operator should be evaluated. Not required if
            the operator is static.

        Returns
        -------
        float
            The FPR at `x`.
        """
        
        return la.norm(x - self.operator(x, *args, **kwargs))
    
    def compose(self, other):
        """
        TODO

        Parameters
        ----------
        other : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        
        return CompositeOperator(self, other)
    
    def average(self, a):
        """
        TODO

        Parameters
        ----------
        a : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        
        return AveragedOperator(self, a)
    
    def sample(self, t):
        
        if self.is_dynamic:
            return SampledOperator(self, t)
        else:
            return self
    
    def __add__(self, other):
        
        return SumOperator(self, other)
    
    def __mul__(self, c):
        
        if np.isscalar(c):
            return ScaledOperator(self, c)
        else:
            raise TypeError("Can't multiply Operator by {}.".format(type(c)))
    
class SumOperator(Operator):
    
    def __init__(self, o1, o2):
        
        # handle time-varying operators
        times = [o.time for o in (o1, o2) if o.is_dynamic]
        time = None if len(times)==0 else times[0]
        
        super().__init__(o1.dom, rng=o1.rng, time=time)
        self.o1, self.o2 = o1, o2
    
    def operator(self, x, *args, **kwargs):
        
        # first operator
        y = self.o1.operator(x, *args, **kwargs) if self.o1.is_dynamic \
            else self.o1.operator(x, **kwargs)
        # second operator
        return y + self.o2.operator(x, *args, **kwargs) if self.o2.is_dynamic \
               else self.o2.operator(x, **kwargs)

class ScaledOperator(Operator):
    
    def __init__(self, o, c):
        
        super().__init__(o.dom, rng=o.rng, time=o.time)
        self.o, self.c = o, c
    
    def operator(self, x, *args, **kwargs):
        
        return self.c*self.o.operator(x, *args, **kwargs) if self.o.is_dynamic \
               else self.c*self.o.operator(x, **kwargs)

class CompositeOperator(Operator):
    
    def __init__(self, o2, o1):
        # o2 is the outermost operator, o1 the innermost
        
        # handle time-varying operators
        times = [o.time for o in (o1, o2) if o.is_dynamic]
        time = None if len(times)==0 else times[0]
        
        super().__init__(o1.dom, rng=o2.rng, time=time)
        self.o1, self.o2 = o1, o2
    
    def operator(self, x, *args, **kwargs):
        
        # first operator
        y = self.o1.operator(x, *args, **kwargs) if self.o1.is_dynamic \
            else self.o1.operator(x, **kwargs)
        # second operator
        return self.o2.operator(y, *args, **kwargs) if self.o2.is_dynamic \
               else self.o2.operator(y, **kwargs)

class AveragedOperator(Operator):
    
    def __init__(self, o, a):
        
        super().__init__(o.dom, rng=o.rng, time=o.time)
        self.o, self.a = o, a
    
    def operator(self, x, *args, **kwargs):
        
        # evaluate operator
        y = self.o.operator(x, *args, **kwargs) if self.ox.is_dynamic \
               else self.o.operator(x, **kwargs)
        
        return (1-self.a)*x + self.a*y

class SampledOperator(Operator):
                
    def __init__(self, o, t):

        super().__init__(o.dom, rng=o.rng)
        self.o, self.t = o, t
    
    def operator(self, x, **kwargs):
        
        return self.o.operator(x, self.t, **kwargs)


#%% OPERATORS EXAMPLES

class Identity(Operator):
    
    def operator(self, x, *args, **kwargs):
        
        return x

class Gradient(Operator):
    
    def __init__(self, f, s):
        """
        Gradient step operator (or forward operator).

        Parameters
        ----------
        f : pyregression.Function
            The cost function.
        s : float
            The step-size.
        """
        
        time = f.time if f.is_dynamic else None
        super().__init__(f.dom, time=time)
        self.f, self.s = f, s
    
    def operator(self, x, *args):
                
        return x - self.s*self.f.gradient(x, *args) if self.is_dynamic \
               else x - self.s*self.f.gradient(x)

class Proximal(Operator):
    
    def __init__(self, f, p):
        
        time = f.time if f.is_dynamic else None
        super().__init__(f.dom, time=time)
        self.f, self.p = f, p
    
    def operator(self, x, *args):
        
        return self.f.proximal(x, *args, penalty=self.p) if self.is_dynamic \
               else self.f.proximal(x, penalty=self.p)

class Reflective(Operator):
    
    def __new__(cls, f, p):
        
        return 2*Proximal(f, p) - Identity(f.dom)
    
class ForwardBackward(Operator):
    
    def __new__(cls, f, g, p):
        
        return CompositeOperator(Proximal(g, p), Gradient(f, p))
    
class PeacemanRachford(Operator):
    
    def __new__(cls, f, g, p):
        
        return CompositeOperator(Reflective(g, p), Reflective(f, p))