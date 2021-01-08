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
    
    `Operator` objects support the following operations:
        
        - composition,
        - averaging (a.k.a. relaxation),
        - addition,
        - product by a scalar.
    
    An `Operator` should expose the `operator` method to perform evaluations
    of :math:`\mathcal{T}`. The class then implements the `fpr` which computes
    the fixed point residual :math:`\| \mathcal{T} x - x\|` at a given point
    :math:`x`.
    
    The `Operator` can also be time-varying, in which case the function 
    `sample` is also available.
    
    Attributes
    ----------
    dom : tvopt.sets.Set
        The operator's domain.
    rng : tvopt.sets.Set
        The operator's range.
    time : tvopt.sets.T
        The time domain :math:`\mathbb{R}_+`. If the cost is static this is None.
    is_dynamic : bool
        Attribute to check if the cost is static or dynamic.
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
        
        This method returns the FPR
        
        .. math:: \| \mathcal{T} x - x \|.

        Parameters
        ----------
        x : array_like
            The point whose FPR should be evaluated.
        *args
            The time at which the operator should be evaluated. Not required if
            the operator is static.
        **kwargs
            Any other required argument.

        Returns
        -------
        float
            The FPR at `x`.
        """
        
        return la.norm(x - self.operator(x, *args, **kwargs))
    
    def compose(self, other):
        """
        Compose the operator with a second one.
        
        The method composes the operator with a second one, provided that the
        range of the former coincides with the domain of the latter.

        Parameters
        ----------
        other : Operator
            The second operator.

        Returns
        -------
        Operator
            An operator object defining the composition.
        
        See Also
        --------
        CompositeOperator : The composite operator object.
        """
        
        return CompositeOperator(self, other)
    
    def average(self, a):
        """
        Averaging (a.k.a. relaxation) of the operator.
        
        The method returns an averaged version of the operator, defined as
        
        .. math:: (1 - a) \mathcal{I} + a \mathcal{T}
        
        where :math:`a \in (0,1]` and :math:`\mathcal{I}` is the identity
        operator.

        Parameters
        ----------
        a : float
            The relaxation constant.

        Returns
        -------
        Operator
            DESCRIPTION.
        
        See Also
        --------
        AveragedOperator : The averaged operator object.
        """
        
        return AveragedOperator(self, a)
    
    def sample(self, t):
        """
        Sample the operator.
        
        This method returns a `SampledOperator` object which consists of the
        operator as evaluated at time `t`.
        
        If the operator is static, the operator itself is returned.

        Parameters
        ----------
        t : float
            The time at which the cost should be sampled.

        Returns
        -------
        Operator
            The sampled operator or, if static, the operator itself.
        
        See Also
        --------
        SampledOperator : The sampled operator object.
        """
        
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
    

# -------- AUXILIARY CLASSES

class SumOperator(Operator):
    r"""
    Sum of operators.
    
    This class defines an operator as the sum of two given operators. That is,
    given the operators :math:`\mathcal{T}` and :math:`\mathcal{R}`, 
    the class defines :math:`\mathcal{T} + \mathcal{R}`. The domains and ranges
    of the two operators must be compatible.
    """
    
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
    r"""
    Scaled operator.
    
    This class defines an operator multiplied by a scalar. That is, given the 
    operator :math:`\mathcal{T}` and :math:`c \in \mathbb{R}`, the class defines
    :math:`c \mathcal{T}`.
    """
    
    def __init__(self, o, c):
        
        super().__init__(o.dom, rng=o.rng, time=o.time)
        self.o, self.c = o, c
    
    def operator(self, x, *args, **kwargs):
        
        return self.c*self.o.operator(x, *args, **kwargs) if self.o.is_dynamic \
               else self.c*self.o.operator(x, **kwargs)

class CompositeOperator(Operator):
    r"""
    Composition of operators.
    
    This class defines an operator as the composition of two given operators.
    That is, given the operators :math:`\mathcal{T}` and :math:`\mathcal{R}`, 
    the class defines :math:`\mathcal{T} \circ \mathcal{R}`. The domains and 
    ranges of the two operators must be compatible.
    """
    
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
    r"""
    Averaged operator.
    
    This class defines an operator as the averaging (or relaxation) of a given
    operator. That is, given the operator :math:`\mathcal{T}`, and 
    :math:`a \in (0,1]`, the class defines 
    :math:`(1 - a) \mathcal{I} + a \mathcal{T}`.
    """
    
    def __init__(self, o, a):
        
        super().__init__(o.dom, rng=o.rng, time=o.time)
        self.o, self.a = o, a
    
    def operator(self, x, *args, **kwargs):
        
        # evaluate operator
        y = self.o.operator(x, *args, **kwargs) if self.ox.is_dynamic \
               else self.o.operator(x, **kwargs)
        
        return (1-self.a)*x + self.a*y

class SampledOperator(Operator):
    r"""
    Sampled operator.
    
    This class defines a static operator by sampling a given dynamic operator,
    that is, by fixing the time argument to a given value.
    """
                
    def __init__(self, o, t):

        super().__init__(o.dom, rng=o.rng)
        self.o, self.t = o, t
    
    def operator(self, x, **kwargs):
        
        return self.o.operator(x, self.t, **kwargs)


#%% OPERATORS EXAMPLES

class Identity(Operator):
    r"""
    Identity operator.
    """
    
    def operator(self, x, *args, **kwargs):
        
        return x

class Gradient(Operator):
    r"""
    Gradient operator.
    
    This class defines the gradient step operator (or forward operator) of a 
    given cost function :math:`f`.
    """
    
    def __init__(self, f, s):
        """
        Gradient step operator (or forward operator).

        Parameters
        ----------
        f : tvopt.costs.Cost
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
    r"""
    Proximal operator.
    
    This class defines the proximal operator (or backward operator) of a 
    given cost function :math:`f`. That is, the operator:
    
    .. math:: \operatorname{prox}_{\rho f}(x) = \operatorname{arg\,min}_{y} \left\{ f(y) + \frac{1}{2 \rho} \|y - x\|^2 \right\}.
    """
    
    def __init__(self, f, p):
        """
        Proximal operator (or backwrd operator).

        Parameters
        ----------
        f : tvopt.costs.Cost
            The cost function.
        p : float
            The penalty parameter.
        """
        
        time = f.time if f.is_dynamic else None
        super().__init__(f.dom, time=time)
        self.f, self.p = f, p
    
    def operator(self, x, *args):
        
        return self.f.proximal(x, *args, penalty=self.p) if self.is_dynamic \
               else self.f.proximal(x, penalty=self.p)

class Reflective(Operator):
    r"""
    Reflective operator.
    
    This class defines the reflective operator of a given cost function 
    :math:`f`. That is, the operator:
    
    .. math:: \operatorname{refl}_{\rho f}(x) = 2 \operatorname{prox}_{\rho f}(x) - x.
    """
    
    def __new__(cls, f, p):
        
        return 2*Proximal(f, p) - Identity(f.dom)
    
class ForwardBackward(Operator):
    r"""
    Forward-backward operator.
    
    This class defines the forward-backward operator of two given cost
    functions :math:`f` and :math:`g`. That is, the operator:
    
    .. math:: \operatorname{prox}_{\rho g}\left( x - \rho \nabla f(x) \right).
    """
    
    def __new__(cls, f, g, p):
        
        return CompositeOperator(Proximal(g, p), Gradient(f, p))
    
class PeacemanRachford(Operator):
    r"""
    Peaceman-Rachford operator.
    
    This class defines the Peaceman-Rachford operator of two given cost
    functions :math:`f` and :math:`g`. That is, the operator:
    
    .. math:: \operatorname{refl}_{\rho g} \left( \operatorname{refl}_{\rho f}(x) \right).
    """
    
    def __new__(cls, f, g, p):
        
        return CompositeOperator(Reflective(g, p), Reflective(f, p))