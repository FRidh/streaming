"""
Operators
=========

This module defines the operators that are used in streams.
Multiple dispatch is used.
"""

from functools import partial
from multipledispatch import dispatch, Dispatcher
import numpy as np
import operator
from multipledispatch.conflict import AmbiguityWarning
import warnings
warnings.filterwarnings("ignore", category=AmbiguityWarning)

## Add basic operators to module
import sys
_thismodule = sys.modules[__name__]

# Our own multiple dispatch function that includes a namespace
dispatch = partial(dispatch)#, namespace='streaming')



#_basic_objects = (object, float, int)


_BINARY_OPERATORS_WITH_REVERSE = [
    'add',
    'truediv',
    'floordiv',
    'pow',
    'mod',
    'mul',
    'matmul',
    'sub',
    ]

_BINARY_OPERATORS_WITHOUT_REVERSE = [
    'lt',
    'le',
    'eq',
    'ne',
    'ge',
    'gt',
    ]

#_BINARY_LOGICAL_OPERATORS = [
    #'and',
    #'or',
    #'xor',
    #]

_BINARY_OPERATORS = _BINARY_OPERATORS_WITH_REVERSE + _BINARY_OPERATORS_WITHOUT_REVERSE #+ _BINARY_LOGICAL_OPERATORS

import functools

# Binary operators using multiple dispatch
for op in _BINARY_OPERATORS:
    # Create a dispatcher for each operator
    D = Dispatcher(op)
    # And store the dispatcher on this module
    setattr(_thismodule, op, D)
    # Furthermore, we like to add (object, object) operations
    D.add((object, object), getattr(operator, op))


# Logical AND

@dispatch(object, object)
def logical_and(a, b):
    return a and b

@dispatch(object, np.ndarray)
def logical_and(a, b):
    return np.logical_and(a, b)

@dispatch(np.ndarray, object)
def logical_and(a, b):
    return np.logical_and(a, b)

# Logical OR

@dispatch(object, object)
def logical_or(a, b):
    return a and b

@dispatch(object, np.ndarray)
def logical_or(a, b):
    return np.logical_or(a, b)

@dispatch(np.ndarray, object)
def logical_or(a, b):
    return np.logical_or(a, b)

# Logical XOR

@dispatch(object, object)
def logical_xor(a, b):
    return a and b

@dispatch(object, np.ndarray)
def logical_xor(a, b):
    return np.logical_xor(a, b)

@dispatch(np.ndarray, object)
def logical_xor(a, b):
    return np.logical_xor(a, b)
