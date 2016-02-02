import cytoolz
from functools import singledispatch

@singledispatch
def peek(seq):
    return cytoolz.peek(seq)


__all__ = ['peek']