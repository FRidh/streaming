import itertools
from functools import singledispatch
import numpy as np

# Ambigious
#@singledispatch
#def chain(iterable):
    #return itertools.chain.from_iterable(iterable)

@singledispatch
def cycle(iterable):
    return itertools.cycle(iterable)

@singledispatch
def tee(iterable, n=2):
    return itertools.tee(iterable, n)

@singledispatch
def toarray(iterable):
    return np.array(list(iterable))

@singledispatch
def repeat_item(iterable, n):
    """Repeat items in `iterable` `n` time."""
    yield from itertools.chain.from_iterable(map(lambda i: itertools.repeat(i, n), iterable))


__all__ = ['cycle', 'tee', 'repeat_item', 'toarray']
