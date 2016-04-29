import itertools
from functools import singledispatch
import numpy as np

# Ambigious
#@singledispatch
#def chain(iterable):
    #return itertools.chain.from_iterable(iterable)

@singledispatch
def cycle(iterable):
    """Make an iterator returning elements from the iterable and saving a copy of each.
    When the iterable is exhausted, return elements from the saved copy. Repeats indefinitely.

    This function uses single dispatch.

    .. seealso:: :func:`itertools.cycle`
    """
    return itertools.cycle(iterable)

@singledispatch
def tee(iterable, n=2):
    """Return `n` independent iterators from a single iterable.

    This function uses single dispatch.

    .. seealso:: :func:`itertools.tee`
    """
    return itertools.tee(iterable, n)

@singledispatch
def toarray(iterable):
    """Store elements in `iterable` in an array.

    :rtype: :class:`np.ndarray`

    This function uses single dispatch.
    """
    return np.array(list(iterable))

@singledispatch
def repeat_each(iterable, n):
    """Repeat items in `iterable` `n` times."""
    yield from itertools.chain.from_iterable(map(lambda i: itertools.repeat(i, n), iterable))


__all__ = ['cycle', 'tee', 'repeat_each', 'toarray']
