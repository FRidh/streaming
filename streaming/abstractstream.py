"""
Abstract Streams
================

The module :mod:`streaming.abtractstream` contains an abstract base class for streams.
Furthermore, additional signatures are added to allow the operators defined in :mod:`streaming.operators` to work with `class:`streaming.abstractstream.AbstractStream`.

"""

import abc
import collections
from functools import partial
import itertools
from multipledispatch.conflict import AmbiguityWarning
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=AmbiguityWarning)
import streaming
from streaming.operators import *
from streaming.itertools import *

import cytoolz
#from streaming.operators import _basic_objects

## Add basic operators to module
import sys
_thismodule = sys.modules[__name__]

#def _binary_operator(operator):
    #"""Apply `operator` to binary inputs.
    #"""
    #def op(x, y):
        #if isinstance(y, collections.Stream):
            #yield from map(operator, x, y)
        #else: # Consider it a constant
            #yield from map(lambda i: operator(i, y) for i in x)
    #return op

#def _reversed_binary_operator(operator):
    #"""Apply `operator` to binary inputs.
    #"""
    #def op(y, x):
        #if isinstance(y, collections.Stream):
            #yield from map(operator, x, y)
        #else: # Consider it a constant
            #yield from map(lambda i: operator(i, y) for i in x)
    #return op


###def _unary_operator(operator):
    ###"""Apply `operator` to unary input.
    ###"""
    ###def op(x):
        ###yield from map(operator, x)
    ###return op


#class MetaStream(abc.ABCMeta):

    #def __new__(cls, name, bases, attrs):

        ## Add binary operators
        #for op in streaming.operators._BINARY_OPERATORS:
            #attrs['__'+op+'__'] = getattr(streaming.operators, op)
            ##attrs['__r'+op+'__'] = _reversed_binary_operator(getattr(operator, op))
            ##attrs['__'+op+'__'] = _binary_operator(getattr(operator, op))
            ##attrs['__r'+op+'__'] = _reversed_binary_operator(getattr(operator, op))
            ##setattr(cls, '__'+op+'__', getattr(streaming.operators, op))
        #attrs['__div__'] = attrs['__truediv__']

        ### Add unary operators
        ##for operator in _UNARY_OPERATORS:
            ##attrs['__'+operator+'__'] = _unary_operator(operator)

        #return super().__new__(cls, name, bases, attrs)


def _wrapped_binary_op(op):
    def wrapped(a, b):
        return op(a, b)
    return wrapped

def _reverse_wrapped_binary_op(op):
    """Swap inputs."""
    def reverser(a, b):
        #print("Op: {}, Left: {}, Right {}".format(op, a, b))
        def wrapped(x, y):
            #print("Reversed, now Op: {}, Left: {}, Right {}".format(op, x, y))
            return op(x, y)
        return wrapped(b, a)
    return reverser


class Operators(abc.ABCMeta):

    def __new__(cls, name, bases, attrs):

        cls = super().__new__(cls, name, bases, attrs)

        import streaming

        # Add binary operators
        for op in streaming.operators._BINARY_OPERATORS:
            setattr(cls, '__'+op+'__', _wrapped_binary_op(getattr(streaming.operators, op)))
        for op in streaming.operators._BINARY_OPERATORS:#_WITH_REVERSE:
            setattr(cls, '__r'+op+'__', _reverse_wrapped_binary_op(getattr(streaming.operators, op)))
        return cls


class AbstractStream(collections.Iterator, metaclass=Operators):
    """Abstract stream."""

    __array_priority__ = 1000
    """Array priority is required to force usage of __radd__ in case left-side object is an `np.ndarray`."""
    __numpy_ufunc__ = None
    """Disable np ufuncs entirely, force usage of __rop__."""

    def __init__(self, iterator):

        if not isinstance(iterator, collections.abc.Iterable):
            raise ValueError("Iterable required, iterator preferred.")
        elif isinstance(iterator, collections.abc.Iterable):
            iterator = iter(iterator)

        # Try and find the actual iterator
        # Could move this into elif
        obj = iterator
        while hasattr(obj, '_iterator'):
            obj = obj._iterator
        self._iterator = obj

        super().__init__()

    ##def __bool__(self):
        #return True

    def __iter__(self):
        yield from self._iterator

    def __next__(self):
        return next(self._iterator)

    #def _fastmap(self, func):
        #return self.map(func)

    @abc.abstractproperty
    def nblock(self):
        """Amount of samples per block."""
        pass

    @abc.abstractmethod
    def _construct(self, iterable):
        """Construct instance of this type from given `iterable`"""
        pass

    @abc.abstractmethod
    def blocks(self):
        """Stream with blocks.

        :rtype: :class:`BlockStream`
        """
        pass

    def cos(self):
        """Cosine."""
        return self.map(np.cos)

    #@abc.abstractmethod
    def copy(self):
        """Make a copy of the stream."""
        self._iterator, iterator = itertools.tee(self._iterator)
        return self._construct(iterator)

    def cycle(self):
        """Cycle the stream.

        .. seealso:: :func:`itertools.cycle`
        """
        return streaming.itertools.cycle(self)

    @abc.abstractmethod
    def drop(self, n):
        """Drop the first `n` items."""
        return self._construct(cytoolz.drop(n, self))

    def exp(self):
        """Exponential."""
        return self.map(np.exp)

    @abc.abstractmethod
    def map(self, func):
        """Map `func` to each sample in `Stream`.
        """
        return AbstractStream(map(func, self._iterator))
        #stream = self.samples()
        #return type(stream)(map(func, stream))

    #@abc.abstractmethod
    #def mapblock(self, func):
        #pass

    def nsamples(self):
        """Amount of samples in stream.

        .. warning:: This consumes the stream.
        """
        return cytoolz.count(self.samples())

    def peek(self):
        """Check the first item in the stream."""
        first, self._iterator = cytoolz.peek(self._iterator)
        return first

    def repeat_each(self, n):
        """Repeat each item `n` times before yielding the next.

        """
        return repeat_each(self, n)

    @abc.abstractmethod
    def samples(self):
        """Stream with samples.

        :rtype: :class:`Stream`
        """
        pass

    def sin(self):
        """Sine."""
        return self.map(np.sin)

    def sqrt(self):
        """Square root."""
        return self.map(np.sqrt)

    @abc.abstractmethod
    def take(self, n):
        """Take the first `n`."""
        return self._construct(cytoolz.take(n, self._iterator))

    def take_nth(self, n):
        """Take every `n`th."""
        return self._construct(cytoolz.take_nth(n, self._iterator))

    def tan(self):
        """Tangens."""
        return self.map(np.tan)

    def tee(self, n=2):
        """Split stream in `n` streams.

        .. seealso:: :func:`itertools.tee`
        """
        return tee(self, n=n)

    def toarray(self):
        """Convert to array.

        :rtype: :class:`np.ndarray`
        """
        return toarray(self)


# Binary operators (AbstractStream, object)
def _binary_op_abstractstream_object(op, a, b):
    return AbstractStream(op(i, b) for i in a._iterator)

# Binary operators (object, AbstractStream)
def _binary_op_object_abstractstream(op, a, b):
    return AbstractStream(op(a, i) for i in b._iterator)

# Binary operators (AbstractStream, AbstractStream)
def _binary_op_abstractstream_abstractstream(op, a, b):
    return AbstractStream(map(op, a._iterator, b._iterator))

for op in streaming.operators._BINARY_OPERATORS:
    # Get the dispatcher for this operation
    D = getattr(streaming.operators, op)
    # And add the specific implementations
    D.add((AbstractStream, AbstractStream), partial(_binary_op_abstractstream_abstractstream, D))
    D.add((AbstractStream, object), partial(_binary_op_abstractstream_object, D))
    D.add((object, AbstractStream), partial(_binary_op_object_abstractstream, D))


# Itertools

@tee.register(AbstractStream)
def _(iterable, n=2):
    return tuple(AbstractStream(it) for it in itertools.tee(iterable, n))

@cycle.register(AbstractStream)
def _(iterable):
    return AbstractStream(iterable)(itertools.cycle(iterable))

# Other helpful functions

def count(start=0, step=1):
    return AbstractStream(itertools.count(start=start, step=step))

@repeat_each.register(AbstractStream)
def _(iterable, n):
    return AbstractStream(repeat_each(iterable._iterator, n))
