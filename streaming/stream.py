"""
Stream
======

The Stream module contains the classes :class:`Stream` and :class:`BlockStream`.

"""
import collections
import itertools
import numpy as np
import operator
import cytoolz

from multipledispatch.conflict import AmbiguityWarning
import warnings
warnings.filterwarnings("ignore", category=AmbiguityWarning)

#import streaming
from streaming.operators import *
from streaming.abstractstream import *
from streaming.itertools import *

class Stream(AbstractStream):
    """Stream of samples.
    """

    def _construct(self, iterator):
        return Stream(iterator)

    @property
    def nblock(self):
        raise AttributeError("Stream does not have a blocksize")

    def blocks(self, nblock, noverlap=0):
        blocks = map(np.array, streaming._iterator.blocks(self._iterator, nblock, noverlap))
        return BlockStream(blocks, nblock, noverlap)

    def drop(self, nsamples):
        """Drop the first `n` samples."""
        return super().drop(nsamples)

    def map(self, func):
        return self._construct(map(func, self._iterator))

    def take(self, nsamples):
        """Take the first `nsamples` samples."""
        return super().take(nsamples)

    def samples(self):
        return self


class BlockStream(AbstractStream):
    """Stream of blocks of samples.
    """

    def __init__(self, iterator, nblock, noverlap=0):

        # We either peek, and we know the type and blocksize assuming a homogeneous stream
        # or we don't and we need nblock as argument and hope the blocks are of the right type.
        #try:
        #first, iterator = peek(iterator)
        #except StopIteration:
                #raise ValueError("Empty iterator.")
        ## The elements in the iterator have to be iterators.
        ## Because the idea of the blockstream is to use vectorized operations
        ## we require a stricter class, np.ndarray
        #if not isinstance(first, np.ndarray):
            #raise ValueError("Items in iterator have to be of class `np.ndarray`.")

        # With the above disabled we get failing tests for eq and ne operations
        # See https://github.com/numpy/numpy/issues/5016

        super().__init__(iterator)

        if noverlap > nblock-1:
            raise ValueError("`noverlap` should be smaller than `nblock-1`")

        self._nblock = nblock
        self._noverlap = noverlap

    def _construct(self, iterator):
        return BlockStream(iterator, self.nblock, self.noverlap)

    @property
    def nblock(self):
        return self._nblock

    @property
    def noverlap(self):
        return self._noverlap

    def blocks(self, nblock, noverlap=0):

        blocks = streaming._iterator.change_blocks(self._iterator, self.nblock, self.noverlap, nblock, noverlap)
        return BlockStream(map(np.array, blocks), nblock, noverlap)

    def drop(self, n):
        """Drop the first `n` blocks.

        .. note:: If you want to drop `n` samples, use `s.samples().drop(n)`.
        """
        return self._construct(cytoolz.drop(n, self))

    def map(self, func):
        """Map `func` to each block in :class:`BlockStream`.
        """
        return self._construct(map(func, self))

    def mean(self):
        """Mean value calculated over `nblock` samples.

        This function returns a :class:`Stream`.
        """
        return Stream(self.map(np.mean))

    def nblocks(self):
        """Amount of blocks in stream.

        .. warning:: This consumes the stream.
        """
        return count(self)

    def std(self):
        """Standard deviation calculated over `nblock` samples.

        This function returns a :class:`Stream`.
        """
        return Stream(self.map(np.std))

    def sum(self):
        """Sum calculated over `nblock` samples.

        This function returns a :class:`Stream`.
        """
        return Stream(self.map(np.sum))

    def take(self, nblocks):
        """Take `nblocks` from stream.
        """
        return self._construct(cytoolz.take(nblocks, self._iterator))

    def samples(self):
        """Iterate over samples.

        :returns: Stream of samples insteads of blocks. Possible overlap is taking into account.

        """
        return Stream(streaming._iterator.samples(self._iterator, self.nblock, self.noverlap))

    def var(self):
        """Variance calculated over `nblock` samples.

        This function returns a :class:`Stream`.
        """
        return Stream(self.map(np.var))


# Binary operators (AbstractStream, Stream)
def _binary_op_abstractstream_stream(op, a, b):
    return _binary_op_abstractstream_abstractstream(op, a, b)

# Binary operators (AbstractStream, BlockStream)
def _binary_op_abstractstream_blockstream(op, a, b):
    return _binary_op_abstractstream_stream(op, a, b.samples())

# Binary operators (Stream, Stream)
def _binary_op_stream_stream(op, a, b):
    return Stream(map(op, a._iterator, b._iterator))

# Binary operators (BlockStream, BlockStream)
def _binary_op_blockstream_blockstream(op, a, b):
    if a.nblock == b.nblock:
        return BlockStream(map(op, a, b), nblock=a.nblock)
    #elif a.nblock % b.nblock == 0:
        ## A is multiple of B
        #pass
    #elif b.nblock % a.nblock == 0:
        ## B is multiple of A
        #pass
    else:
        return op(a.samples(), b.samples())

# Binary operators (Stream, object)
def _binary_op_stream_object(op, a, b):
    return Stream(op(i, b) for i in a._iterator)

# Binary operators (object, Stream)
def _binary_op_object_stream(op, a, b):
    return Stream(op(a, i) for i in b._iterator)

# Binary operators (BlockStream, object)
def _binary_op_blockstream_object(op, a, b):
    return BlockStream((op(i, b) for i in a._iterator), a.nblock)

# Binary operators (object, BlockStream)
def _binary_op_object_blockstream(op, a, b):
    return BlockStream((op(a, i) for i in b._iterator), b.nblock)

# Binary operators (Stream, BlockStream)
def _binary_op_stream_blockstream(op, a, b):
    return op(a, b.samples())
    #return Stream(map(op, a._iterator, b._iterator))

# Binary operators (BlockStream, Stream)
def _binary_op_blockstream_stream(op, a, b):
    return op(a.samples(), b)
    #return BlockStream(map(op, a._iterator, b._iterator), a.nblock)

for op in streaming.operators._BINARY_OPERATORS:
    # Get the dispatcher for this operation
    D = getattr(streaming.operators, op)
    # And add the specific implementations
    D.add((Stream, Stream), partial(_binary_op_stream_stream, D))
    D.add((BlockStream, BlockStream), partial(_binary_op_blockstream_blockstream, D))
    D.add((AbstractStream, Stream), partial(_binary_op_abstractstream_stream, D))
    D.add((AbstractStream, BlockStream), partial(_binary_op_abstractstream_blockstream, D))
    D.add((Stream, object), partial(_binary_op_stream_object, D))
    D.add((object, Stream), partial(_binary_op_object_stream, D))
    D.add((BlockStream, object), partial(_binary_op_blockstream_object, D))
    D.add((object, BlockStream), partial(_binary_op_object_blockstream, D))
    D.add((Stream, BlockStream), partial(_binary_op_stream_blockstream, D))
    D.add((BlockStream, Stream), partial(_binary_op_blockstream_stream, D))

# Itertools

@tee.register(Stream)
def _(iterable, n=2):
    return tuple(Stream(it) for it in itertools.tee(iterable, n))

@cycle.register(Stream)
def _(iterable):
    return Stream(itertools.cycle(iterable))

@toarray.register(Stream)
def _(iterable):
    return np.array(list(iterable))

@tee.register(BlockStream)
def _(iterable, n=2):
    return tuple(BlockStream(it, iterable.nblock) for it in itertools.tee(iterable, n))

@cycle.register(BlockStream)
def _(iterable):
    return BlockStream(itertools.cycle(iterable), nblock=iterable.nblock)

@toarray.register(BlockStream)
def _(iterable):
    return iterable.samples().toarray()

# Other helpful functions

def count(start=0, step=1):
    """Count.

    .. seealso:: :func:`itertools.count`

    """
    return Stream(itertools.count(start=start, step=step))

@repeat_each.register(Stream)
def _(iterable, n):
    return Stream(repeat_each(iterable._iterator, n))

@repeat_each.register(BlockStream)
def _(iterable, n):
    return repeat_each(iterable.samples(), n).blocks(iterable.nblock)

__all__ = ['Stream', 'BlockStream', 'count']
