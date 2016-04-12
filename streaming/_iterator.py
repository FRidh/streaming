"""
Iterator
========

The :mod:`streaming.iterator` module is a private module. Most algorithms for streams are implemented here using basic iterators.
"""


import cytoolz
#import toolz
import itertools
import numpy as np
import operator
from ._cython import _interpolate_linear_cython as interpolate_linear
from ._cython import _interpolate_linear as interpolate_linear

try:
    from scipy.signal import fftconvolve as _convolve
except ImportError:
    _convolve = np.convolve

def blocks(iterable, nblock, noverlap=0):
    """Partition iterable into blocks.

    :param iterable: Iterable.
    :param nblock: Samples per block.
    :param noverlap: Amount of samples to overlap
    :returns: Blocks.

    """
    # We use a different function for performance reasons
    if noverlap==0:
        return _blocks(iterable, nblock)
    else:
        return _overlapping_blocks(iterable, nblock, noverlap)


def _blocks(iterable, nblock):
    """Partition iterable into blocks.

    :param iterable: Iterable.
    :param nblock: Samples per block.
    :returns: Blocks.

    """
    iterator = iter(iterable)
    partitions = cytoolz.partition(nblock, iterator)
    yield from partitions


def _overlapping_blocks(iterable, nblock, noverlap):
    """Partition iterable into overlapping blocks of size `nblock`.

    :param iterable: Iterable.
    :param nblock: Samples per block.
    :param noverlap: Amount of samples to overlap.
    :returns: Blocks.
    """
    iterator = iter(iterable)
    nadvance = nblock - noverlap

    if nadvance < 1:
        raise ValueError("`noverlap` has to be smaller than `nblock-1`.")

    # First `noverlap` samples
    previous = list(cytoolz.take(noverlap, iterator))
    advances = map(list, cytoolz.partition(nadvance, iterator))

    for advance in advances:
        block = previous + advance # Concat lists, change type.
        yield block
        previous = block[-noverlap:]


def change_blocks(iterator, nblock, noverlap, nblock_new, noverlap_new):
    """Change blocksize and/or overlap of iterator.

    :param iterator: Iterator.
    :param nblock: Current blocksize.
    :param noverlap: Current overlap.
    :param nblock_new: New blocksize.
    :param noverlap_new: New overlap.
    :returns: Iterator with new blocksize and/or overlap.

    """

    # Same block size, same overlap
    if nblock_new==nblock and noverlap_new==noverlap:
        return iterator

    # New block size is multiple of old block size, same overlap
    elif not nblock_new % nblock and noverlap_new==noverlap:
        # factor is multiple of current blocksize
        factor = nblock_new // nblock
        # therefore we concat `factor` blocks into a new block
        partitioned = map(np.concatenate, cytoolz.partition(factor, iterator))
        return partitioned

    # Convert to samples and create blocks
    else:
        return blocks(samples(iterator, nblock, noverlap), nblock_new, noverlap_new)


def samples(iterator, nblock, noverlap=0):
    """Convert iterator with (overlapped) blocks to iterator with individual samples.

    :param iterator: Iterator.
    :param nblock: Samples per block
    :param noverlap: Amount of samples to overlap
    """
    if noverlap!=0:
        nadvance = nblock - noverlap
        iterator = map(lambda x: x[0:nadvance], iterator)
    yield from itertools.chain.from_iterable(iterator)

    :param iterable: Iterable.
    :param kind: Function to apply to each block of samples.
    """
    yield from map(kind, cytoolz.partition(nblock, iterable))


def convolve(signal, nblock, ir, initial_values=None):

    signal = blocked(nblock, signal)

    convolved = blocked_convolve(signal, ir, nblock=nblock, initial_values=initial_values)
    yield from itertools.chain(*convolved)


def blocked_convolve(signal, impulse_responses, nblock=None, ntaps=None, initial_values=None):
    """Convolve iterable `signal` with time-variant `ir`.

    :param signal: Signal.
    :param impulse_responses: Impulse responses.
    :param nblock: Samples per block.
    :param ntaps: Length of impulse responses.
    :param initial_values. Value to use before convolution kicks in.
    :returns: Convolution.

    Each item (`block`) in `signal` corresponds an impulse response (`ir`) in `impulse_responses`.
    """

    if nblock is None:
        first_block, signal = cytoolz.peek(signal)
        nblock = len(first_block)
        del first_block
    if ntaps is None:
        first_ir, impulse_responses = cytoolz.peek(impulse_responses)
        ntaps = len(first_ir)
        del first_ir

    if initial_values is None:
        tail_previous_block = np.zeros(ntaps-1)
    else:
        tail_previous_block = initial_values

    if not nblock >= ntaps:
        raise ValueError("Amount of samples in block should be the same or more than the amount of filter taps.")

    for block, ir in zip(signal, impulse_responses):

        # The result of the convolution consists of a head, a body and a tail
        # - the head and tail have length `taps-1`
        # - the body has length `signal-taps+1`??
        try:
            convolved = _convolve(block, ir, mode='full')
        except ValueError:
            raise GeneratorExit

        # The final block consists of
        # - the head and body of the current convolution
        # - and the tail of the previous convolution.
        resulting_block = convolved[:-ntaps+1]
        #resulting_block[:ntaps-1] += tail_previous_block
        resulting_block[:ntaps-1] = resulting_block[:ntaps-1] + tail_previous_block # Because of possibly different dtypes
        # We store the tail for the  next cycle
        tail_previous_block = convolved[-ntaps+1:]

        # Yield the result of this step
        yield resulting_block


def diff(iterator, initial_value=0.0):
    """Differentiate `iterator`.
    """
    current = next(iterator)
    while True:
        old = current
        current = next(iterator)
        yield current-old

def cumsum(iterator):
    """Cumulative sum.

    .. seealso:: :func:`itertools.accumulate` and :func:`np.cumsum`
    """
    yield from itertools.accumulate(iterator, operator.add)


def cummul(iterator):
    """Cumulative p.

    .. seealso:: :func:`itertools.accumulate` and :func:`np.cumsum`
    """
    yield from itertools.accumulate(iterator, operator.mul)


def vdl(signal, times, delay, initial_value=0.0):
    """Variable delay line which delays `signal` at 'times' with 'delay'.

    :param signal: Signal to be delayed.
    :type signal: Iterator
    :param delay: Delay.
    :type delay: Iterator
    :param initial_value: Sample to yield before first actual sample is yielded due to initial delay.

    .. note:: Times and delay should have the same unit, e.g. both in samples or both in seconds.
    """
    dt0, delay = cytoolz.peek(delay)
    times, _times = itertools.tee(times)

    # Yield initial value before interpolation kicks in
    # Note that this method, using tee, buffers all samples that will be discarded.
    # Therefore, room for optimization!
    n = 0
    if initial_value is not None:
        while next(_times) < dt0:
            n += 1
            yield initial_value

    times1, times2 = itertools.tee(times)
    interpolated = interpolate_linear(map(operator.add, times2, delay), signal, times1)
    yield from cytoolz.drop(n, interpolated) # FIXME: move drop before interpolation, saves memory


__all__ = ['blocked', 'blocked_convolve', 'convolve', 'diff', 'interpolate_linear', 'vdl']
