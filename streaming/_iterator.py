"""
Iterator
========

The :mod:`streaming.iterator` module is a private module. Most algorithms for streams are implemented here using basic iterators.
"""
import cytoolz
import itertools
import operator
import numpy as np
import operator
import collections
from ._cython import _interpolate_linear as interpolate_linear
from ._cython import _filter_ba, diff

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
        block = previous + advance # Concat lists
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

    # Old block size is multiple of new block size, sample overlap
    elif not nblock % nblock_new and noverlap_new==noverlap:
        # Partition each block in blocks with size nblock_new
        partition = lambda x: cytoolz.partition(nblock_new, x)
        # And chain the iterables
        partitioned = itertools.chain.from_iterable(map(partition, iterator))
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


# Some convenience functions

def sliding_mean(iterable, nwindow, noverlap=0):
    """Sliding mean.

    :param iterable: Iterable.
    :param nwindow: Window size in samples.
    :param noverlap: Amount of samples to overlap.
    :returns: Iterable of means.
    """
    yield from map(np.mean, blocks(iterable, nwindow, noverlap))


def sliding_std(iterable, nwindow, noverlap=0):
    """Sliding standard deviation.

    :param iterable: Iterable.
    :param nwindow: Window size in samples.
    :param noverlap: Amount of samples to overlap.
    :returns: Iterable of standard deviations.
    """
    yield from map(np.std, blocks(iterable, nwindow, noverlap))


def sliding_var(iterable, nwindow, noverlap=0):
    """Sliding variance.

    :param iterable: Iterable.
    :param nwindow: Window size in samples.
    :param noverlap: Amount of samples to overlap.
    :returns: Iterable of standard deviations.
    """
    yield from map(np.var, blocks(iterable, nwindow, noverlap))


# Convolution

def convolve(signal, impulse_responses, nblock, ntaps=None, initial_values=None):
    """Convolve signal with impulse response.

    :param signal: Signal, not in blocks.
    :param impulse_responses: Impulse responses of length `ntaps`.
    :param nblock: Blocksize to use for the convolution.
    :param ntaps: Length of impulse responses.
    :param initial_values. Value to use before convolution kicks in.

    .. note:: This function takes samples and yields samples. It wraps :func:`convolve_overlap_add` and therefore requires a blocksize for computing the convolution.

    """
    signal = blocks(signal, nblock)
    convolved = convolve_overlap_add(signal, impulse_responses, nblock=nblock, ntaps=ntaps, initial_values=initial_values)
    yield from itertools.chain.from_iterable(convolved)

def _convolve_crossfade(block, ir1, ir2, fading1):
    """Convolve block with two impulse responses and crossfade the result.

    :param block: Block.
    :param ir1: Impulse response 1.
    :param ir2: Impulse response 2.
    :param fading1: Fading.
    :returns: Crossfaded convolutions of block and impulse responses.

    We switch from `ir1` to `ir2`.

    A more efficient method is presented in *Efficient time-varying FIR filtering using
    crossfading implemented in the DFT domain* by Frank Wefers.

    """
    # Convolve segment with both impulse responses.
    convolved1 = _convolve(block, ir1, mode='full')
    convolved2 = _convolve(block, ir2, mode='full')
    # Fading windows
    fading2 = 1. - fading1
    # Crossfaded convolutions
    convolved = convolved1 * fading1 + convolved2 * fading2
    return convolved


def convolve_overlap_add_spectra(signal, spectra, nblock, nbins, initial_values=None):
    """Convolve iterable `signal` with impulse responses of `spectra.


    This function directly uses the spectra to compute the acyclic convolution.

    """
    return NotImplemented

def convolve_overlap_add(signal, impulse_responses, nhop, ntaps, initial_values=None):
    """Convolve iterable `signal` with time-variant `impulse_responses`.

    :param signal: Signal in blocks of size `nblock`.
    :param impulse_responses: Impulse responses of length `ntaps`.
    :param nhop: Impulse responses is updated every `nhop` samples. This should correspond to the blocksize of `signal`.
    :param ntaps: Length of impulse responses.
    :param initial_values. Value to use before convolution kicks in.
    :returns: Convolution.

    This function implements the overlap-add method. Time-variant `impulse_responses is supported.
    Each item (`block`) in `signal` corresponds to an impulse response (`ir`) in `impulse_responses`.

    This implementation of overlap-add buffers only one segment. Therefore, only `nblock>=ntaps` is supported.

    .. warning:: This function cannot be used when `ntaps > nblock`.

    .. seealso:: :func:`convolve_overlap_save`

    """
    nblock = nhop

    if not nblock >= ntaps:
        raise ValueError("Amount of samples in block should be the same or more than the amount of filter taps.")

    if initial_values is None:
        tail_previous_block = np.zeros(ntaps-1)
    else:
        tail_previous_block = initial_values

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
        #print(resulting_block)
        #resulting_block[:ntaps-1] += tail_previous_block
        resulting_block[:ntaps-1] = resulting_block[:ntaps-1] + tail_previous_block # Because of possibly different dtypes
        # We store the tail for the  next cycle
        tail_previous_block = convolved[-ntaps+1:]
        #print(tail_previous_block)

        # Yield the result of this step
        yield resulting_block


def convolve_overlap_discard(signal, impulse_response, nblock_in=None, nblock_out=None):
    """Convolve signal with linear time-invariant `impulse_response` using overlap-discard method.

    :param signal: Signal. Can either consists of blocks or samples. `nblock_in` should be set to the block size of the signal.
    :param impulse_response: Linear time-invariant impulse response of filter.
    :param nblock_in: Actual input blocksize of signal. Should be set to `None` is `signal` is sample-based.
    :param nblock_out: Desired output blocksize.
    :returns: Convolution of `signal` with `impulse_response`.
    :rtype: Generator consisting of arrays.

    Setting the input blocksize can be useful because this gives control over the delay of the process.
    Setting the output blocksize is convenient because you know on beforehand the output blocksize.
    Setting neither will result in blocksize of one, or individual samples. This will be slow.
    Setting both is not possible.


    .. note:: The *overlap-discard* method is more commonly known as *overlap-save*.

    """
    # Amount of filter taps
    ntaps = len(impulse_response)
    # Amount of overlap that is needed
    noverlap = ntaps -1

    # In the following block we create overlapping windows.

    # Both are set.
    if nblock_in is not None and nblock_out is not None:
        raise ValueError("Set block size of either input or output.")

    # Only output blocksize is explicitly mentioned
    elif nblock_out is not None:
        nblock_in = nblock_out + ntaps -1
    # Only input blocksize is explicitly mentioned
    elif nblock_in is not None:
        if not nblock_in >= ntaps:
            raise ValueError("Amount of samples in block should be the same or more than the amount of filter taps.")
        nblock_out = nblock_in - ntaps + 1
    else:
        nblock_in = ntaps
        nblock_out = nblock_in - ntaps + 1

    windows = blocks(signal, nblock_in, noverlap)

    ## We have sample-based signal and we want blocks with specified size out.
    #if nblock_in is None and nblock_out is not None:
        #nblock_in = nblock_out + ntaps -1
        #windows = blocks(signal, nblock_in, noverlap)
    ## We have sample-based signal and we want samples out (actually blocks of size 1).
    #elif nblock_in is None and nblock_out is None:
        #nblock_in = ntaps
        #windows = blocks(signal, nblock_in, noverlap)
    ## We have block-based signal and we don't mind output block size
    #elif nblock_in is not None and nblock_out is None:
        #if not nblock_in >= ntaps:
            #raise ValueError("Amount of samples in block should be the same or more than the amount of filter taps.")
        #nblock_out = nblock_in - ntaps + 1
        #windows = change_blocks(signal, nblock_in, 0, nblock_in, noverlap)
    ## We have block-based signal and we have specified an output block. We need to change the block size.
    #elif nblock_in is not None and nblock_out is not None:
        #if not nblock_in >= ntaps:
            #raise ValueError("Amount of samples in block should be the same or more than the amount of filter taps.")
        #nblock_in_new = nblock_out + ntaps -1
        #windows = change_blocks(signal, nblock_in, 0, nblock_in_new, noverlap, )
        #nblock_in = nblock_in_new

    # Convolve function to use
    _convolve_func = lambda x: _convolve(x, impulse_response, mode='valid')

    # Convolved blocks
    convolved = map(_convolve_func, windows )

    return convolved, nblock_out


def convolve_overlap_save(signal, impulse_responses, nhop, ntaps):
    """Convolve signal with linear time-invariant `impulse_response` using overlap-discard method.

    :param signal: Signal. Consists of samples.
    :param impulse_responses: Linear time-variant impulses response of filter.
    :param nhop: Impulse response is renewed every `nhop` samples.
    :returns: Convolution of `signal` with `impulse_responses`.
    :rtype: Generator consisting of arrays.

    .. note:: The *overlap-discard* method is more commonly known as *overlap-save*.

    """
    nwindow = nhop + ntaps - 1
    noverlap = ntaps - 1
    windows = blocks(signal, nwindow, noverlap)
    # Convolve function to use
    _convolve_func = lambda x, y: _convolve(x, y, mode='valid')
    # Convolved blocks
    convolved = map(_convolve_func, windows, impulse_responses )
    yield from convolved


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


#def diff(iterator, initial_value=0.0):
    #"""Differentiate `iterator`.
    #"""
    #current = next(iterator)
    #while True:
        #old = current
        #current = next(iterator)
        #yield current-old

#def integrate(iterator, initial_value=0.0):
    #total = 0.0
    #while True:
        #total += next(iterator)
        #yield total

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


# Reference implementation

def filter_ba_reference(x, b, a):
    """Filter signal `x` with linear time-invariant IIR filter that has numerator coefficients `b` and denominator coefficients `a`.

    :param b: Numerator coefficients.
    :param a: Denominator coefficients. The first value is always a zero.
    :param x: Signal.
    :returns: Filtered signal.

    This function applies a linear time-invariant IIR filter using the difference equation

    .. math:: y[n] = -\sum_k=1^M a_k y[n-k] + \sum_k=0^(N-1) b_k x[n-k]

    """
    b = np.array(b)
    a = np.array(a[1:])
    na = len(a)
    nb = len(b)

    # Buffers
    xd = collections.deque([0]*nb, nb)
    yd = collections.deque([0]*na, na)

    # Invert filter coefficients order
    b = b[::-1]
    a = a[::-1]

    while True:
        # Update inputs buffer with new signal value
        xd.append(next(x))
        # Calculate output from difference equation
        result = -sum(a*yd) + sum(b*xd)
        # Update outputs buffer with new output value
        yd.append(result)
        # Yield current output value
        yield result


def filter_ba(x, b, a):
    """Apply IIR filter to `x`.

    :param x: Signal.
    :param b: Numerator coefficients.
    :param a: Denominator coefficients.
    :returns: Filtered signal.

    .. seealso:: :func:`filter_sos` and :func:`scipy.signal.lfilter`
    """
    a = a[1:] # Drop the first value that is a one. See difference equation.

    # Drop trailing zeros. They're not contributing and introduce a bug as well (leading zero in result).
    while a[-1] == 0.0:
        a = a[:-1]
    while b[-1] == 0.0:
        b = b[:-1]
    a = np.array(a)
    b = np.array(b)

    na = len(a)
    nb = len(b)

    # Buffers
    xd = np.zeros(nb)
    yd = np.zeros(na)

    yield from _filter_ba(x, b, a, xd, yd, nb, na)


def filter_sos(x, sos):
    """Apply IIR filter to `x`.

    :param x: Signal.
    :param sos: Second-order sections.
    :returns: Filtered signal.

    .. seealso:: :func:`filter_ba` and :func:`scipy.signal.sosfilt`

    """
    sos = np.atleast_2d(sos)
    if sos.ndim != 2:
        raise ValueError('sos array must be 2D')

    n_sections, m = sos.shape
    if m != 6:
        raise ValueError('sos array must be shape (n_sections, 6)')

    for section in sos:
        x = filter_ba(x, section[:3], section[3:])
    yield from x


__all__ = ['blocks', 'convolve_overlap_add', 'convolve', 'convolve_overlap_discard', 'diff', 'interpolate_linear', 'filter_ba', 'filter_ba_reference', 'filter_sos', 'vdl']
