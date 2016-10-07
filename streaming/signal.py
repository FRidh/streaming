"""
Signal Processing
=================
The Signal Processing module contains functions for signal processing.

"""

from multipledispatch import dispatch, Dispatcher
import numbers
import numpy as np
import noisy
import streaming
from streaming.stream import Stream, BlockStream, count
import itertools
from functools import singledispatch
from streaming._iterator import _convolve



def constant(value, nblock=None):
    """Stream with constant value.

    :rtype: :class:`Stream` or :class:`BlockStream` if `nblock` is not `None`.
    """

    if nblock is not None:
        return BlockStream([np.ones(nblock)*value], nblock).cycle()
    else:
        return Stream([value]).cycle()


def convolve_overlap_add(signal, impulse_responses, nhop, ntaps, initial_values=None):
    """Convolve `signal` with `impulse_responses`.

    :param signal: Signal
    :type signal: :class:`Stream` or :class:`BlockStream`
    :param impulse_responses: :class:`Stream`
    :param ntaps: Amount of taps.
    :returns: Convolution of `signal` with `impulse_responses`.
    :rtype: :class:`BlockStream`

    .. seealso:: :func:`streaming._iterator.blocked_convolve`
    """
    noverlap = 0
    signal = signal.blocks(nhop, noverlap=noverlap)
    return BlockStream(streaming._iterator.convolve_overlap_add(signal._iterator, impulse_responses._iterator, nhop=nhop, ntaps=ntaps,
                                                       initial_values=initial_values), nblock=nhop, noverlap=noverlap)

# Backwards compatibility
convolve = convolve_overlap_add


def convolve_overlap_save(signal, impulse_responses, nhop, ntaps):
    """Convolve signal with linear time-variant `impulse_responses` using overlap-save method.

    :param signal: Signal.
    :param impulse_responses: Impulse responses of the filter. Each impulse response belongs to a hop.
    :param nhop: Hop in samples.
    :param ntaps: Length of each impulse response.
    :returns: Stream with blocksize equal to `nhop`.
    :rtype: BlockStream

    """
    #return BlockStream(streaming._iterator.convolve_overlap_save(signal.samples()._iterator, impulse_responses._iterator, nhop, ntaps), nblock=nhop)

    # It can be more efficient to repeat code here, because with Stream/BlockStream we don't always need to convert from sample-based to block-based.
    nwindow = nhop + ntaps - 1
    noverlap = ntaps - 1
    windows = signal.blocks(nblock=nwindow, noverlap=noverlap)
    # Convolve function to use
    _convolve_func = lambda x, y: _convolve(x, y, mode='valid')
    # Convolved blocks
    convolved = BlockStream(iter(map(_convolve_func, windows, impulse_responses )), nblock=nhop, noverlap=0)
    return convolved


def vdl(signal, times, delay, initial_value=0.0):
    """Variable delay line which delays `signal` at `times` with `delay`.

    :param signal: Signal to be delayed.
    :param times: Sample times corresponding to signal before delay.
    :param delay: Delays to apply to signal.
    :param initial_value: Value to return before first sample.
    :returns: Delayed version of `signal`.
    :rtype: :class:`Stream`

    .. seealso:: :func:`streaming._iterator.vdl`
    """
    return Stream(streaming._iterator.vdl(signal.samples()._iterator, times.samples()._iterator, delay.samples()._iterator, initial_value=initial_value))


def interpolate(x, y, xnew):
    """Interpolate `y` at `xnew`.

    :param x: Previous sample positions
    :param y: Previous sample values
    :param xnew: New sample positions
    :returns: Interpolated sample positions
    :rtype: :class:`Stream`

    .. seealso:: :func:`streaming._iterator.interpolate_linear`
    """
    return Stream(streaming._iterator.interpolate_linear(x.samples()._iterator, y.samples()._iterator, xnew.samples()._iterator))


def sine(frequency, fs):
    """Sine with `frequency` and sample frequency `fs`.

    :param frequency: Frequency of the sine.
    :param fs: Sample frequency.
    :returns: Sine.
    :rtype: :class:`Stream`
    """
    return np.sin(2.*np.pi*frequency*times(1./fs))


def times(dt):
    if isinstance(dt, numbers.Number):
        return count(step=dt)
    else:
        raise ValueError("dt has to be a scalar number.")




def noise(nblock=None, state=None, color='white', ntaps=None):
    """Generate white noise with standard Gaussian distribution.

    :param nblock: Amount of samples per block.
    :param state: State of PRNG.
    :type state: :class:`np.random.RandomState`
    :returns: When `nblock=None`, individual samples are generated and a :class:`streaming.Stream` is
    returned. When integer, a :class:`streaming.BlockStream` is returned.
    """
    if state is None:
        state = np.random.RandomState()

    # Generate white noise
    if nblock is None:
        # Generate individual samples.
        stream = Stream((state.randn(1)[0] for i in itertools.count()))
    else:
        # Generate blocks.
        stream = BlockStream((state.randn(nblock) for i in itertools.count()), nblock=nblock, noverlap=0)

    # Apply filter for specific color
    if color is not 'white':
        if ntaps is None:
            raise ValueError("Amount of taps has not been specified.")
        ir = noisy.COLORS[color](ntaps)
        if nblock is None:
            nhop = ntaps
        else:
            nhop = max(nblock, ntaps)
        stream = convolve_overlap_add(stream, constant(ir), nhop, ntaps).samples().drop(ntaps//2)

    # Output as desired
    if nblock is None:
        return stream.samples()
    else:
        return stream.blocks(nblock)


def cumsum(x):
    """Cumulative sum.

    .. seealso:: :func:`itertools.accumulate`
    """
    return Stream(streaming._iterator.cumsum(x.samples()._iterator))


def diff(x):
    """Differentiate `x`.

    :returns: Differentiated `x`.
    :rtype: :class:`Stream`

    .. seealso:: :func:`streaming._iterator.diff`
    .. note:: Typically when differentiating one needs to multiple as well with the sample frequency
    """
    return Stream(streaming._iterator.diff(x.samples()._iterator))


def filter_ba(x, b, a):
    """Apply IIR filter to `x`.

    :param x: Signal.
    :param b: Numerator coefficients.
    :param a: Denominator coefficients.
    :returns: Filtered signal.

    .. seealso:: :func:`scipy.signal.lfilter`
    """
    return Stream(streaming._iterator.filter_ba(x.samples()._iterator, b, a))


def filter_sos(x, sos):
    """Apply IIR filter to `x`.

    :param x: Signal.
    :param sos: Second-order sections.
    :returns: Filtered signal.

    .. seealso:: :func:`scipy.signal.sosfilt`

    """
    return Stream(streaming._iterator.filter_sos(x.samples()._iterator, sos))


__all__ = ['constant', 'convolve', 'convolve_overlap_add', 'convolve_overlap_save', 'interpolate', 'noise', 'sine', 'times', 'vdl']
