"""
Signal Processing
=================
The Signal Processing module contains functions for signal processing.

"""

from multipledispatch import dispatch, Dispatcher
import numbers
import numpy as np
import streaming
from streaming.stream import Stream, BlockStream, count
import itertools

def times(dt):
    if isinstance(dt, numbers.Number):
        return count(step=dt)
    else:
        raise ValueError("dt has to be a scalar number.")

def convolve(signal, impulse_responses, nblock, ntaps=None, initial_values=None):
    """Convolve `signal` with `impulse_responses`.

    :param signal: Signal
    :type signal: :class:`Stream` or :class:`BlockStream`
    :param impulse_responses: :class:`Stream`
    :returns: Convolution of `signal` with `impulse_responses`.
    :rtype: :class:`BlockStream`

    .. seealso:: :func:`streaming._iterator.blocked_convolve`
    """
    signal = signal.blocks(nblock)
    noverlap = 0
    return BlockStream(streaming._iterator.blocked_convolve(signal.blocks(nblock)._iterator, impulse_responses._iterator, nblock=nblock,
                                                            ntaps=ntaps, initial_values=initial_values), nblock=nblock, noverlap=noverlap)


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


def constant(value, nblock=None):
    """Stream with constant value.

    :rtype: :class:`Stream` or :class:`BlockStream` if `nblock` is not `None`.
    """

    if nblock is not None:
        return BlockStream([np.ones(nblock)*value], nblock).cycle()
    else:
        return Stream([value]).cycle()


def sine(frequency, fs):
    """Sine with `frequency` and sample frequency `fs`.

    :param frequency: Frequency of the sine.
    :param fs: Sample frequency.
    :returns: Sine.
    :rtype: :class:`Stream`
    """
    return np.sin(2.*np.pi*frequency*times(1./fs))


def noise(nblock=None, state=None):
    """Generate white noise with standard Gaussian distribution.

    :param nblock: Amount of samples per block.
    :param state: State of PRNG.
    :type state: :class:`np.random.RandomState`
    :returns: When `nblock=None`, individual samples are generated and a :class:`streaming.Stream` is
    returned. When integer, a :class:`streaming.BlockStream` is returned.
    """
    if state is None:
        state = np.random.RandomState()

    if nblock is None:
        # Return individual samples.
        return Stream((state.randn(1)[0] for i in itertools.count()))
    else:
        # Return blocks.
        return BlockStream((state.randn(nblock) for i in itertools.count()), nblock=nblock, noverlap=0)


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



#def integrate(x):
    #"""Integrate `x`.
    #"""
    #return



#class Filterbank(object):

    #def __init__(self, frequencies):
        #pass


#__all__ = ['constant', 'convolve', 'interpolate', 'sine', 'times', 'vdl']
