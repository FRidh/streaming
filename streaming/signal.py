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
from scipy.signal import firwin

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
    return BlockStream(streaming._iterator.blocked_convolve(signal, impulse_responses, nblock=nblock, ntaps=ntaps, initial_values=initial_values), nblock=nblock)


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


def bandpass_filter(lowcut, highcut, fs, ntaps):
    """Design bandpass FIR filter.
    """
    return firwin(ntaps, [lowcut, highcut], pass_zero=False, nyq=fs)

def bandstop_filter(lowcut, highcut, fs, ntaps):
    """Design bandstop FIR filter.
    """
    return firwin(ntaps, [lowcut, highcut], pass_zero=True, nyq=fs)

def lowpass_filter(cut, fs, ntaps):
    """Design lowpass FIR filter.
    """
    return firwin(ntaps, cut, pass_zero=True, nyq=fs)

def highpass_filter(cut, fs, ntaps):
    """Design highpass FIR filter.
    """
    return firwin(ntaps, cut, pass_zero=False, nyq=fs)


def diff(x):
    """Differentiate `x`.

    :returns: Differentiated `x`.
    :rtype: :class:`Stream`

    .. seealso:: :func:`streaming._iterator.diff`
    """
    return Stream(streaming._iterator.diff(x.samples()._iterator))



#class Filterbank(object):

    #def __init__(self, frequencies):
        #pass


#__all__ = ['constant', 'convolve', 'interpolate', 'sine', 'times', 'vdl']
