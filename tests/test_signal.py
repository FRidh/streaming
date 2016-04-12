import itertools
import numpy as np
import pytest
import streaming
from streaming.stream import Stream, BlockStream
from streaming.signal import *

@pytest.fixture(params=[128, 1024])
def nblock(request):
    return request.param

#@pytest.fixture(params=['Stream', 'BlockStream'])
#def classname(request):
    #return request.param

@pytest.fixture(params=[True, False])
def blocked(request):
    return request.param

@pytest.fixture
def samples():
    return 5000

@pytest.fixture
def signal_array(samples):
    return np.arange(samples)

@pytest.fixture
def signal(signal_array, blocked, nblock):
    signal = Stream(signal_array)
    if blocked:
        signal = signal.blocks(nblock)
    return signal

@pytest.fixture(params=[True, False])
def time_variant(request):
    return request.param

@pytest.fixture
def impulse_responses(samples, nblock, time_variant):
    taps = 64
    if time_variant: # Unique IR for each block
        nblocks = samples // nblock
        return Stream(iter(np.random.randn(nblocks, taps)))
    else: # Same IR for each block
        return Stream(itertools.cycle([np.random.randn(taps)]))
    return request.param


def test_convolve(signal, impulse_responses, nblock, samples):
    out = convolve(signal, impulse_responses, nblock)
    assert isinstance(out, BlockStream)
    out_array = out.toarray()
    assert len(out_array) == samples // nblock * nblock


@pytest.fixture
def times(samples):
    return Stream(range(samples))

@pytest.fixture
def delay(samples):
    return Stream(np.ones(samples) * 0.1 + np.arange(samples) * 0.001)

def test_vdl(signal, times, delay):

    delayed = streaming.signal.vdl(signal, times, delay)
    obtained = delayed.toarray()

@pytest.fixture(params=[None, 128, 1024])
def nblock_noise(request):
    return request.param

def test_noise(nblock_noise):
    nblock = nblock_noise


    seed = 100
    state = np.random.RandomState(seed=seed)

    if nblock is None:
        nsamples = 1000
    else:
        nsamples = nblock * 4

    stream = noise(nblock, state)
    out = stream.samples().take(nsamples).toarray()

    out_ref = state = np.random.RandomState(seed=seed).randn(nsamples)

    assert np.allclose(out, out_ref)
