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

@pytest.fixture(params=[5000, 8000])
def nsamples(request):
    return request.param

@pytest.fixture
def sequence(nsamples):
    return np.arange(nsamples)

@pytest.fixture
def stream(sequence, blocked, nblock):
    stream = Stream(sequence)
    if blocked:
        stream = stream.blocks(nblock)
    return stream

@pytest.fixture(params=[True, False])
def time_variant(request):
    return request.param

@pytest.fixture(params=[10, 32, 64])
def ntaps(request):
    return request.param

@pytest.fixture
def impulse_responses(nsamples, ntaps, nblock, time_variant):
    if time_variant: # Unique IR for each block
        nblocks = nsamples // nblock
        return Stream(iter(np.random.randn(nblocks, ntaps)))
    else: # Same IR for each block
        return Stream(itertools.cycle([np.random.randn(ntaps)]))

@pytest.fixture
def impulse_response(ntaps):
    return np.random.randn(ntaps)


def test_convolve_overlap_add_invariant(sequence, stream, impulse_response, nsamples, nblock, ntaps):
    nhop = nblock
    # Compute convolutions
    obtained = convolve_overlap_add(stream, constant(impulse_response), nhop, ntaps).toarray()
    reference = np.convolve(sequence, impulse_response, mode='valid')

    reference_length = nsamples - ntaps + 1
    obtained_length = nsamples // nblock * nblock
    length = min(reference_length, obtained_length)

    #print(len(obtained), len(reference), obtained_length, reference_length)
    assert len(obtained) == obtained_length
    assert len(reference) == reference_length
    assert np.allclose(obtained[ntaps:length-ntaps], reference[ntaps:length-ntaps])


def test_convolve_overlap_save_invariant(sequence, stream, impulse_response, nsamples, nblock, ntaps):
    """Test :func:`convolve_overlap_save` in the case of a time-invariant filter.
    """
    nhop = nblock
    nwindow = nhop + ntaps - 1
    obtained = convolve_overlap_save(stream, constant(impulse_response), nhop, ntaps).toarray()
    reference = np.convolve(sequence, impulse_response, mode='valid')

    # Length of obtained is always a bit smaller. Let's compute actual length.
    reference_length = nsamples - ntaps + 1
    obtained_length = (nsamples // nhop * nhop )# // nblock * nblock) // nhop * nhop

    obtained_length = ((nsamples // nblock * nblock) - ntaps + 1) // nhop * nhop

    length = min(reference_length, obtained_length)
    print(reference_length, obtained_length, len(obtained), len(reference))
    assert len(obtained) == obtained_length
    assert len(reference) == reference_length
    assert np.allclose(obtained[ntaps:length-ntaps], reference[ntaps:length-ntaps])

@pytest.fixture
def times(nsamples):
    return Stream(range(nsamples))

@pytest.fixture
def delay(nsamples):
    return Stream(np.ones(nsamples) * 0.1 + np.arange(nsamples) * 0.001)

#def test_vdl(stream, times, delay):

    #delayed = streaming.signal.vdl(stream, times, delay)
    #obtained = delayed.toarray()

#@pytest.fixture(params=[None, 128, 1024])
#def nblock_noise(request):
    #return request.param

#def test_noise(nblock_noise):
    #nblock = nblock_noise


    #seed = 100
    #state = np.random.RandomState(seed=seed)

    #if nblock is None:
        #nsamples = 1000
    #else:
        #nsamples = nblock * 4

    #stream = noise(nblock, state)
    #out = stream.nsamples().take(nsamples).toarray()

    #out_ref = state = np.random.RandomState(seed=seed).randn(nnsamples)

    #assert np.allclose(out, out_ref)
