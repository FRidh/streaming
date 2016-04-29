import itertools
import scipy.signal
import numpy as np
import pytest
from streaming._iterator import *

#class TestConvolve(object):
    #"""Test `convolve`.
    #"""

    #def test_convolve_lti(self):
        #"""Test with a linear time-invariant system.
        #"""
        #nblock = 8
        #ntaps = 3
        #blocks = 5

        #x = np.ones(blocks*nblock)
        #ir = itertools.cycle( [np.ones(ntaps)] )
        #result = np.fromiter(convolve(iter(x), ir, nblock), dtype='float64')

        #assert len(result) == nblock*blocks

        ## Convolution of a signal consisting of ones and a filter of 3 ones, should result in all threes.
        ## At least, after the filter has fully kicked in.
        #assert np.allclose(result[3-1:], 3.)


    #def test_convolve_ltv(self):
        #"""Test with a linear-time-invariant system.

        #The test is the same as in `test_signal.test_convolve_ltv`.
        #The result however does not include the tail of the convolution!
        #"""
        #x = np.array([1, 1, 1, 1, 1, 1])
        #ir = np.array([
            #[1, 2],
            #[0, 1],
            #[0, 1],
            #])

        #nblock = 2

        #result = np.fromiter(convolve(iter(x), iter(ir), nblock), dtype='float64')
        #expected_result = np.array([1,2,1])
        #np.testing.assert_array_equal(result, expected_result)


class TestInterpolate(object):

    @pytest.fixture
    def interpolator(self):
        return interpolate_linear

    def test_interpolate_linear(self, interpolator):

        samples = 10
        x = np.arange(samples)
        y = np.random.randn(samples)
        #y = np.arange(samples)*5
        xnew = np.arange(samples)*0.3
        xnew = xnew[xnew <= x.max()]

        expected = np.interp(xnew, x, y)

        obtained = np.array(list(interpolator(iter(x), iter(y), iter(xnew))))

        assert np.allclose(obtained, expected)


def test_vdl():

    samples = 10

    signal = np.ones(samples)
    times = np.arange(samples)
    # Constant delay of 3 samples
    delay = np.ones(samples) * 3

    delayed = list(vdl(iter(signal), iter(times), iter(delay)))

    head = delayed[0:3]
    tail = delayed[3:]

    # Signal begins with 3 zeros,
    assert np.allclose(head, 0)
    # and then we get 7 samples
    assert len(tail)==7
    # which are all ones.
    assert np.allclose(tail, 1)



def test_filter_ba():

    fs = 44100.
    duration = 1.0
    nsamples = int(fs*duration)
    order = 3
    cutoff = 4000.

    x = np.random.randn(nsamples)

    b, a = scipy.signal.butter(order, cutoff/(fs/2.0), btype='low', output='ba')

    y_ref = scipy.signal.lfilter(b, a, x)
    y = np.array(list(filter_ba(iter(x), b, a)))

    assert np.abs(y_ref-y).mean() < 1e-12


def test_filter_sos():

    fs = 44100.
    duration = 1.0
    nsamples = int(fs*duration)
    order = 3
    cutoff = 4000.

    x = np.random.randn(nsamples)

    sos = scipy.signal.butter(order, cutoff/(fs/2.0), btype='low', output='sos')

    y_ref = scipy.signal.sosfilt(sos, x)
    y = np.array(list(filter_sos(iter(x), sos)))

    assert np.abs(y_ref-y).mean() < 1e-12


class TestConvolveOverlapDiscard(object):

    def test_samples_nblock_in(self):
        """Input blocksize is given."""
        pass

    def test_samples_nblock_out(self):
        """Output blocksize is given."""
        pass

    def test_samples_nblock_in_nblock_out(self):
        """Both input and output blocksize are given."""
        pass

    def test_samples(self):
        """Both input and output blocksize are NOT given."""
        signal = np.random.randn(100)
        ir = np.random.randn(5)
        pass

    #def test_blocks_blocks(self):
        #"""Signal is block-based and an output blocksize is given."""
        #signal = np.random.randn(100)
        #ir = np.random.randn(5)
        #nblock_out = 4
        #nblock_in = 7

        ##nblocks_in = len(signal) // nblock_in
        ##nsamples_in = nblocks_in * nblock_in
        ##nblocks_out = nsamples_in // nblock_out
        ##nsamples_out = nblocks_out * nblock_out
        ##print(nsamples_out)
        #nsamples_out = 92 # FIXME

        #out = np.array(list(itertools.chain.from_iterable(convolve_overlap_discard(blocks(iter(signal), nblock_in), ir, nblock_in=nblock_in, nblock_out=nblock_out))))
        #reference = np.convolve(signal, ir, mode='valid')[:nsamples_out]
        #assert np.allclose(out, reference)


    #def test_blocks_whatever(self):
        #"""Signal is block-based and an output blocksize is not given."""
        #signal = np.random.randn(100)
        #ir = np.random.randn(5)
        #nblock_in = 7

        #nsamples_out = 93 #FIXME

        #out = np.array(list(itertools.chain.from_iterable(convolve_overlap_discard(blocks(iter(signal), nblock_in), ir, nblock_in=nblock_in))))
        #reference = np.convolve(signal, ir, mode='valid')[:nsamples_out]
        #assert np.allclose(out, reference)

    #def test_samples_blocks(self):
        #"""Signal is sample-based and an output blocksize is given."""
        #signal = np.random.randn(100)
        #ir = np.random.randn(5)
        #nblock_out = 4

        #nsamples_out = len(signal) - len(ir) + 1
        #nblocks = nsamples_out // nblock_out
        #nsamples_out = nblocks * nblock_out

        #out = np.array(list(itertools.chain.from_iterable(convolve_overlap_discard(iter(signal), ir, nblock_out=nblock_out))))
        #reference = np.convolve(signal, ir, mode='valid')[:nsamples_out]
        #assert np.allclose(out, reference)


    #def test_samples_samples(self):
        #"""Signal is sample-based and output is samples as well."""
        #signal = np.random.randn(100)
        #ir = np.random.randn(5)
        ## That means block_out will be set to 1 in the function
        #nblock_out = 1

        ##nsamples_out = len(signal) - len(ir) + 1
        #out = np.array(list(itertools.chain.from_iterable(convolve_overlap_discard(iter(signal), ir))))
        #reference = np.convolve(signal, ir, mode='valid')
        #assert np.allclose(out, reference)

