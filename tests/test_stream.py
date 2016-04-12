"""Test streams.
"""
import pytest
import multipledispatch
from multipledispatch.conflict import AmbiguityWarning
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=AmbiguityWarning)
#import warnings
#with warnings.catch_warnings():
    #warnings.filterwarnings("ignore", category=AmbiguityWarning)
import streaming
from streaming._iterator import blocks
from streaming.stream import *
from streaming.itertools import *
from streaming.operators import _BINARY_OPERATORS

import operator
import numbers

_BINARY_OPERATORS.remove('matmul') # Don't test this one

@pytest.fixture(params=_BINARY_OPERATORS)
def binary_operator(request):
    return request.param



class _TestOperators(object):
    """Test Operators metaclass.
    """

    @pytest.fixture(params=[0, 0.0, 2, 2.0, np.ones(5), np.zeros(5), np.zeros(128)])
    def other(self, request):
        return request.param




@dispatch(object, object)
def _incompatible_shape(a, b):
    return False

@dispatch(BlockStream, np.ndarray)
def _incompatible_shape(a, b):
    print("wrong_shape func")
    if a.nblock != len(b):
        return True
    else:
        return False

@dispatch(np.ndarray, BlockStream)
def _incompatible_shape(a, b):
    print("wrong_shape func")
    if b.nblock != len(a):
        return True
    else:
        return False

class _TestAbstractStream(_TestOperators):
    """Tests that should be performed for each subclass of AbstractStream.
    """

    def test_binary_operators(self, binary_operator, stream, other):
        # Operator to use
        op = getattr(operator, binary_operator)

        try:
            # Exception for eq/ne operators, because of ambigious behaviour numpy
            if _incompatible_shape(stream, other) and op not in [operator.eq, operator.ne]:
                with pytest.raises(ValueError):
                    list(op(stream, other))
            else:
                list(op(stream, other))
        except ZeroDivisionError:
            pass

    def test_reverse_binary_operators(self, binary_operator, other, stream):
        # Operator to use
        op = getattr(operator, binary_operator)

        try:
            if _incompatible_shape(other, stream) and op not in [operator.eq, operator.ne]:
                with pytest.raises(ValueError):
                    list(op(other, stream))
            else:
                list(op(other, stream))
        except ZeroDivisionError:
            pass

    @pytest.fixture
    def nsamples(self):
        return 1000

    @pytest.fixture
    def nblock(self):
        return 128

    @pytest.fixture
    def array(self, nsamples):
        return np.arange(nsamples)

    def test_blocks(self, stream):
        out = stream.blocks(10)
        assert isinstance(out, BlockStream)
        list(out)

    def test_peek(self, stream):
        first = stream.peek()
        assert first == list(stream)[0] # All in case of blockstream

    def test_samples(self, stream):
        out = stream.samples()
        assert isinstance(out, Stream)
        list(out)


class TestStream(_TestAbstractStream):
    """Tests specific to Stream.
    """

    @pytest.fixture
    def stream(self, array, nsamples):
        return Stream(array)
        #return Stream(range(nsamples))

    def test_constructor(self, nsamples):
        dtype = 'float64'
        array = np.arange(nsamples, dtype=dtype)
        stream = Stream(array)
        out = np.fromiter(stream, dtype=dtype)
        assert(np.all(out==array))

    def test_copy(self, stream):
        c = stream.copy()
        a = np.array(list(c))
        b = np.array(list(stream))
        assert type(a) is type(b)
        assert np.allclose(a, b)

    def test_cycle(self, stream, nsamples):
        out = stream.cycle().take(nsamples*2).toarray()
        assert np.allclose(out[0:nsamples], out[nsamples:])


class TestBlockStream(_TestAbstractStream):
    """Tests specific to BlockStream.
    """

    @pytest.fixture
    def stream(self, array, nblock):
        return BlockStream(map(np.array, blocks(array, nblock)), nblock)

    @pytest.fixture(params=['mean', 'std', 'var'])
    def reduction(self, request):
        return request.param

    def test_constructor(self, array, stream, nsamples, nblock):
        dtype = 'float64'
        assert isinstance(stream, BlockStream)
        assert stream.nblock == nblock
        assert np.all( stream.toarray() == array[:nsamples//nblock*nblock] )

    def test_copy(self, stream):
        c = stream.copy()
        assert c.nblock == stream.nblock
        a = np.array(list(c))
        b = np.array(list(stream))
        assert type(a) is type(b)
        assert np.allclose(a, b)

    def test_cycle(self, stream, nsamples, nblock):
        nsamples = nsamples // nblock * nblock
        out = stream.cycle().samples().take(nsamples*2).toarray()
        assert np.allclose(out[0:nsamples], out[nsamples:])

    def test_peek(self, stream):
        first = stream.peek()
        print(list(stream.copy())[0])
        assert np.allclose(first, list(stream)[0]) # First block

    def test_reductions(self, array, nblock, nsamples, stream, reduction):
        """Test reductions like `mean` and `std`.
        """
        nblocks = nsamples//nblock
        nsamples = nblocks*nblock
        arr = array[:nsamples].reshape(nblocks, -1)
        expected = getattr(arr, reduction)(axis=-1)
        obtained = getattr(stream.blocks(nblock), reduction)().toarray()
        assert np.allclose(expected, obtained)



class TestItertools(object):

    @pytest.fixture(params=[range(100), Stream(range(100)), Stream(range(100)).blocks(nblock=10)])
    def stream(self, request):
        return request.param

    def test_tee(self, stream):
        n = 3
        streams = tee(stream, n=n)
        assert len(streams)==n

        if isinstance(stream, AbstractStream):
            for s in streams:
                assert(type(s) == type(stream))




