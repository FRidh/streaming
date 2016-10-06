import numpy as np
from streaming.itertools import *
from streaming import Stream, BlockStream
import cytoolz


class TestIterable:

    def test_cycle(self):
        nsamples = 10
        stream = range(nsamples)
        cycled_stream = cycle(stream)
        cycled = list(cytoolz.take(nsamples*2, cycled_stream))
        assert np.allclose( cycled[0:nsamples], cycled[nsamples:])

    def test_repeat_each(self):
        nsamples = 10
        nrepeats = 5
        stream = range(nsamples)
        repeated_stream = list(repeat_each(stream, nrepeats))
        # Check whether the elements indeed repeat.
        assert np.allclose(repeated_stream[0::5], repeated_stream[3::5])



class TestStream:

    def test_repeat_each(self):
        nsamples = 10
        nrepeats = 5
        stream = Stream(range(nsamples))
        repeated_stream = repeat_each(stream, nrepeats)
        assert type(repeated_stream) is type(stream)
        repeated_stream = list(repeated_stream)
        # Check whether the elements indeed repeat.
        assert np.allclose(repeated_stream[0::5], repeated_stream[3::5])


class TestBlockStream:

    def test_repeat_each(self):
        nsamples = 10
        nrepeats = 5
        nblock = 4
        stream = Stream(range(nsamples)).blocks(nblock)
        repeated_stream = repeat_each(stream, nrepeats)
        assert type(repeated_stream) is type(stream)
        repeated_stream = list(repeated_stream.samples())
        # Check whether the elements indeed repeat.
        assert np.allclose(repeated_stream[0::5], repeated_stream[3::5])
