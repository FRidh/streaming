# streaming

[![Build Status](https://travis-ci.org/FRidh/streaming.svg?branch=master)](https://travis-ci.org/FRidh/streaming)

`streaming` is a Python library for working with streams of data.
Streams are in this case iterables that have operators defined.

    In [1]: s = Stream(range(20))
    In [2]: list(s + 10)
    Out[2]: [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

Operations can be done sample by sample by using an instance of `Stream`, or on blocks of samples by using an instance of `BlockStream`.
Switching from one to the other is easy,

    In [3]: s = Stream(range(100000))
    Out[3]: <streaming.stream.Stream at 0x7fc479462518>
    In [4]: s.blocks(8192)
    Out[4]: <streaming.stream.BlockStream at 0x7fc479462908>

This library was written for a signal processing tool.


## Installation

To install, use

`pip install`

or if you want to have an editable install

`pip install -e`

## Tests

The test suite can be run with

`python setup.py test`

or

`py.test`


## Documentation

## License

The BSD 2-Clause License applies to the code.
