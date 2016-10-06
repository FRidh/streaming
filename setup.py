import sys
from setuptools import setup
from setuptools.command.test import test as TestCommand

import numpy as np
from Cython.Build import cythonize

class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest
        errno = pytest.main(self.test_args)
        sys.exit(errno)

CLASSIFIERS = [
    'License :: OSI Approved :: BSD License',
    'Programming Language :: Cython',
    'Programming Language :: Python :: 3 :: Only',
    'Programming Language :: Python :: 3.5',
    'Topic :: Scientific/Engineering',
    ]

setup(
    name='streaming',
    version='0.1',
    description="Streaming data with Python",
    author='Frederik Rietdijk',
    author_email='freddyrietdijk@fridh.nl',
    license='LICENSE',
    packages=['streaming'],
    zip_safe=False,
    install_requires=[
        'cython',
        'cytoolz',
        'multipledispatch',
        'numpy',
        'noisy',
        ],
    classifiers=CLASSIFIERS,
    tests_require = [ 'pytest', 'scipy' ],
    cmdclass = {'test': PyTest},
    ext_modules=cythonize('streaming/*.pyx'),
    include_dirs=[np.get_include()]
    )
