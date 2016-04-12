cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def _interpolate_linear(object x, object y, object xnew):
    """Interpolate `y` for `x` at new positions `xnew`.
    """
    cdef double xl, yl, xr, yr, xnewi, ynewi

    xl = 0.0#next(x)
    yl = 0.0#next(y)
    xr = next(x)
    yr = next(y)

    # Determine for each new point
    for xnewi in xnew:
        # The x and y to the left and to the right of it
        # by stepping through x and y one by one
        while xr <= xnewi:
            xl = xr
            yl = yr
            xr = next(x)
            yr = next(y)
        ynewi = (yr-yl)/(xr-xl) * (xnewi-xl) + yl
        yield ynewi


@cython.boundscheck(False)
@cython.cdivision(False) # We rely on Python division because we use negative indices.
@cython.wraparound(False)
def _filter_ba(object x, double[:] b, double[:] a, double[:] xd, double[:] yd, int nb, int na):
    """
    :param b: Numerator coefficients.
    :param a: Denominator coefficients.
    :param x: Signal.
    """
    cdef int xi, yi, i
    cdef double result, A, B

    a = a[::-1]
    b = b[::-1]

    xi = 0
    yi = 0

    while True:
        #print("Input buffer: {}".format(xd))
        #print("Output buffer: {}".format(yd))

        xd[xi] = next(x)

        result = 0.0
        A = 0.0
        B = 0.0
        # Summation b coefficients
        for i in range(nb):
            B += b[i] * xd[(xi-i)%nb]

        # Summation a coefficients
        for i in range(na):
            A -= a[i] * yd[(yi+i)%na]

        result = A + B
        #print("A: {}, B:{}".format(A, B))
        # Final result
        yd[yi] = result
        yield result

        xi = (xi+1)%nb
        yi = (yi+1)%na


# Cython version of diff is 3x as fast
def diff(object iterator, double initial_value=0.0):
    """Differentiate `iterator`.
    """
    cdef double current, old
    current = next(iterator)
    while True:
        old = current
        current = next(iterator)
        yield current-old



