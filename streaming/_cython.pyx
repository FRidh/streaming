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
