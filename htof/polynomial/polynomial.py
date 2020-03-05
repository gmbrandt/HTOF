import numpy as np
from numpy.polynomial.polynomial import Polynomial
from scipy.special import factorial


def taylorvander(x, deg):
    """
    Taylor series analog for the Vandermonde matrix of given degree.
    Returns the Vandermonde matrix of degree `deg` and sample points
    `x`. The Vandermonde matrix is defined by
    .. math:: V[..., i] = x^i / i!,
    where `0 <= i <= deg`. The leading indices of `V` index the elements of
    `x` and the last index is the power of `x`.
    Parameters
    ----------
    x : array_like
        Array of points. The dtype is converted to float64 or complex128
        depending on whether any of the elements are complex. If `x` is
        scalar it is converted to a 1-D array.
    deg : int
        Degree of the resulting matrix.
    Returns
    -------
    vander : ndarray.
        The Vandermonde matrix. The shape of the returned matrix is
        ``x.shape + (deg + 1,)``, where the last index is the power of `x`.
        The dtype will be the same as the converted `x`.
        See Also
        --------
        numpy.polynomial.polynomial.polyvander
    Examples
    --------
    >>> x = np.array([1, 2, 3, 5])
    >>> deg = 2
    >>> taylorvander(x, deg)
    array([[ 1.,  1.,  0.5],
           [ 1.,  2.,   2.],
           [ 1.,  3.,  4.5],
           [ 1.,  5., 12.5]])

    """
    return np.polynomial.polynomial.polyvander(x, deg) / factorial(np.arange(deg + 1))


class TaylorSeries(Polynomial):
    def __init__(self, coef, domain=None, window=None):
        mp = factorial(np.arange(len(coef)))
        super(TaylorSeries, self).__init__(coef / mp, domain=domain, window=window)
