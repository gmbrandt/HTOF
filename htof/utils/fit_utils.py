"""
Module for generating the chi-squared matrix (and vectors) for the 9 parameter fit to the epoch astrometry.
"""

import numpy as np

FIT_BASIS = np.polynomial.polynomial.polyvander


def _evaluate_basis_functions(w_ra, w_dec, ra_t, dec_t, basis, deg):
    f = np.hstack([[w_ra], np.zeros(2*deg + 2)])
    g = np.hstack([[w_dec], np.zeros(2*deg + 2)])
    f[1:][::2], g[1:][1::2] = basis(ra_t, deg), basis(dec_t, deg)
    return f, g


def ra_sol_vec(a, b, c, d, ra_t, dec_t, w_ra=0, w_dec=0, basis=FIT_BASIS, deg=3):
    """
    :params floats a,b,c,d: components of the inverse covariance matrix for the observation.
            i.e. the ivar matrix should be np.array([[a, b],[c, d]])
    :param ra_t: delta t for right ascension
    :param dec_t: delta t declination
    :param w_ra: pertubation from parallax alone for right ascension
    :param w_dec: pertubation from parallax alone for declination
    :param deg: degree for the fit in ra and dec.
    :param basis: method such that basis(t, degree) returns an array of shape (1, degree+1) with the basis
    functions evaluated at the time t. E.g. if basis= np.polynomial.polynomial.polyvander then basis(5, 3) returns
                  array([[  1.,   5.,  25., 125.]])
    This return is typically called the Vandermonde matrix. E.g. for Legendre polynomial basis we would feed
    basis=np.polynomial.legendre.legvander
    :return:
    """
    f, g = _evaluate_basis_functions(w_ra, w_dec, ra_t, dec_t, basis=basis, deg=deg)
    ra_vec = np.array([2*a*f[i] + (b+c)*g[i] for i in range(2*deg + 3)], dtype=float)
    return ra_vec


def dec_sol_vec(a, b, c, d, ra_t, dec_t, w_ra=0, w_dec=0, basis=FIT_BASIS, deg=3):
    """
    :params floats a,b,c,d: components of the inverse covariance matrix for the observation.
            i.e. the ivar matrix should be np.array([[a, b],[c, d]])
    :param ra_t: delta t for right ascension
    :param dec_t: delta t declination
    :param w_ra: pertubation from parallax alone for right ascension
    :param w_dec: pertubation from parallax alone for declination
    :param deg: degree for the fit in ra and dec.
    :param basis: method such that basis(t, degree) returns an array of shape (1, degree+1) with the basis
    functions evaluated at the time t. E.g. if basis= np.polynomial.polynomial.polyvander then basis(5, 3) returns
                  array([[  1.,   5.,  25., 125.]])
    This return is typically called the Vandermonde matrix. E.g. for Legendre polynomial basis we would feed
    basis=np.polynomial.legendre.legvander
    :return:
    """
    f, g = _evaluate_basis_functions(w_ra, w_dec, ra_t, dec_t, basis=basis, deg=deg)
    dec_vec = np.array([(b+c)*f[i] + 2*d*g[i] for i in range(2*deg + 3)], dtype=float)
    return dec_vec


def chi2_matrix(a, b, c, d, ra_t, dec_t, w_ra=0, w_dec=0, basis=FIT_BASIS, deg=3):
    """
    :params floats a,b,c,d: components of the inverse covariance matrix for the observation.
            i.e. the ivar matrix should be np.array([[a, b],[c, d]])
    :param ra_t: delta t for right ascension
    :param dec_t: delta t declination
    :param w_ra: pertubation from parallax alone for right ascension
    :param w_dec: pertubation from parallax alone for declination
    :param deg: degree for the fit in ra and dec.
    :param basis: method such that basis(t, degree) returns an array of shape (1, degree+1) with the basis
    functions evaluated at the time t. E.g. if basis= np.polynomial.polynomial.polyvander then basis(5, 3) returns
                  array([[  1.,   5.,  25., 125.]])
    This return is typically called the Vandermonde matrix. E.g. for Legendre polynomial basis we would feed
    basis=np.polynomial.legendre.legvander
    :return:
    """

    A = np.zeros((2*deg + 3, 2*deg + 3), dtype=np.float32)
    f, g = _evaluate_basis_functions(w_ra, w_dec, ra_t, dec_t, basis=basis, deg=deg)
    for k in range(A.shape[0]):
        A[k] = [2 * f[i] * a * f[k] + g[i] * (b+c) * f[k] + 
                f[i] * (b+c) * g[k] + 2 * g[i] * d * g[k] for i in range(2*deg + 3)]
    return np.array(A, dtype=float)


def transform_coefficients_to_unnormalized_domain(coeffs, ra_min_t, ra_max_t, dec_min_t, dec_max_t,
                                                  deg, use_parallax):
    # refactor using Polynomial.convert or Polynomial.cast:
    # http://omz-software.com/pythonista/numpy/reference/generated/numpy.polynomial.polynomial.Polynomial.html
    temp_coeffs = coeffs[1 * use_parallax:]
    fa, ga, fb, gb, fc, gc, fd, gd = np.pad(temp_coeffs, (0, 2 * 3 + 2 - len(temp_coeffs)),
                                            mode='constant', constant_values=0)
    coeffs[1 * use_parallax:][::2] = _transform(fa, fb, fc, fd, ra_min_t, ra_max_t)[:deg + 1]
    coeffs[1 * use_parallax:][1::2] = _transform(ga, gb, gc, gd, dec_min_t, dec_max_t)[:deg + 1]
    return coeffs


def _transform(ap, bp, cp, dp, minx, maxx):
    j, h = 2/(maxx - minx), -2*minx/(maxx - minx) - 1
    a = ap + bp*h + cp*h**2 + dp*h**3
    b = bp*j + 2*cp*h*j + 3*dp*h**2*j
    c = cp*j**2 + 3*dp*h*j**2
    d = dp*j**3
    return a, b, c, d
