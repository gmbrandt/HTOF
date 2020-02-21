"""
Module for generating the chi-squared matrix (and vectors) for the N parameter fit to the epoch astrometry.
"""

import numpy as np
from htof.polynomial import polynomial

FIT_BASIS = polynomial.TaylorSeries
FIT_VANDER = polynomial.taylorvander
# FIT_VANDER must follow np.polynomial.polynomial.Polynomial and
# np.polynomial.polynomial.polyvander syntax, respectively. For example, one could set:
# FIT_BASIS = np.polynomial.polynomial.Polynomial
# FIT_VANDER = np.polynomial.polynomial.polyvander
# if they wanted to use a polynomial basis without the 1/2, 1/6 etc... prefactors.


def _evaluate_basis_functions(w_ra, w_dec, ra_t, dec_t, vander, deg):
    f = np.hstack([[w_ra], np.zeros(2*deg + 2)])
    g = np.hstack([[w_dec], np.zeros(2*deg + 2)])
    f[1:][::2], g[1:][1::2] = vander(ra_t, deg), vander(dec_t, deg)
    return f, g


def ra_sol_vec(a, b, c, d, ra_t, dec_t, w_ra=0, w_dec=0, vander=FIT_VANDER, deg=3):
    """
    :params floats a,b,c,d: components of the inverse covariance matrix for the observation.
            i.e. the ivar matrix should be np.array([[a, b],[c, d]])
    :param ra_t: delta t for right ascension
    :param dec_t: delta t declination
    :param w_ra: pertubation from parallax alone for right ascension
    :param w_dec: pertubation from parallax alone for declination
    :param deg: degree for the fit in ra and dec.
    :param vander: method such that vander(t, degree) returns an array of shape (1, degree+1) with the basis
    functions evaluated at the time t. E.g. if vander= np.polynomial.polynomial.polyvander then basis(5, 3) returns
                  array([[  1.,   5.,  25., 125.]])
    This return is typically called the Vandermonde matrix. E.g. for Legendre polynomial basis we would feed
    vander=np.polynomial.legendre.legvander
    :return:
    """
    f, g = _evaluate_basis_functions(w_ra, w_dec, ra_t, dec_t, vander=vander, deg=deg)
    ra_vec = np.array([a*f[i] + (b+c)/2*g[i] for i in range(2*deg + 3)], dtype=float)
    return ra_vec


def dec_sol_vec(a, b, c, d, ra_t, dec_t, w_ra=0, w_dec=0, vander=FIT_VANDER, deg=3):
    """
    :params floats a,b,c,d: components of the inverse covariance matrix for the observation.
            i.e. the ivar matrix should be np.array([[a, b],[c, d]])
    :param ra_t: delta t for right ascension
    :param dec_t: delta t declination
    :param w_ra: pertubation from parallax alone for right ascension
    :param w_dec: pertubation from parallax alone for declination
    :param deg: degree for the fit in ra and dec.
    :param vander: method such that vander(t, degree) returns an array of shape (1, degree+1) with the basis
    functions evaluated at the time t. E.g. if vander= np.polynomial.polynomial.polyvander then basis(5, 3) returns
                  array([[  1.,   5.,  25., 125.]])
    This return is typically called the Vandermonde matrix. E.g. for Legendre polynomial basis we would feed
    vander=np.polynomial.legendre.legvander
    :return:
    """
    f, g = _evaluate_basis_functions(w_ra, w_dec, ra_t, dec_t, vander=vander, deg=deg)
    dec_vec = np.array([(b+c)/2*f[i] + d*g[i] for i in range(2*deg + 3)], dtype=float)
    return dec_vec


def chi2_matrix(a, b, c, d, ra_t, dec_t, w_ra=0, w_dec=0, vander=FIT_VANDER, deg=3):
    """
    :params floats a,b,c,d: components of the inverse covariance matrix for the observation.
            i.e. the ivar matrix should be np.array([[a, b],[c, d]])
    :param ra_t: delta t for right ascension
    :param dec_t: delta t declination
    :param w_ra: pertubation from parallax alone for right ascension
    :param w_dec: pertubation from parallax alone for declination
    :param deg: degree for the fit in ra and dec.
    :param vander: method such that vander(t, degree) returns an array of shape (1, degree+1) with the basis
    functions evaluated at the time t. E.g. if vander= np.polynomial.polynomial.polyvander then basis(5, 3) returns
                  array([[  1.,   5.,  25., 125.]])
    This return is typically called the Vandermonde matrix. E.g. for Legendre polynomial basis we would feed
    vander=np.polynomial.legendre.legvander
    :return:
    """

    A = np.zeros((2*deg + 3, 2*deg + 3), dtype=np.float32)
    f, g = _evaluate_basis_functions(w_ra, w_dec, ra_t, dec_t, vander=vander, deg=deg)
    # note that b = c for any realistic covariance (or inverse covariance) matrix.
    for k in range(A.shape[0]):
        A[k] = [f[i] * a * f[k] + g[i] * (b+c)/2 * f[k] +
                f[i] * (b+c)/2 * g[k] + g[i] * d * g[k] for i in range(2*deg + 3)]
    return np.array(A, dtype=float)


def transform_coefficients_to_unnormalized_domain(coeffs, ra_min_t, ra_max_t, dec_min_t, dec_max_t,
                                                  use_parallax, old_domain=None, basis=FIT_BASIS):
    # Using basis= taylor does not convert from normalized to unnormalized properly.
    # TODO fix htof.polynomial.polynomial.TaylorSeries so that its .convert() method works properly
    basis = np.polynomial.polynomial.Polynomial
    if old_domain is None:
        old_domain = [-1, 1]
    padded = np.pad(coeffs[1 * use_parallax:], (0, 2 * 3 + 2 - len(coeffs[1 * use_parallax:])),
                    mode='constant', constant_values=0)
    ra_coeffs, dec_coeffs = padded[::2], padded[1::2]
    ra_poly = basis(ra_coeffs, domain=[ra_min_t, ra_max_t], window=[-1, 1])
    dec_poly = basis(dec_coeffs, domain=[dec_min_t, dec_max_t], window=[-1, 1])

    new_coeffs = np.copy(coeffs).astype(np.float64)
    new_coeffs[1 * use_parallax::2] = ra_poly.convert(domain=old_domain).coef
    new_coeffs[1 * use_parallax + 1::2] = dec_poly.convert(domain=old_domain).coef
    # probably need to return the scl of the .mapparms here in order to calculate the unnormed covariance matrix.
    return new_coeffs


def chisq_of_fit(coeffs, ra, dec, ra_epochs, dec_epochs, inv_covs, ra_plx=None,
                 dec_plx=None, use_parallax=True, basis=FIT_BASIS):
    ra_model = basis(coeffs[1 * use_parallax:][::2])(ra_epochs)
    dec_model = basis(coeffs[1 * use_parallax:][1::2])(dec_epochs)
    if use_parallax:
        ra_model += coeffs[0] * ra_plx
        dec_model += coeffs[0] * dec_plx

    modelminusdata = np.hstack([(ra_model - ra).reshape(-1, 1), (dec_model - dec).reshape(-1, 1)])
    chisquared = 0
    for i in range(len(ra_model)):
        chisquared += np.matmul(np.matmul(modelminusdata[i], inv_covs[i]), modelminusdata[i])
    return chisquared