"""
Module for generating the chi-squared matrix (and vectors) for the 9 parameter fit to the epoch astrometry.
"""

import numpy as np


def ra_sol_vec(a, b, c, d, ra_t, dec_t, w_ra=0, w_dec=0, basis=np.polynomial.polynomial.polyvander, deg=3):
    """
    :params floats a,b,c,d: components of the inverse covariance matrix for the observation.
            i.e. the ivar matrix should be np.array([[a, b],[c, d]])
    :param ra_t: delta t for right ascension
    :param dec_t: delta t declination
    :param w_ra: pertubation from parallax alone for right ascension
    :param w_dec: pertubation from parallax alone for declination
    :param basis: method such that basis(t, degree) returns an array of shape (1, degree+1) with the basis
    :param deg: degree for the fit in ra and dec.
    functions evaluated at the time t. E.g. the default polynomial basis is such that basis(5, 3) returns
                  array([[  1.,   5.,  25., 125.]])
    This return is typically called the Vandermonde matrix. E.g. for Legendre polynomial basis we would feed
    basis=np.polynomial.legendre.legvander
    :return:
    """
    fk = np.hstack([[w_ra], np.zeros(2*deg + 2)])
    gk = np.hstack([[w_dec], np.zeros(2*deg + 2)])
    # set order of variables to be parallax, ra0, dec0, mu_ra, mu_dec,...
    fk[1:][::2], gk[1:][1::2] = basis(ra_t, deg), basis(dec_t, deg)
    # assemble solution vector.
    ra_vec = np.array([2*a*fk[i] + (b+c)*gk[i] for i in range(2*deg + 3)])
    return ra_vec


def dec_sol_vec(a, b, c, d, ra_t, dec_t, w_ra=0, w_dec=0, basis=np.polynomial.polynomial.polyvander, deg=3):
    """
    :params floats a,b,c,d: components of the inverse covariance matrix for the observation.
            i.e. the ivar matrix should be np.array([[a, b],[c, d]])
    :param ra_t: delta t for right ascension
    :param dec_t: delta t declination
    :param w_ra: pertubation from parallax alone for right ascension
    :param w_dec: pertubation from parallax alone for declination
    :param basis: method such that basis(t, degree) returns an array of shape (1, degree+1) with the basis
    :param deg: degree for the fit in ra and dec.
    functions evaluated at the time t. E.g. the default polynomial basis is such that basis(5, 3) returns
                  array([[  1.,   5.,  25., 125.]])
    This return is typically called the Vandermonde matrix. E.g. for Legendre polynomial basis we would feed
    basis=np.polynomial.legendre.legvander
    :return:
    """
    fk = np.hstack([[w_ra], np.zeros(2*deg + 2)])
    gk = np.hstack([[w_dec], np.zeros(2*deg + 2)])
    # set order of variables to be parallax, ra0, dec0, mu_ra, mu_dec,...
    fk[1:][::2], gk[1:][1::2] = basis(ra_t, deg), basis(dec_t, deg)
    # assemble solution vector.
    dec_vec = [(b+c)*fk[i] + 2*d*gk[i] for i in range(2*deg + 3)]
    return dec_vec


def chi2_matrix(a, b, c, d, ra_t, dec_t, w_ra=0, w_dec=0, basis=np.polynomial.polynomial.polyvander, ra_deg=3, dec_deg=3):
    """
    :params floats a,b,c,d: components of the inverse covariance matrix for the observation.
            i.e. the ivar matrix should be np.array([[a, b],[c, d]])
    :param ra_t: delta t for right ascension
    :param dec_t: delta t declination
    :param w_ra: pertubation from parallax alone for right ascension
    :param w_dec: pertubation from parallax alone for declination
    :param basis: method such that basis(t, degree) returns an array of shape (1, degree+1) with the basis
    functions evaluated at the time t. E.g. the default polynomial basis is such that basis(5, 3) returns
                  array([[  1.,   5.,  25., 125.]])
    This return is typically called the Vandermonde matrix. E.g. for Legendre polynomial basis we would feed
    basis=np.polynomial.legendre.legvander
    :return:
    """
    A = np.zeros((9, 9), dtype=np.float32)
    fk = np.hstack([[w_ra], basis(ra_t, ra_deg)[0], np.zeros(dec_deg + 1)])
    gk = np.hstack([[w_dec], np.zeros(ra_deg + 1), basis(dec_t, dec_deg)[0]])
    return np.array(A)
