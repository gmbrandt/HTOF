"""
Module for generating the chi-squared matrix (and vectors) for the 9 parameter fit to the epoch astrometry.
"""

import numpy as np


def ra_sol_vec(a, b, c, d, ra_t, dec_t, w_ra=0, w_dec=0):
    """
    :params floats a,b,c,d: components of the inverse covariance matrix for the observation.
            i.e. the ivar matrix should be np.array([[a, b],[c, d]])
    :param ra_t: delta t for right ascension
    :param dec_t: delta t declination
    :param w_ra: pertubation from parallax alone for right ascension
    :param w_dec: pertubation from parallax alone for declination
    :return:
    """
    ra_vec = np.zeros(9).astype(np.float64)

    ra_vec[0] = 2 * a
    ra_vec[1] = b + c
    ra_vec[2] = 2 * a * ra_t
    ra_vec[3] = b * dec_t + c * dec_t
    ra_vec[4] = a * ra_t ** 2
    ra_vec[5] = b/2 * dec_t ** 2 + c/2 * dec_t ** 2
    ra_vec[6] = a/3 * ra_t ** 3
    ra_vec[7] = b/6 * dec_t ** 3 + c/6 * dec_t ** 3
    ra_vec[8] = 2 * a * w_ra + b * w_dec + c * w_dec
    return ra_vec


def dec_sol_vec(a, b, c, d, ra_t, dec_t, w_ra=0, w_dec=0):
    """
    :params floats a,b,c,d: components of the inverse covariance matrix for the observation.
            i.e. the ivar matrix should be np.array([[a, b],[c, d]])
    :param ra_t: delta t for right ascension
    :param dec_t: delta t declination
    :param w_ra: pertubation from parallax alone for right ascension
    :param w_dec: pertubation from parallax alone for declination
    :return:
    """
    dec_vec = np.zeros(9).astype(np.float64)

    dec_vec[0] = b + c
    dec_vec[1] = 2 * d
    dec_vec[2] = b * ra_t + c * ra_t
    dec_vec[3] = 2 * d * dec_t
    dec_vec[4] = b/2 * ra_t ** 2 + c/2 * ra_t ** 2
    dec_vec[5] = d * dec_t ** 2
    dec_vec[6] = b/6 * ra_t ** 3 + c/6 * ra_t ** 3
    dec_vec[7] = d/3 * dec_t ** 3
    dec_vec[8] = b * w_ra + c * w_ra + 2 * d * w_dec
    return dec_vec


def chi2_matrix(a, b, c, d, ra_t, dec_t, w_ra=0, w_dec=0):
    """
    :params floats a,b,c,d: components of the inverse covariance matrix for the observation.
            i.e. the ivar matrix should be np.array([[a, b],[c, d]])
    :param dec_t: delta t for right ascension
    :param ra_t: delta t for declination
    :param w_ra: pertubation from parallax alone for right ascension
    :param w_dec: pertubation from parallax alone for declination
    :return:
    """
    A = np.zeros((9, 9))

    A[0] = 0
    A[1] = 0
    A[2] = 0
    A[3] = 0
    A[4] = 0
    A[5] = 0
    A[6] = 0
    A[7] = 0
    A[8] = 0
    return A
