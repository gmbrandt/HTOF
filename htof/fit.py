"""
Module for fitting astrometric data.

Author: G. Mirek Brandt
"""

import numpy as np
from htof.utils.fit_utils import ra_sol_vec, dec_sol_vec, chi2_matrix, transform_coefficients_to_unnormalized_domain
from htof.utils.fit_utils import chisq_of_fit


class AstrometricFitter(object):
    """
    :param inverse_covariance_matrices: ndarray of length epoch times with the 2x2 inverse covariance matrices
                                        for each epoch
    :param epoch_times: 1D array
                        array with each epoch in Barycentric Julian Date (BJD).
    :param parallactic_pertubations: list
    :param parameters: int.
                       number of parameters in the fit. Options are 4, 5, 7, and 9.
                       4 is just offset and proper motion, 5 includes parallax, 7 and 9 include accelerations and jerks.
    The pertubations due to parallactic motion alone with unit parallax. Where parallactic_pertubations[0], and
    parallactic_pertubations[1] are the pertubations for right ascension and declination respectively.
    For each component this should be the quantity which is linear in parallax angle, i.e.:
    Parallax_motion_ra - central_ra.
    The units of this parallactic motion should be exactly the same as the ra's and dec's which you will fit
    later on.
    """
    def __init__(self, inverse_covariance_matrices=None, epoch_times=None,
                 astrometric_chi_squared_matrices=None, astrometric_solution_vector_components=None,
                 parallactic_pertubations=None, fit_degree=1, use_parallax=False,
                 central_epoch_ra=0, central_epoch_dec=0):
        if parallactic_pertubations is None:
            parallactic_pertubations = [np.zeros_like(epoch_times), np.zeros_like(epoch_times)]
        self.use_parallax = use_parallax
        self.parallactic_pertubations = parallactic_pertubations
        self.inverse_covariance_matrices = inverse_covariance_matrices
        self.epoch_times = epoch_times
        self.fit_degree = fit_degree
        self.central_epoch_dec = central_epoch_dec
        self.central_epoch_ra = central_epoch_ra

        if astrometric_solution_vector_components is None:
            self.astrometric_solution_vector_components = self._init_astrometric_solution_vectors(fit_degree)
        if astrometric_chi_squared_matrices is None:
            self._chi2_matrix = self._init_astrometric_chi_squared_matrix(fit_degree)

    def fit_line(self, ra_vs_epoch, dec_vs_epoch, return_all=False):
        """
        :param ra_vs_epoch: 1d array of right ascension, ordered the same as the covariance matrices and epochs.
        :param dec_vs_epoch: 1d array of declination, ordered the same as the covariance matrices and epochs.
        :param return_all: bool. True to return the solution vector as well as the 1-sigma error estimates on the parameters.
        :return: Array:
                 [ra0, dec0, mu_ra, mu_dec]
        """
        solution = np.linalg.solve(self._chi2_matrix, self._chi2_vector(ra_vs_epoch=ra_vs_epoch, dec_vs_epoch=dec_vs_epoch))
        errors = np.sqrt(np.diagonal(np.linalg.pinv(self._chi2_matrix)))

        # transforming out of normalized coordinates.
        c_ra, c_dec = self.central_epoch_ra, self.central_epoch_dec
        t = self.epoch_times
        solution = transform_coefficients_to_unnormalized_domain(solution, t.min() - c_ra, t.max() - c_ra,
                                                                 t.min() - c_dec, t.max() - c_dec, self.use_parallax)
        errors = transform_coefficients_to_unnormalized_domain(errors, t.min() - c_ra, t.max() - c_ra,
                                                               t.min() - c_dec, t.max() - c_dec, self.use_parallax)

        chisq = chisq_of_fit(solution, ra_vs_epoch, dec_vs_epoch,
                             self.epoch_times - self.central_epoch_ra, self.epoch_times - self.central_epoch_dec,
                             self.inverse_covariance_matrices, *self.parallactic_pertubations,
                             use_parallax=self.use_parallax)

        return solution if not return_all else (solution, errors, chisq)

    def _chi2_vector(self, ra_vs_epoch, dec_vs_epoch):
        ra_solution_vecs = self.astrometric_solution_vector_components['ra']
        dec_solution_vecs = self.astrometric_solution_vector_components['dec']
        # sum together the individual solution vectors for each epoch
        return np.dot(ra_vs_epoch, ra_solution_vecs) + np.dot(dec_vs_epoch, dec_solution_vecs)

    def _init_astrometric_solution_vectors(self, fit_degree):
        # order of variables: 0, 1, 2, ... = \[Alpha]o, \[Delta]o, \[Mu]\[Alpha], \[Mu]\[Delta],  a\[Alpha], a\[Delta]
        # j\[Alpha], j\[Delta], \[Omega]
        num_epochs = len(self.epoch_times)
        plx = 1 * self.use_parallax
        astrometric_solution_vector_components = {'ra': np.zeros((num_epochs, 2 * fit_degree + 2 + plx)),
                                                  'dec': np.zeros((num_epochs, 2 * fit_degree + 2 + plx))}
        normed_epochs = normalize(self.epoch_times, [np.max(self.epoch_times), np.min(self.epoch_times)])
        for obs in range(num_epochs):
            a, b, c, d = unpack_elements_of_matrix(self.inverse_covariance_matrices[obs])
            dec_time, ra_time = normed_epochs[obs], normed_epochs[obs]
            w_ra, w_dec = self.parallactic_pertubations[0][obs], self.parallactic_pertubations[1][obs]
            clip_i = 0 if self.use_parallax else 1
            astrometric_solution_vector_components['ra'][obs] = ra_sol_vec(a, b, c, d, ra_time, dec_time,
                                                                           w_ra, w_dec, deg=fit_degree)[clip_i:]
            astrometric_solution_vector_components['dec'][obs] = dec_sol_vec(a, b, c, d, ra_time, dec_time,
                                                                             w_ra, w_dec, deg=fit_degree)[clip_i:]
        return astrometric_solution_vector_components

    def _init_astrometric_chi_squared_matrix(self, fit_degree):
        # order of variables column-wise: 0, 1, 2, ... = \[Alpha]o, \[Delta]o, \[Mu]\[Alpha], \[Mu]\[Delta],
        # a\[Alpha], a\[Delta], j\[Alpha], j\[Delta], \[Omega]
        num_epochs = len(self.epoch_times)
        plx = 1 * self.use_parallax
        astrometric_chi_squared_matrices = np.zeros((num_epochs, 2 * fit_degree + 2 + plx, 2 * fit_degree + 2 + plx))
        normed_epochs = normalize(self.epoch_times, [np.max(self.epoch_times), np.min(self.epoch_times)])
        for obs in range(num_epochs):
            a, b, c, d = unpack_elements_of_matrix(self.inverse_covariance_matrices[obs])
            dec_time, ra_time = normed_epochs[obs], normed_epochs[obs]
            w_ra, w_dec = self.parallactic_pertubations[0][obs], self.parallactic_pertubations[1][obs]
            clip_i = 0 if self.use_parallax else 1
            astrometric_chi_squared_matrices[obs] = chi2_matrix(a, b, c, d, ra_time, dec_time,
                                                                w_ra, w_dec, deg=fit_degree)[clip_i:, clip_i:]
        return np.sum(astrometric_chi_squared_matrices, axis=0)


def unpack_elements_of_matrix(matrix):
    return matrix.flatten()


def normalize(coordinates, domain):
    """
    :param coordinates: ndarray
    :param domain: ndarray. max and min value of input coordinates.
    :return: coordinates normalized to run from -1 to 1.
    """
    coordinates = 2. * (coordinates - min(domain))/(max(domain) - min(domain)) - 1.
    return coordinates
