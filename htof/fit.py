"""
Module for fitting astrometric data.

Author: G. Mirek Brandt
"""

import numpy as np
import warnings
from htof.parse import fractional_year_epoch_to_jd
from htof.utils.fit_utils import ra_sol_vec, dec_sol_vec, chi2_matrix


class AstrometricFitter(object):
    """
    :param inverse_covariance_matrices: ndarray of length epoch times with the 2x2 inverse covariance matrices
                                        for each epoch
    :param epoch_times: 1D array
                        array with each epoch in Barycentric Julian Date (BJD).
    :param parallactic_pertubations: dictionary
    :param parameters: int.
                       number of parameters in the fit. Options are 4, 5, 7, and 9.
                       4 is just offset and proper motion, 5 includes parallax, 7 and 9 include accelerations and jerks.
    The pertubations due to parallactic motion alone with unit parallax. Where parallactic_pertubations['ra'], and
    parallactic_pertubations['dec'] are the pertubations for right ascension and declination
    For each component this should be the quantity which is linear in parallax angle, i.e.:
    Parallax_motion_ra - central_ra.
    The units of this parallactic motion should be exactly the same as the ra's and dec's which you will fit
    later on.
    """
    def __init__(self, inverse_covariance_matrices=None, epoch_times=None,
                 astrometric_chi_squared_matrices=None, astrometric_solution_vector_components=None,
                 parallactic_pertubations=None, parameters=5,
                 central_epoch_ra=0, central_epoch_dec=0, central_epoch_fmt='BJD'):
        if parameters not in [5, 7, 9]:
            raise ValueError('parameters argument of AstrometricFitter not equal to any one of 5, 7, or 9.')
        if parallactic_pertubations is None:
            parallactic_pertubations = [np.zeros_like(epoch_times), np.zeros_like(epoch_times)]
            parameters -= 1

        self.parallactic_pertubations = parallactic_pertubations
        self.inverse_covariance_matrices = inverse_covariance_matrices
        self.epoch_times = epoch_times
        self.central_epoch_dec, self.central_epoch_ra = _verify_epoch(central_epoch_dec,
                                                                      central_epoch_ra,
                                                                      central_epoch_fmt)

        if astrometric_solution_vector_components is None:
            self.astrometric_solution_vector_components = self._init_astrometric_solution_vectors(parameters)
        if astrometric_chi_squared_matrices is None:
            self._chi2_matrix = self._init_astrometric_chi_squared_matrix(parameters)

    def fit_line(self, ra_vs_epoch, dec_vs_epoch, full_output=False):
        #TODO rename to fit, since this fits for jerks now. Make sure this does not cause
        # issues in Tim's orbit code.
        """
        :param ra_vs_epoch: 1d array of right ascension, ordered the same as the covariance matrices and epochs.
        :param dec_vs_epoch: 1d array of declination, ordered the same as the covariance matrices and epochs.
        :param full_output: bool. True to return the solution vector as well as chi_squared for the fit.
        :return: Array:
                 [ra0, dec0, mu_ra, mu_dec]
        """
        print(np.linalg.cond(self._chi2_matrix))
        return np.linalg.solve(self._chi2_matrix, self._chi2_vector(ra_vs_epoch=ra_vs_epoch,
                                                                    dec_vs_epoch=dec_vs_epoch))

    def _chi2_vector(self, ra_vs_epoch, dec_vs_epoch):
        ra_solution_vecs = self.astrometric_solution_vector_components['ra']
        dec_solution_vecs = self.astrometric_solution_vector_components['dec']
        # sum together the individual solution vectors for each epoch
        return np.dot(ra_vs_epoch, ra_solution_vecs) + np.dot(dec_vs_epoch, dec_solution_vecs)

    def _init_astrometric_solution_vectors(self, parameters):
        # order of variables: 0, 1, 2, ... = \[Alpha]o, \[Delta]o, \[Mu]\[Alpha], \[Mu]\[Delta],  a\[Alpha], a\[Delta]
        # j\[Alpha], j\[Delta], \[Omega]
        p = parameters
        num_epochs = len(self.epoch_times)
        astrometric_solution_vector_components = {'ra': np.zeros((num_epochs, p)),
                                                  'dec': np.zeros((num_epochs, p))}
        for obs in range(num_epochs):
            a, b, c, d = unpack_elements_of_matrix(self.inverse_covariance_matrices[obs])
            epoch_time = self.epoch_times[obs]
            dec_time = epoch_time - self.central_epoch_dec
            ra_time = epoch_time - self.central_epoch_ra
            w_ra, w_dec = self.parallactic_pertubations[0][obs], self.parallactic_pertubations[1][obs]

            astrometric_solution_vector_components['ra'][obs] = ra_sol_vec(a, b, c, d, ra_time, dec_time,
                                                                           w_ra, w_dec)[:p]
            astrometric_solution_vector_components['dec'][obs] = dec_sol_vec(a, b, c, d, ra_time, dec_time,
                                                                             w_ra, w_dec)[:p]
        return astrometric_solution_vector_components

    def _init_astrometric_chi_squared_matrix(self, parameters):
        # order of variables column-wise: 0, 1, 2, ... = \[Alpha]o, \[Delta]o, \[Mu]\[Alpha], \[Mu]\[Delta],
        # a\[Alpha], a\[Delta], j\[Alpha], j\[Delta], \[Omega]
        p = parameters
        num_epochs = len(self.epoch_times)
        astrometric_chi_squared_matrices = np.zeros((num_epochs, p, p))
        for obs in range(num_epochs):
            a, b, c, d = unpack_elements_of_matrix(self.inverse_covariance_matrices[obs])
            epoch_time = self.epoch_times[obs]
            dec_time = epoch_time - self.central_epoch_dec
            ra_time = epoch_time - self.central_epoch_ra
            w_ra, w_dec = self.parallactic_pertubations[0][obs], self.parallactic_pertubations[1][obs]

            astrometric_chi_squared_matrices[obs] = chi2_matrix(a, b, c, d, ra_time, dec_time,
                                                                w_ra, w_dec)[:p, :p]
        return np.sum(astrometric_chi_squared_matrices, axis=0)

    def _chisq_of_fit(self, sol_vector, ra_vs_epoch, dec_vs_epoch):
        if self.use_parallax:
            plx = sol_vector[-1]
            ra_sol, dec_sol = sol_vector[:-1][::2], sol_vector[:-1][1::2]
        else:
            plx = 0
            ra_sol, dec_sol = sol_vector[::2], sol_vector[1::2]
        dec_time = self.epoch_times - self.central_epoch_dec
        ra_time = self.epoch_times - self.central_epoch_ra
        model_ra = np.polynomial.polynomial.polyval(ra_time, ra_sol * np.array([1, 1, 1/2, 1/6])[:num_p])
        model_dec = np.polynomial.polynomial.polyval(dec_time, dec_sol * np.array([1, 1, 1/2, 1/6])[:num_p])
        model_minus_data = np.array([model_ra - ra_vs_epoch, model_dec - dec_vs_epoch])
        return np.sum([np.dot(np.dot(model_minus_data[:, i].T, self.inverse_covariance_matrices[i]),
                              model_minus_data[:, i]) for i in range(model_minus_data.shape[1])])


def unpack_elements_of_matrix(matrix):
    return matrix.flatten()


def _verify_epoch(central_epoch_dec, central_epoch_ra, central_epoch_fmt):
    if central_epoch_fmt == 'frac_year':
        if central_epoch_dec > 3000 or central_epoch_ra > 3000:
            warnings.warn('central epoch in RA or DEC was chosen to be > 3000. Are you sure this'
                          'is a fractional year date and not a BJD? If BJD, set central_epoch_fmt=BJD.',
                          UserWarning)  # pragma: no cover
        central_epoch_dec = fractional_year_epoch_to_jd(central_epoch_dec, half_day_correction=True)
        central_epoch_ra = fractional_year_epoch_to_jd(central_epoch_ra, half_day_correction=True)
    return central_epoch_dec, central_epoch_ra