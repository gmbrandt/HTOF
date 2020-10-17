"""
Module for fitting astrometric data.

Author: G. Mirek Brandt
"""
from numba import jit
import numpy as np
import warnings
from htof.utils.fit_utils import ra_sol_vec, dec_sol_vec, chi2_matrix, transform_coefficients_to_unnormalized_domain
from htof.utils.fit_utils import chisq_of_fit


class AstrometricFitter(object):
    """
    :param inverse_covariance_matrices: ndarray of length epoch times with the 2x2 inverse covariance matrices
                                        for each epoch
    :param epoch_times: 1D array
                        array with each epoch. If these are years, then the returned proper motions will
                         be in Angle/year as well. The unit of epoch_times should be the same as central_epoch_ra
                         and central_epoch_dec.
    :param parallactic_pertubations: dict
           {'ra_plx': array-like, 'dec_plx': array-like}
           the pertubations from parallactic motion. e.g. central_ra + parallactic_pertubations['ra_plx']
           would give the skypath (in RA) of the object throughout the time observed, from parallax alone.
    :param fit_degree: int.
                       number of degrees in polynomial fit. fit_degree=1 with parallax would be a five parameter
                       fit. fit_degree=2 with parallax would be a 7 parameter fit.
    """
    def __init__(self, inverse_covariance_matrices=None, epoch_times=None,
                 astrometric_chi_squared_matrices=None, astrometric_solution_vector_components=None,
                 parallactic_pertubations=None, fit_degree=1, use_parallax=False,
                 central_epoch_ra=0, central_epoch_dec=0, normed=False):
        if normed:
            self._on_normed()
        if parallactic_pertubations is None:
            parallactic_pertubations = {'ra_plx': np.zeros_like(epoch_times),
                                        'dec_plx': np.zeros_like(epoch_times)}
        self.use_parallax = use_parallax
        self.parallactic_pertubations = parallactic_pertubations
        self.inverse_covariance_matrices = inverse_covariance_matrices
        self.epoch_times = epoch_times
        self.fit_degree = fit_degree
        self.central_epoch_dec = central_epoch_dec
        self.central_epoch_ra = central_epoch_ra
        self.normed = normed
        self.ra_epochs, self.dec_epochs = self._init_epochs()

        if astrometric_solution_vector_components is None:
            self.astrometric_solution_vector_components = self._init_astrometric_solution_vectors(fit_degree)
        if astrometric_chi_squared_matrices is None:
            self._chi2_matrix, astrometric_chi_squared_matrices = self._init_astrometric_chi_squared_matrix(fit_degree)
        self.astrometric_chi_squared_matrices = astrometric_chi_squared_matrices

    def _on_normed(self):
        warnings.warn('the normed fitting option (normed=True) will be removed in a future release.'
                      ' Do not use normed=True because it will cause the returned fit errors to be incorrect.',
                      PendingDeprecationWarning)

    def fit_line(self, ra_vs_epoch, dec_vs_epoch, return_all=False):
        """
        :param ra_vs_epoch: 1d array of right ascension, ordered the same as the covariance matrices and epochs.
        :param dec_vs_epoch: 1d array of declination, ordered the same as the covariance matrices and epochs.
        :param return_all: bool. True to return the solution vector as well as the 1-sigma error estimates on the parameters.
        :return: ndarray: best fit astrometric parameters.
                 E.g. [ra0, dec0, mu_ra, mu_dec] if use_parallax=False
                 or, [parallax_angle, ra0, dec0, mu_ra, mu_dec] if use_parallax=True
        """
        # performing the SVD fit.
        # linalg.pinv calculates the pseudo inverse via singular-value-decomposition.
        # hermitian=True forces the _chi2_matrix to be symmetric.
        cov_matrix = np.linalg.pinv(self._chi2_matrix, hermitian=True)
        solution = np.matmul(cov_matrix, self._chi2_vector(ra_vs_epoch=ra_vs_epoch, dec_vs_epoch=dec_vs_epoch))
        errors = np.sqrt(np.diagonal(cov_matrix))
        # calculating chisq of the fit.
        chisq = chisq_of_fit(solution, ra_vs_epoch, dec_vs_epoch,
                             self.ra_epochs, self.dec_epochs,
                             self.inverse_covariance_matrices, **self.parallactic_pertubations,
                             use_parallax=self.use_parallax)
        if self.normed:
            # transforming out of normalized coordinates.
            c_ra, c_dec = self.central_epoch_ra, self.central_epoch_dec
            t = self.epoch_times
            solution = transform_coefficients_to_unnormalized_domain(solution, t.min() - c_ra, t.max() - c_ra,
                                                                     t.min() - c_dec, t.max() - c_dec, self.use_parallax)
            # TODO : One should not transform the errors like the solution vector.
            # TODO: easy but bulky way: reconstruct the chi^2 matrix here.
            errors = transform_coefficients_to_unnormalized_domain(errors, t.min() - c_ra, t.max() - c_ra,
                                                                   t.min() - c_dec, t.max() - c_dec, self.use_parallax)
        return solution if not return_all else (solution, errors, chisq)

    def _chi2_vector(self, ra_vs_epoch, dec_vs_epoch):
        # this is the vector of partial derivatives of chisquared with respect to each astrometric parameter
        ra_solution_vecs = self.astrometric_solution_vector_components['ra']
        dec_solution_vecs = self.astrometric_solution_vector_components['dec']
        # sum together the individual solution vectors for each epoch
        return np.dot(ra_vs_epoch, ra_solution_vecs) + np.dot(dec_vs_epoch, dec_solution_vecs)

    def _init_astrometric_solution_vectors(self, fit_degree):
        num_epochs = len(self.epoch_times)
        plx = 1 * self.use_parallax
        astrometric_solution_vector_components = {'ra': np.zeros((num_epochs, 2 * fit_degree + 2 + plx)),
                                                  'dec': np.zeros((num_epochs, 2 * fit_degree + 2 + plx))}
        for obs in range(num_epochs):
            a, b, c, d = unpack_elements_of_matrix(self.inverse_covariance_matrices[obs])
            w_ra, w_dec = self.parallactic_pertubations['ra_plx'][obs], self.parallactic_pertubations['dec_plx'][obs]
            clip_i = 0 if self.use_parallax else 1
            astrometric_solution_vector_components['ra'][obs] = ra_sol_vec(a, b, c, d,
                                                                           self.ra_epochs[obs], self.dec_epochs[obs],
                                                                           w_ra, w_dec, deg=fit_degree)[clip_i:]
            astrometric_solution_vector_components['dec'][obs] = dec_sol_vec(a, b, c, d,
                                                                             self.ra_epochs[obs], self.dec_epochs[obs],
                                                                             w_ra, w_dec, deg=fit_degree)[clip_i:]
        return astrometric_solution_vector_components

    def _init_astrometric_chi_squared_matrix(self, fit_degree):
        num_epochs = len(self.epoch_times)
        plx = 1 * self.use_parallax
        astrometric_chi_squared_matrices = np.zeros((num_epochs, 2 * fit_degree + 2 + plx, 2 * fit_degree + 2 + plx))
        for obs in range(num_epochs):
            a, b, c, d = unpack_elements_of_matrix(self.inverse_covariance_matrices[obs])
            w_ra, w_dec = self.parallactic_pertubations['ra_plx'][obs], self.parallactic_pertubations['dec_plx'][obs]
            clip_i = 0 if self.use_parallax else 1
            astrometric_chi_squared_matrices[obs] = chi2_matrix(a, b, c, d,
                                                                self.ra_epochs[obs], self.dec_epochs[obs],
                                                                w_ra, w_dec, deg=fit_degree)[clip_i:, clip_i:]
        return np.sum(astrometric_chi_squared_matrices, axis=0), astrometric_chi_squared_matrices

    def _init_epochs(self):
        if not self.normed:
            # comment so that unit test registers.
            return np.array(self.epoch_times) - self.central_epoch_ra, np.array(self.epoch_times) - self.central_epoch_dec
        if self.normed:
            normed_epochs = normalize(self.epoch_times, [np.max(self.epoch_times), np.min(self.epoch_times)])
            return 1.*normed_epochs, 1.*normed_epochs


class AstrometricFastFitter(AstrometricFitter):
    """
    A faster version of AstrometricFitter. Can not return errors or the chisquared. Roughly 30 times faster
    per call of fit_line than AstrometricFitter.
    """
    def fit_line(self, ra_vs_epoch, dec_vs_epoch):
        """
        :param ra_vs_epoch: 1d array of right ascension, ordered the same as the covariance matrices and epochs.
        :param dec_vs_epoch: 1d array of declination, ordered the same as the covariance matrices and epochs.
        :return: ndarray: best fit astrometric parameters.
                 E.g. [ra0, dec0, mu_ra, mu_dec] if use_parallax=False
                 or, [parallax_angle, ra0, dec0, mu_ra, mu_dec] if use_parallax=True
        """
        return fast_fit_line(self._chi2_matrix, self.astrometric_solution_vector_components['ra'],
                             self.astrometric_solution_vector_components['dec'], ra_vs_epoch, dec_vs_epoch)

    def _on_normed(self):
        raise NotImplementedError('AstrometricFastFitter cannot implement the normed=True fit feature')


@jit(nopython=True)
def fast_fit_line(chi2mat, ra_solution_vecs, dec_solution_vecs, ra_vs_epoch, dec_vs_epoch):
    return np.linalg.solve(chi2mat, np.dot(ra_vs_epoch, ra_solution_vecs) + np.dot(dec_vs_epoch, dec_solution_vecs))


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
