import numpy as np
import warnings
from htof.parse import fractional_year_epoch_to_jd


class AstrometricFitter(object):
    """
    :param inverse_covariance_matrices: ndarray of length epoch times with the 2x2 inverse covariance matrices
                                        for each epoch
    :param epoch_times: 1D ndarray with the times for each epoch.
    """
    def __init__(self, inverse_covariance_matrices=None, epoch_times=None,
                 astrometric_chi_squared_matrices=None, astrometric_solution_vector_components=None,
                 central_epoch_ra=0, central_epoch_dec=0, central_epoch_fmt='BJD'):
        self.inverse_covariance_matrices = inverse_covariance_matrices
        self.epoch_times = epoch_times
        self.central_epoch_dec, self.central_epoch_ra = _verify_epoch(central_epoch_dec,
                                                                      central_epoch_ra,
                                                                      central_epoch_fmt)
        if astrometric_solution_vector_components is None:
            self.astrometric_solution_vector_components = self._init_astrometric_solution_vectors()
        if astrometric_chi_squared_matrices is None:
            self._chi2_matrix = self._init_astrometric_chi_squared_matrix()

    def fit_line(self, ra_vs_epoch, dec_vs_epoch):
        """
        :param ra_vs_epoch: 1d array of right ascension, ordered the same as the covariance matrices and epochs.
        :param dec_vs_epoch: 1d array of declination, ordered the same as the covariance matrices and epochs.
        :return: Array:
                 [ra0, dec0, mu_ra, mu_dec]
        """
        return np.linalg.solve(self._chi2_matrix, self._chi2_vector(ra_vs_epoch=ra_vs_epoch,
                                                                    dec_vs_epoch=dec_vs_epoch))

    def _chi2_vector(self, ra_vs_epoch, dec_vs_epoch):
        ra_solution_vecs = self.astrometric_solution_vector_components['ra']
        dec_solution_vecs = self.astrometric_solution_vector_components['dec']
        # sum together the individual solution vectors for each epoch
        return np.dot(ra_vs_epoch, ra_solution_vecs) + np.dot(dec_vs_epoch, dec_solution_vecs)

    def _init_astrometric_solution_vectors(self):
        num_epochs = len(self.epoch_times)
        astrometric_solution_vector_components = {'ra': np.zeros((num_epochs, 4)),
                                                  'dec': np.zeros((num_epochs, 4))}
        for obs in range(num_epochs):
            d, b, c, a = unpack_elements_of_matrix(self.inverse_covariance_matrices[obs])
            b, c = -b, -c
            epoch_time = self.epoch_times[obs]
            dec_time = epoch_time - self.central_epoch_dec
            ra_time = epoch_time - self.central_epoch_ra
            ra_vec, dec_vec = np.zeros(4).astype(np.float64), np.zeros(4).astype(np.float64)
            ra_vec[0] = -(-2 * d * ra_time)
            ra_vec[1] = -((b + c) * dec_time)
            ra_vec[2] = -(-2 * d)
            ra_vec[3] = -(b + c)

            dec_vec[0] = -((b + c) * ra_time)
            dec_vec[1] = -(- 2 * a * dec_time)
            dec_vec[2] = -(b + c)
            dec_vec[3] = -(- 2 * a)

            astrometric_solution_vector_components['ra'][obs] = ra_vec
            astrometric_solution_vector_components['dec'][obs] = dec_vec
        return astrometric_solution_vector_components

    def _init_astrometric_chi_squared_matrix(self):
        num_epochs = len(self.epoch_times)
        astrometric_chi_squared_matrices = np.zeros((num_epochs, 4, 4))
        for obs in range(num_epochs):
            d, b, c, a = unpack_elements_of_matrix(self.inverse_covariance_matrices[obs])
            b, c = -b, -c
            epoch_time = self.epoch_times[obs]
            dec_time = epoch_time - self.central_epoch_dec
            ra_time = epoch_time - self.central_epoch_ra

            A = np.zeros((4, 4))

            A[:, 0] = np.array([2 * d * ra_time,
                                (-b - c) * dec_time,
                                2 * d,
                                (-b - c)])
            A[:, 1] = np.array([(-b - c) * ra_time,
                                2 * a * dec_time,
                                (-b - c),
                                2 * a])
            A[:, 2] = np.array([2 * d * ra_time ** 2,
                                (-b - c) * ra_time * dec_time,
                                2 * d * ra_time,
                                (-b - c) * ra_time])
            A[:, 3] = np.array([(-b - c) * ra_time * dec_time,
                                2 * a * dec_time ** 2,
                                (-b - c) * dec_time,
                                2 * a * dec_time])

            astrometric_chi_squared_matrices[obs] = A
        return np.sum(astrometric_chi_squared_matrices, axis=0)


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
