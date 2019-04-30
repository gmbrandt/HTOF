import numpy as np


class AstrometricFitter(object):
    """
    :param inverse_covariance_matrices: ndarray of length epoch times with the 2x2 inverse covariance matrices
                                        for each epoch
    :param epoch_times: 1D ndarray with the times for each epoch.
    """
    def __init__(self, inverse_covariance_matrices=None, epoch_times=None,
                 astrometric_chi_squared_matrices=None, astrometric_solution_vector_components=None):
        self.inverse_covariance_matrices = inverse_covariance_matrices
        self.epoch_times = epoch_times
        if astrometric_solution_vector_components is None:
            self.astrometric_solution_vector_components = self._init_astrometric_solution_vectors()
        if astrometric_chi_squared_matrices is None:
            self._chi2_matrix = self._init_astrometric_chi_squared_matrix()

    def fit_line(self, ra_vs_epoch, dec_vs_epoch):
        """
        :param ra_vs_epoch: 1d array of right ascension, ordered the same as the covariance matrices and epochs.
        :param dec_vs_epoch: 1d array of declination, ordered the same as the covariance matrices and epochs.
        :return:
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
        for epoch in range(num_epochs):
            d, b, c, a = unpack_elements_of_matrix(self.inverse_covariance_matrices[epoch])
            b, c = -b, -c
            epoch_time = self.epoch_times[epoch]
            ra_vec, dec_vec = np.zeros(4).astype(np.float64), np.zeros(4).astype(np.float64)
            ra_vec[0] = -(-2 * d * epoch_time)
            ra_vec[1] = -((b + c) * epoch_time)
            ra_vec[2] = -(-2 * d)
            ra_vec[3] = -(b + c)

            dec_vec[0] = -((b + c) * epoch_time)
            dec_vec[1] = -(- 2 * a * epoch_time)
            dec_vec[2] = -(b + c)
            dec_vec[3] = -(- 2 * a)

            astrometric_solution_vector_components['ra'][epoch] = ra_vec
            astrometric_solution_vector_components['dec'][epoch] = dec_vec
        return astrometric_solution_vector_components

    def _init_astrometric_chi_squared_matrix(self):
        num_epochs = len(self.epoch_times)
        astrometric_chi_squared_matrices = np.zeros((num_epochs, 4, 4))
        for epoch in range(num_epochs):
            d, b, c, a = unpack_elements_of_matrix(self.inverse_covariance_matrices[epoch])
            b, c = -b, -c
            epoch_time = self.epoch_times[epoch]

            A = np.zeros((4, 4))

            A[:, 0] = np.array([2 * d * epoch_time,
                                (-b - c) * epoch_time,
                                2 * d,
                                (-b - c)])
            A[:, 1] = np.array([(-b - c) * epoch_time,
                                2 * a * epoch_time,
                                (-b - c),
                                2 * a])
            A[:, 2] = np.array([2 * d * epoch_time ** 2,
                                (-b - c) * epoch_time ** 2,
                                2 * d * epoch_time,
                                (-b - c) * epoch_time])
            A[:, 3] = np.array([(-b - c) * epoch_time ** 2,
                                2 * a * epoch_time ** 2,
                                (-b - c) * epoch_time,
                                2 * a * epoch_time])

            astrometric_chi_squared_matrices[epoch] = A
        return np.sum(astrometric_chi_squared_matrices, axis=0)


def unpack_elements_of_matrix(matrix):
    return matrix.flatten()
