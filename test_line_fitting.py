import numpy as np
import HTOF.hip_proper_motion_fit as hipfit


def test_unpack_elements_of_matrix():
    A = np.arange(4).reshape((2, 2))
    assert np.allclose(np.arange(4), hipfit.unpack_elements_of_matrix(A))


def test_chi2_matrix_single_epoch():
    covariance_matrix = np.array([[5, 1], [12, 2]])
    ra, dec = 91, 82
    epoch_delta_t = 30
    A, c = hipfit.chi2_matrix_single_epoch(covariance_matrix, epoch_delta_t, ra, dec)
    expected_A = np.array([[-60, 195, -1800, 5850],
                           [195, -150, 5850, -4500],
                           [-2, 13/2, -60, 195],
                           [13/2, -5, 195, -150]])
    expected_c = (-1)*np.array([-10530, -5445, -351, -363/2])
    assert np.allclose(expected_A, A)
    assert np.allclose(expected_c, c)

"""
integration tests
"""


def generate_linear_astrometric_data(correlation_coefficient=0.0, sigma_ra=0.1, sigma_dec=0.1):
    astrometric_data = {}
    num_measurements = 20
    mu_ra, mu_dec = 1, 2
    ra0, dec0 = 30, 40
    epoch_start = 0
    epoch_end = 200
    astrometric_data['epoch_delta_t'] = np.linspace(epoch_start, epoch_end, num=num_measurements)
    astrometric_data['dec'] = dec0 + astrometric_data['epoch_delta_t']*mu_dec
    astrometric_data['ra'] = ra0 + astrometric_data['epoch_delta_t']*mu_ra
    cc = correlation_coefficient
    astrometric_data['covariance_matrix'] = np.zeros((num_measurements, 2, 2))
    astrometric_data['covariance_matrix'][:] = np.array([[sigma_ra**2, sigma_ra*sigma_dec*cc],
                                                       [sigma_ra*sigma_dec*cc, sigma_dec**2]])
    astrometric_data['linear_solution_vector'] = np.array([ra0, dec0, mu_ra, mu_dec])

    return astrometric_data


def generate_parabolic_astrometric_data(correlation_coefficient=0.0, sigma_ra=0.1, sigma_dec=0.1, crescendo=False):
    astrometric_data = {}
    num_measurements = 20
    mu_ra, mu_dec = -1, 2
    acc_ra, acc_dec = -0.1, 0.2
    ra0, dec0 = -30, 40
    epoch_start = 0
    epoch_end = 200
    astrometric_data['epoch_delta_t'] = np.linspace(epoch_start, epoch_end, num=num_measurements)
    astrometric_data['dec'] = dec0 + astrometric_data['epoch_delta_t']*mu_dec + \
                              1 / 2 * acc_dec * astrometric_data['epoch_delta_t'] ** 2
    astrometric_data['ra'] = ra0 + astrometric_data['epoch_delta_t']*mu_ra + \
                             1 / 2 * acc_ra * astrometric_data['epoch_delta_t'] ** 2
    cc = correlation_coefficient
    astrometric_data['covariance_matrix'] = np.zeros((num_measurements, 2, 2))
    astrometric_data['covariance_matrix'][:] = np.array([[sigma_ra**2, sigma_ra*sigma_dec*cc],
                                                       [sigma_ra*sigma_dec*cc, sigma_dec**2]])
    if crescendo:
        astrometric_data['covariance_matrix'][:, 0, 0] *= np.linspace(1/10, 4, num=num_measurements)
        astrometric_data['covariance_matrix'][:, 1, 1] *= np.linspace(4, 1/10, num=num_measurements)
    astrometric_data['linear_solution_vector'] = np.array([ra0, dec0, mu_ra, mu_dec])
    return astrometric_data


def test_fitting_to_linear_astrometric_data():
    astrometric_data = generate_linear_astrometric_data(correlation_coefficient=0, sigma_ra=0.1, sigma_dec=0.1)
    solution_vector = hipfit.line_of_best_fit(astrometric_data)
    assert np.allclose(solution_vector, astrometric_data['linear_solution_vector'])
