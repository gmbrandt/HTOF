import numpy as np
import htof.main as hipfit


def test_unpack_elements_of_matrix():
    A = np.arange(4).reshape((2, 2))
    assert np.allclose(np.arange(4), hipfit.unpack_elements_of_matrix(A))


def test_chi2_matrix_single_epoch():
    covariance_matrix = np.array([[5, 1], [12, 2]])
    epoch_time = 30
    expected_A = np.array([[-60, 195, -1800, 5850],
                           [195, -150, 5850, -4500],
                           [-2, 13/2, -60, 195],
                           [13/2, -5, 195, -150]])
    fitter = hipfit.AstrometricFitter(covariance_matrices=[covariance_matrix], epoch_times=[epoch_time],
                                      astrometric_solution_vector_components=[])
    assert np.allclose(expected_A, fitter._chi2_matrix())


def test_chi2_solution_vector_single_epoch():
    covariance_matrix = np.array([[5, 1], [12, 2]])
    ra, dec = 91, 82
    epoch_time = 30
    expected_c = (-1)*np.array([-10530, -5445, -351, -363/2])
    fitter = hipfit.AstrometricFitter(covariance_matrices=np.array([covariance_matrix]),
                                      epoch_times=np.array([epoch_time]),
                                      astrometric_chi_squared_matrices=[])
    assert np.allclose(expected_c, fitter._chi2_vector(ra_vs_epoch=np.array([ra]),
                                                       dec_vs_epoch=np.array([dec])))


def test_fitting_to_linear_astrometric_data():
    astrometric_data = generate_linear_astrometric_data(correlation_coefficient=0, sigma_ra=0.1, sigma_dec=0.1)
    fitter = hipfit.AstrometricFitter(covariance_matrices=astrometric_data['covariance_matrix'],
                                      epoch_times=astrometric_data['epoch_delta_t'])

    assert np.allclose(fitter.fit_line(astrometric_data['ra'], astrometric_data['dec']),
                       astrometric_data['linear_solution'])


"""
Util functions
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
    astrometric_data['linear_solution'] = np.array([ra0, dec0, mu_ra, mu_dec])

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
    astrometric_data['linear_solution'] = np.array([ra0, dec0, mu_ra, mu_dec])
    return astrometric_data
