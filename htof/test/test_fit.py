import numpy as np
import pytest
import mock
from htof.fit import unpack_elements_of_matrix, AstrometricFitter, _verify_epoch


class TestAstrometricFitter:
    def test_chi2_matrix_single_epoch(self):
        covariance_matrix = np.array([[5, 1], [12, 2]])
        epoch_time = 30
        expected_A = np.array([[-60, 195, -1800, 5850],
                               [195, -150, 5850, -4500],
                               [-2, 13/2, -60, 195],
                               [13/2, -5, 195, -150]])
        fitter = AstrometricFitter(inverse_covariance_matrices=[np.linalg.pinv(covariance_matrix)], epoch_times=[epoch_time],
                                   astrometric_solution_vector_components=[])
        assert np.allclose(expected_A, fitter._chi2_matrix)

    def test_chi2_solution_vector_single_epoch(self):
        covariance_matrix = np.array([[5, 1], [12, 2]])
        ra, dec = 91, 82
        epoch_time = 30
        expected_c = (-1)*np.array([-10530, -5445, -351, -363/2])
        fitter = AstrometricFitter(inverse_covariance_matrices=np.array([np.linalg.pinv(covariance_matrix)]),
                                   epoch_times=np.array([epoch_time]),
                                   astrometric_chi_squared_matrices=[])
        assert np.allclose(expected_c, fitter._chi2_vector(ra_vs_epoch=np.array([ra]),
                                                           dec_vs_epoch=np.array([dec])))

    def test_fitting_to_linear_astrometric_data(self):
        astrometric_data = generate_linear_astrometric_data(correlation_coefficient=0, sigma_ra=0.1, sigma_dec=0.1)
        fitter = AstrometricFitter(inverse_covariance_matrices=astrometric_data['inverse_covariance_matrix'],
                                   epoch_times=astrometric_data['epoch_delta_t'])

        assert np.allclose(fitter.fit_line(astrometric_data['ra'], astrometric_data['dec']),
                           astrometric_data['linear_solution'])

    def test_fitting_with_nonzero_central_epoch(self):
        ra_cnt = np.random.randint(1, 100)
        dec_cnt = np.random.randint(1, 100)
        astrometric_data = generate_linear_astrometric_data(correlation_coefficient=0, sigma_ra=0.1, sigma_dec=0.1)
        fitter = AstrometricFitter(inverse_covariance_matrices=astrometric_data['inverse_covariance_matrix'],
                                   epoch_times=astrometric_data['epoch_delta_t'],
                                   central_epoch_dec=dec_cnt, central_epoch_ra=ra_cnt)
        expected_vec = astrometric_data['linear_solution']
        expected_vec[0] += ra_cnt * expected_vec[2]  # r0 = ra_central_time * mu_ra
        expected_vec[1] += dec_cnt * expected_vec[3]  # dec0 = dec_central_time * mu_dec
        assert np.allclose(fitter.fit_line(astrometric_data['ra'], astrometric_data['dec']),
                           expected_vec)


@mock.patch('htof.fit.fractional_year_epoch_to_jd')
def test_verify_epoch(fake_convert):
    def convert(time, *args, **kwargs):
        return int(time)
    fake_convert.side_effect = convert

    ranew, decnew = _verify_epoch(0, 0, 'BJD')
    assert np.allclose([ranew, decnew], 0)
    ranew, decnew = _verify_epoch(2000.1, 2001.2, 'frac_year')
    assert np.allclose([ranew, decnew], [2000, 2001])


def test_verify_warns_on_large_fractional_year():
    with pytest.warns(UserWarning):
        _verify_epoch(central_epoch_dec=5000,
                      central_epoch_ra=5000, central_epoch_fmt='frac_year')


def test_unpack_elements_of_matrix():
    A = np.arange(4).reshape((2, 2))
    assert np.allclose(np.arange(4), unpack_elements_of_matrix(A))


"""
Utility functions
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
    astrometric_data['inverse_covariance_matrix'] = np.zeros((num_measurements, 2, 2))
    astrometric_data['covariance_matrix'][:] = np.array([[sigma_ra**2, sigma_ra*sigma_dec*cc],
                                                       [sigma_ra*sigma_dec*cc, sigma_dec**2]])
    for i in range(len(astrometric_data)):
        astrometric_data['inverse_covariance_matrix'][i] = np.linalg.pinv(astrometric_data['covariance_matrix'][i])
    astrometric_data['linear_solution'] = np.array([ra0, dec0, mu_ra, mu_dec])

    return astrometric_data
