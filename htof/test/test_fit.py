import numpy as np
import pytest
import mock
from htof.fit import unpack_elements_of_matrix, AstrometricFitter, _verify_epoch
from htof.utils.fit_utils import ra_sol_vec, dec_sol_vec, chi2_matrix


class TestAstrometricFitter:
    def test_ra_solution_vector(self):
        assert np.allclose([2, 30, 60, 1050, 900, 18375, 9000, 214375, 326], ra_sol_vec(1, 10, 20, 5, 30, 35, 13, 10))

    def test_dec_solution_vector(self):
        assert np.allclose([30, 10, 900, 350, 13500, 6125, 135000, 214375.0/3, 490], dec_sol_vec(1, 10, 20, 5, 30, 35, 13, 10))

    def test_chi2_matrix(self):
        expected_chi2_matrix = np.array([[2, 30, 60, 1050, 900, 18375, 9000, 214375, 326],
                                         [30, 10, 900, 350, 13500, 6125, 135000, 214375.0/3, 490],
                                         [60, 900, 1800, 31500, 27000, 551250, 270000, 6431250, 9780],
                                         [1050, 350, 31500, 12250, 472500, 214375, 4725000, 7503125.0/3, 17150],
                                         [900, 13500, 27000, 472500, 405000, 8268740, 4050000, 96468750, 146700],
                                         [18375, 6125, 551250, 214375, 8268750, 7503125.0/2, 82687500, 262609375.0/6, 300125],
                                         [9000, 135000, 270000, 4725000, 4050000, 82687500, 40500000, 964687500, 1467000],
                                         [214375, 214375.0/3, 6431250, 7503125.0/3, 96468750, 262609375.0/6, 964687500, 9191328125.0/18, 10504375.0/3],
                                         [326, 490, 9780, 17150, 146700, 300125, 1467000, 10504375.0/3, 9138]])
        agreement = np.isclose(expected_chi2_matrix, chi2_matrix(1, 10, 20, 5, 30, 35, 13, 10))
        if np.all(agreement):
            assert True
        else:
            print('disagreeing chi2 positions:')
            print(np.where(~agreement))
            print('yours and expected:')
            print(chi2_matrix(1, 10, 20, 5, 30, 35, 13, 10)[np.where(~agreement)])
            print(expected_chi2_matrix[np.where(~agreement)])
            assert False

    @mock.patch('htof.fit.chi2_matrix', return_value=np.ones((9, 9)))
    def test_chi2_matrix_many_epoch(self, fake_chi2_matrix_per_epoch):
        ivar = np.ones((2, 2))
        fitter = AstrometricFitter(inverse_covariance_matrices=[ivar, ivar, ivar], epoch_times=[2, 2, 2],
                                   astrometric_solution_vector_components=[])
        assert np.allclose(np.ones((9, 9)) * 3, fitter._init_astrometric_chi_squared_matrix(9))
        assert np.allclose(np.ones((7, 7)) * 3, fitter._init_astrometric_chi_squared_matrix(7))

    def test_chi2_vector(self):
        covariance_matrix = np.array([[5, 1], [12, 2]])
        ra, dec = 91, 82
        epoch_time = 30
        expected_c = [351, 363.0/2, 10530, 5445]
        fitter = AstrometricFitter(inverse_covariance_matrices=np.array([np.linalg.pinv(covariance_matrix)]),
                                   epoch_times=np.array([epoch_time]),
                                   astrometric_chi_squared_matrices=[], parameters=4)
        assert np.allclose(expected_c, fitter._chi2_vector(ra_vs_epoch=np.array([ra]),
                                                           dec_vs_epoch=np.array([dec])))

    def test_fitting_to_linear_astrometric_data(self):
        astrometric_data = generate_linear_astrometric_data(correlation_coefficient=0, sigma_ra=0.1, sigma_dec=0.1)
        fitter = AstrometricFitter(inverse_covariance_matrices=astrometric_data['inverse_covariance_matrix'],
                                   epoch_times=astrometric_data['epoch_delta_t'])

        assert np.allclose(fitter.fit_line(astrometric_data['ra'], astrometric_data['dec']),
                           astrometric_data['linear_solution'])
    # we should have another test where central epochs from our fit differs, but the evaluated ra and dec points
    # should agree with the test data set.

    def test_fitting_with_nonzero_central_epoch(self):
        ra_cnt = np.random.randint(1, 100)
        dec_cnt = np.random.randint(1, 100)
        print(ra_cnt, dec_cnt)
        astrometric_data = generate_linear_astrometric_data(correlation_coefficient=0, sigma_ra=0.1, sigma_dec=0.1)
        fitter = AstrometricFitter(inverse_covariance_matrices=astrometric_data['inverse_covariance_matrix'],
                                   epoch_times=astrometric_data['epoch_delta_t'],
                                   central_epoch_dec=dec_cnt, central_epoch_ra=ra_cnt)
        expected_vec = astrometric_data['linear_solution']
        expected_vec[0] += ra_cnt * expected_vec[2]  # r0 = ra_central_time * mu_ra
        expected_vec[1] += dec_cnt * expected_vec[3]  # dec0 = dec_central_time * mu_dec
        import matplotlib.pyplot as plt
        fit = fitter.fit_line(astrometric_data['ra'], astrometric_data['dec'])
        plt.plot(astrometric_data['epoch_delta_t'], astrometric_data['ra'], 'r+')
        plt.plot(astrometric_data['epoch_delta_t'], (astrometric_data['epoch_delta_t'] - ra_cnt) * fit[2] + fit[0])
        plt.plot(astrometric_data['epoch_delta_t'], (astrometric_data['epoch_delta_t'] - ra_cnt) * expected_vec[2] + expected_vec[0])
        plt.show()
        assert np.allclose(fitter.fit_line(astrometric_data['ra'], astrometric_data['dec']), expected_vec)


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
    A = np.array([[0, 1], [2, 3]])
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
