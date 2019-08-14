import numpy as np
import pytest
import mock
from astropy.time import Time

from htof.fit import unpack_elements_of_matrix, AstrometricFitter, _verify_epoch
from htof.utils.fit_utils import ra_sol_vec, dec_sol_vec, chi2_matrix
from htof.sky_path import parallactic_motion


class TestAstrometricFitter:
    def test_ra_solution_vector(self):
        assert np.allclose([326, 2, 30, 60, 1050, 1800, 36750, 54000, 1286250], ra_sol_vec(1, 10, 20, 5, 30, 35, 13, 10))

    def test_dec_solution_vector(self):
        assert np.allclose([490, 30, 10, 900, 350, 27000, 12250, 810000, 428750], dec_sol_vec(1, 10, 20, 5, 30, 35, 13, 10))

    def test_chi2_matrix(self):
        expected_chi2_matrix = np.array([
        [9138, 326, 490, 9780, 17150, 293400, 600250, 8802000, 21008750],
        [326, 2, 30, 60, 1050, 1800, 36750, 54000, 1286250],
        [490, 30, 10, 900, 350, 27000, 12250, 810000, 428750],
        [9780, 60, 900, 1800, 31500, 54000, 1102500, 1620000, 38587500],
        [17150, 1050, 350, 31500, 12250, 945000, 428750, 28350000, 15006250],
        [293400, 1800, 27000, 54000, 945000, 1620000, 33075000, 48600000, 1157625000],
        [600250, 36750, 12250, 1102500, 428750, 33075000, 15006250, 992250000, 525218750],
        [8802000, 54000, 810000, 1620000, 28350000, 48600000, 992250000, 1458000000, 34728750000],
        [21008750, 1286250, 428750, 38587500, 15006250, 1157625000, 525218750, 34728750000, 18382656250]
        ])
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
                                   astrometric_chi_squared_matrices=[], parameters=5,
                                   parallactic_pertubations=None)
        assert np.allclose(expected_c, fitter._chi2_vector(ra_vs_epoch=np.array([ra]),
                                                           dec_vs_epoch=np.array([dec])))

    def test_fitting_to_linear_astrometric_data(self):
        astrometric_data = generate_astrometric_data(correlation_coefficient=0, sigma_ra=0.1, sigma_dec=0.1)
        fitter = AstrometricFitter(inverse_covariance_matrices=astrometric_data['inverse_covariance_matrix'],
                                   epoch_times=astrometric_data['epoch_delta_t'])

        assert np.allclose(fitter.fit_line(astrometric_data['ra'], astrometric_data['dec']),
                           astrometric_data['linear_solution'])
    # we should have another test where central epochs from our fit differs, but the evaluated ra and dec points
    # should agree with the test data set.

    def test_fitting_with_nonzero_central_epoch(self):
        ra_cnt = np.random.randint(1, 100)
        dec_cnt = np.random.randint(1, 100)
        astrometric_data = generate_astrometric_data(correlation_coefficient=0, sigma_ra=0.1, sigma_dec=0.1)
        fitter = AstrometricFitter(inverse_covariance_matrices=astrometric_data['inverse_covariance_matrix'],
                                   epoch_times=astrometric_data['epoch_delta_t'],
                                   central_epoch_dec=dec_cnt, central_epoch_ra=ra_cnt)
        expected_vec = astrometric_data['linear_solution']
        expected_vec[0] += ra_cnt * expected_vec[2]  # r0 = ra_central_time * mu_ra
        expected_vec[1] += dec_cnt * expected_vec[3]  # dec0 = dec_central_time * mu_dec
        assert np.allclose(fitter.fit_line(astrometric_data['ra'], astrometric_data['dec']), expected_vec)

    def test_fitting_to_non_linear_astrometric_data_without_parallax(self):
        astrometric_data = generate_astrometric_data(correlation_coefficient=0, sigma_ra=0.1, sigma_dec=0.1,
                                                     acc=True, jerk=True)
        fitter = AstrometricFitter(inverse_covariance_matrices=astrometric_data['inverse_covariance_matrix'],
                                   epoch_times=astrometric_data['epoch_delta_t'], parameters=9,
                                   parallactic_pertubations=None)
        assert np.allclose(fitter.fit_line(astrometric_data['ra'], astrometric_data['dec']),
                           astrometric_data['nonlinear_solution'], rtol=1E-2)

    def test_fitting_to_non_linear_astrometric_data_with_parallax(self):
        real_plx = 10
        astrometric_data = generate_astrometric_data(correlation_coefficient=0, sigma_ra=0.1, sigma_dec=0.1,
                                                     acc=False, jerk=False)
        jyear_epochs = Time(astrometric_data['epoch_delta_t'] + 2448090, format='jd').jyear
        ra_pert, dec_pert = parallactic_motion(jyear_epochs, 45, 45, 'degree', 1991.25, parallax=real_plx)
        import matplotlib.pyplot as plt
        t = astrometric_data['epoch_delta_t']
        ra_pert, dec_pert = 3E-11 * t**2.3, 1E-11 * t**2.2  # 1E-11 * t**2, 1E-11 * t**2
        #ra_pert, dec_pert = 1E-11*np.sin(t/100), 1E-11*np.sin(t/100)
        #plt.plot(astrometric_data['epoch_delta_t'], astrometric_data['ra'], 'r')
        astrometric_data['dec'] += dec_pert
        astrometric_data['ra'] += ra_pert
        plt.plot(astrometric_data['epoch_delta_t'], astrometric_data['ra'], 'b', lw=3)
        plt.plot(astrometric_data['epoch_delta_t'], astrometric_data['dec'], 'b--', lw=3)
        fitter = AstrometricFitter(inverse_covariance_matrices=astrometric_data['inverse_covariance_matrix'],
                                   epoch_times=astrometric_data['epoch_delta_t'], parameters=5,
                                   parallactic_pertubations=[ra_pert, dec_pert])
        fit = fitter.fit_line(astrometric_data['ra'], astrometric_data['dec'])

        import matplotlib.pyplot as plt
        t = astrometric_data['epoch_delta_t']
        fp = fit[-1]
        ra0, dec0, mu_ra, mu_dec = fit[:-1]
        print(fit)
        best_fit_ra = ra0 + t * mu_ra + fp * ra_pert
        best_fit_dec = dec0 + t * mu_dec + fp * dec_pert
        plt.plot(t, best_fit_ra, 'k')
        plt.plot(t, best_fit_dec, 'k--')
        plt.figure()
        plt.plot(t, (best_fit_ra - astrometric_data['ra']) / best_fit_ra, 'r+')
        plt.show()
        assert np.allclose(best_fit_dec, astrometric_data['dec'], rtol=1E-4)
        assert np.allclose(best_fit_ra, astrometric_data['ra'], rtol=1E-4)

    def test_fitter_removes_parallax(self):
        astrometric_data = generate_astrometric_data(correlation_coefficient=0, sigma_ra=0.1, sigma_dec=0.1)
        fitter = AstrometricFitter(inverse_covariance_matrices=astrometric_data['inverse_covariance_matrix'],
                                   epoch_times=astrometric_data['epoch_delta_t'], parameters=9,
                                   parallactic_pertubations=None)
        assert fitter._chi2_matrix.shape == (8, 8)
        assert fitter.astrometric_solution_vector_components['ra'][0].shape == (8,)
        assert fitter.astrometric_solution_vector_components['dec'][0].shape == (8,)


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


def generate_astrometric_data(correlation_coefficient=0.0, sigma_ra=0.1, sigma_dec=0.1, acc=False, jerk=False):
    astrometric_data = {}
    num_measurements = 50
    mu_ra, mu_dec = 1E-9, 2E-9
    acc_ra, acc_dec = acc * 2E-12, acc * 1E-12
    jerk_ra, jerk_dec = jerk * 2E-14, jerk * 1E-14
    ra0, dec0 = 2E-9, 1E-9
    epoch_start = 0
    epoch_end = 1000
    t = np.linspace(epoch_start, epoch_end, num=num_measurements)
    astrometric_data['epoch_delta_t'] = t
    astrometric_data['dec'] = dec0 + t * mu_dec + 1/2 * acc_dec * t ** 2 + 1/6 * jerk_dec * t ** 3
    astrometric_data['ra'] = ra0 + t * mu_ra + 1/2 * acc_ra * t ** 2 + 1/6 * jerk_ra * t ** 3
    cc = correlation_coefficient
    astrometric_data['covariance_matrix'] = np.zeros((num_measurements, 2, 2))
    astrometric_data['inverse_covariance_matrix'] = np.zeros((num_measurements, 2, 2))
    astrometric_data['covariance_matrix'][:] = np.array([[sigma_ra**2, sigma_ra*sigma_dec*cc],
                                                       [sigma_ra*sigma_dec*cc, sigma_dec**2]])
    for i in range(len(astrometric_data)):
        astrometric_data['inverse_covariance_matrix'][i] = np.linalg.pinv(astrometric_data['covariance_matrix'][i])
    astrometric_data['linear_solution'] = np.array([ra0, dec0, mu_ra, mu_dec])
    astrometric_data['nonlinear_solution'] = np.array([ra0, dec0, mu_ra, mu_dec, acc_ra, acc_dec, jerk_ra, jerk_dec])

    return astrometric_data
