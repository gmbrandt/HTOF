import numpy as np
import pytest
import mock
from astropy.time import Time

from htof.fit import unpack_elements_of_matrix, AstrometricFitter, normalize
from htof.utils.fit_utils import ra_sol_vec, dec_sol_vec, chi2_matrix, transform_coefficients_to_unnormalized_domain
from htof.sky_path import parallactic_motion


class TestAstrometricFitter:
    def test_ra_solution_vector(self):
        assert np.allclose([326, 2, 30, 60, 1050, 1800, 36750, 54000, 1286250],
                           ra_sol_vec(1, 10, 20, 5, 30, 35, 13, 10, vander=np.polynomial.polynomial.polyvander, deg=3))

    def test_dec_solution_vector(self):
        assert np.allclose([490, 30, 10, 900, 350, 27000, 12250, 810000, 428750],
                           dec_sol_vec(1, 10, 20, 5, 30, 35, 13, 10, vander=np.polynomial.polynomial.polyvander, deg=3))

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
        agreement = np.isclose(expected_chi2_matrix,
                               chi2_matrix(1, 10, 20, 5, 30, 35, 13, 10, vander=np.polynomial.polynomial.polyvander, deg=3))

        assert np.all(agreement)

    @mock.patch('htof.fit.chi2_matrix', return_value=np.ones((9, 9)))
    def test_chi2_matrix_many_epoch(self, fake_chi2_matrix_per_epoch):
        ivar = np.ones((2, 2))
        fitter = AstrometricFitter(inverse_covariance_matrices=[ivar, ivar, ivar], epoch_times=[1, 2, 3],
                                   astrometric_solution_vector_components=[], use_parallax=True, fit_degree=3)
        assert np.allclose(np.ones((9, 9)) * 3, fitter._init_astrometric_chi_squared_matrix(3))
        fake_chi2_matrix_per_epoch.return_value = np.ones((7, 7))
        assert np.allclose(np.ones((7, 7)) * 3, fitter._init_astrometric_chi_squared_matrix(2))

    @mock.patch('htof.fit.ra_sol_vec', return_value=np.ones(5))
    @mock.patch('htof.fit.dec_sol_vec', return_value=np.ones(5))
    def test_chi2_vector(self, mock_ra_vec, mock_dec_vec):
        covariance_matrix = np.array([[5, 1], [12, 2]])
        expected_c = 2 * np.ones(4)
        fitter = AstrometricFitter(inverse_covariance_matrices=np.array([np.linalg.pinv(covariance_matrix)]),
                                   epoch_times=np.array([1]), astrometric_chi_squared_matrices=[],
                                   fit_degree=1, use_parallax=False)
        assert np.allclose(expected_c, fitter._chi2_vector(ra_vs_epoch=np.array([1]),
                                                           dec_vs_epoch=np.array([1])))

    def test_fitting_to_linear_astrometric_data(self):
        astrometric_data = generate_astrometric_data()
        fitter = AstrometricFitter(inverse_covariance_matrices=astrometric_data['inverse_covariance_matrix'],
                                   epoch_times=astrometric_data['epoch_delta_t'])
        assert np.allclose(fitter.fit_line(astrometric_data['ra'], astrometric_data['dec']),
                           astrometric_data['linear_solution'])

    def test_errors_on_linear_astrometric_data(self):
        astrometric_data = generate_astrometric_data()
        fitter = AstrometricFitter(inverse_covariance_matrices=astrometric_data['inverse_covariance_matrix'],
                                   epoch_times=astrometric_data['epoch_delta_t'])
        sol, errs, chisq = fitter.fit_line(astrometric_data['ra'], astrometric_data['dec'], return_all=True)
        assert errs.size == 4

    def test_fitting_with_nonzero_central_epoch(self):
        ra_cnt = np.random.randint(1, 100)
        dec_cnt = np.random.randint(1, 100)
        astrometric_data = generate_astrometric_data()
        expected_vec = astrometric_data['linear_solution']
        expected_vec[0] += ra_cnt * expected_vec[2]  # r0 = ra_central_time * mu_ra
        expected_vec[1] += dec_cnt * expected_vec[3]  # dec0 = dec_central_time * mu_dec
        fitter = AstrometricFitter(inverse_covariance_matrices=astrometric_data['inverse_covariance_matrix'],
                                   epoch_times=astrometric_data['epoch_delta_t'],
                                   central_epoch_dec=dec_cnt, central_epoch_ra=ra_cnt)
        assert np.allclose(fitter.fit_line(astrometric_data['ra'], astrometric_data['dec']), expected_vec)

    def test_fitting_to_cubic_astrometric_data_without_parallax(self):
        astrometric_data = generate_astrometric_data(acc=True, jerk=True)
        fitter = AstrometricFitter(inverse_covariance_matrices=astrometric_data['inverse_covariance_matrix'],
                                   epoch_times=astrometric_data['epoch_delta_t'], use_parallax=False, fit_degree=3,
                                   )
        assert np.allclose(fitter.fit_line(astrometric_data['ra'], astrometric_data['dec']),
                           astrometric_data['nonlinear_solution'])

    def test_fitting_to_linear_astrometric_data_with_parallax(self):
        real_plx = 100
        astrometric_data = generate_astrometric_data(acc=False, jerk=False)
        jyear_epochs = Time(astrometric_data['epoch_delta_t'] + 2448090, format='jd').jyear
        ra_pert, dec_pert = parallactic_motion(jyear_epochs, 45, 45, 'degree', 1991.25, parallax=1)
        t = astrometric_data['epoch_delta_t']
        astrometric_data['dec'] += dec_pert * real_plx
        astrometric_data['ra'] += ra_pert * real_plx
        fitter = AstrometricFitter(inverse_covariance_matrices=astrometric_data['inverse_covariance_matrix'],
                                   epoch_times=astrometric_data['epoch_delta_t'], use_parallax=True,
                                   parallactic_pertubations=[ra_pert, dec_pert], fit_degree=1)
        fit = fitter.fit_line(astrometric_data['ra'], astrometric_data['dec'])
        best_fit_dec = np.polynomial.polynomial.polyval(t, fit[1:][1::2]) + fit[0] * dec_pert
        best_fit_ra = np.polynomial.polynomial.polyval(t, fit[1:][::2]) + fit[0] * ra_pert
        assert np.isclose(fit[0], real_plx)
        assert np.allclose(best_fit_dec, astrometric_data['dec'])
        assert np.allclose(best_fit_ra, astrometric_data['ra'])

    def test_fitting_to_cubic_astrometric_data_with_parallax(self):
        real_plx = 100
        astrometric_data = generate_astrometric_data(acc=True, jerk=True)
        jyear_epochs = Time(astrometric_data['epoch_delta_t'] + 2448090, format='jd').jyear
        ra_pert, dec_pert = parallactic_motion(jyear_epochs, 45, 45, 'degree', 1991.25, parallax=1)
        t = astrometric_data['epoch_delta_t']
        astrometric_data['dec'] += dec_pert * real_plx
        astrometric_data['ra'] += ra_pert * real_plx
        fitter = AstrometricFitter(inverse_covariance_matrices=astrometric_data['inverse_covariance_matrix'],
                                   epoch_times=astrometric_data['epoch_delta_t'], use_parallax=True,
                                   parallactic_pertubations=[ra_pert, dec_pert], fit_degree=3)
        fit = fitter.fit_line(astrometric_data['ra'], astrometric_data['dec'])
        best_fit_dec = np.polynomial.polynomial.polyval(t, fit[1:][1::2]) + fit[0] * dec_pert
        best_fit_ra = np.polynomial.polynomial.polyval(t, fit[1:][::2]) + fit[0] * ra_pert
        assert np.isclose(fit[0], real_plx)
        assert np.allclose(best_fit_dec, astrometric_data['dec'])
        assert np.allclose(best_fit_ra, astrometric_data['ra'])

    def test_fitter_removes_parallax(self):
        astrometric_data = generate_astrometric_data()
        fitter = AstrometricFitter(inverse_covariance_matrices=astrometric_data['inverse_covariance_matrix'],
                                   epoch_times=astrometric_data['epoch_delta_t'], use_parallax=False, fit_degree=3)
        assert fitter._chi2_matrix.shape == (8, 8)
        assert fitter.astrometric_solution_vector_components['ra'][0].shape == (8,)
        assert fitter.astrometric_solution_vector_components['dec'][0].shape == (8,)


def test_unpack_elements_of_matrix():
    A = np.array([[0, 1], [2, 3]])
    assert np.allclose(np.arange(4), unpack_elements_of_matrix(A))


def test_transforming_from_unnormalized_domain():
    normed_coeffs = [50, 1, 1.1, .3, .4, .5, .6, .07, .08]  # parallax, ra0, dec0, mura, mudec, ara, adec, jra, jdec
    x = np.arange(50)
    ra = np.polynomial.polynomial.polyval(normalize(x, (np.min(x), np.max(x))), normed_coeffs[1:][::2])
    dec = np.polynomial.polynomial.polyval(normalize(x, (np.min(x), np.max(x))), normed_coeffs[1:][1::2])
    coeffs = transform_coefficients_to_unnormalized_domain(normed_coeffs, np.min(x), np.max(x), np.min(x), np.max(x),
                                                           True)
    assert np.allclose(np.polynomial.polynomial.polyval(x, coeffs[1:][::2]), ra)
    assert np.allclose(np.polynomial.polynomial.polyval(x, coeffs[1:][1::2]), dec)
    coeffs = transform_coefficients_to_unnormalized_domain(normed_coeffs[1:], np.min(x), np.max(x), np.min(x), np.max(x),
                                                           False)
    assert np.allclose(np.polynomial.polynomial.polyval(x, coeffs[::2]), ra)
    assert np.allclose(np.polynomial.polynomial.polyval(x, coeffs[1::2]), dec)


def test_normalize():
    out = normalize(np.arange(9), (0, 8))
    assert out.min() == -1 and out.max() == 1

"""
Utility functions
"""


def generate_astrometric_data(acc=False, jerk=False):
    astrometric_data = {}
    num_measurements = 50
    mu_ra, mu_dec = 1E-7, 2E-7
    acc_ra, acc_dec = acc * 2E-10, acc * 1E-10
    jerk_ra, jerk_dec = jerk * 2E-13, jerk * 1E-13
    ra0, dec0 = 2E-7, 1E-7
    epoch_start = 0
    epoch_end = 1000
    t = np.linspace(epoch_start, epoch_end, num=num_measurements)
    astrometric_data['epoch_delta_t'] = t
    astrometric_data['dec'] = dec0 + t * mu_dec + acc_dec * t ** 2 + jerk_dec * t ** 3
    astrometric_data['ra'] = ra0 + t * mu_ra + acc_ra * t ** 2 + jerk_ra * t ** 3
    sigma_ra, sigma_dec, cc = 0.1, 0.1, 0.1
    astrometric_data['covariance_matrix'] = np.zeros((num_measurements, 2, 2))
    astrometric_data['inverse_covariance_matrix'] = np.zeros((num_measurements, 2, 2))
    astrometric_data['covariance_matrix'][:] = np.array([[sigma_ra**2, sigma_ra*sigma_dec*cc],
                                                       [sigma_ra*sigma_dec*cc, sigma_dec**2]])
    for i in range(num_measurements):
        astrometric_data['inverse_covariance_matrix'][i] = np.linalg.pinv(astrometric_data['covariance_matrix'][i])
    astrometric_data['linear_solution'] = np.array([ra0, dec0, mu_ra, mu_dec])
    astrometric_data['nonlinear_solution'] = np.array([ra0, dec0, mu_ra, mu_dec, acc_ra, acc_dec, jerk_ra, jerk_dec])

    return astrometric_data
