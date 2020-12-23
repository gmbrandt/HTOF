import numpy as np
import mock
from astropy.time import Time
import pytest
import timeit
from astropy.coordinates import Angle

from htof.fit import unpack_elements_of_matrix, AstrometricFitter, normalize, AstrometricFastFitter
from htof.utils.fit_utils import ra_sol_vec, dec_sol_vec, chi2_matrix, transform_coefficients_to_unnormalized_domain
from htof.sky_path import parallactic_motion


class TestAstrometricFitter:
    def test_init_epochs_no_norm(self):
        fitter = AstrometricFitter(astrometric_solution_vector_components=[], central_epoch_dec=1.5, central_epoch_ra=1,
                                   epoch_times=np.array([1, 2, 3]), astrometric_chi_squared_matrices=[],
                                   fit_degree=1, use_parallax=False, normed=False)
        assert np.allclose(fitter.dec_epochs, [-0.5, 0.5, 1.5])
        assert np.allclose(fitter.ra_epochs, [0, 1, 2])

    def test_init_epochs(self):
        fitter = AstrometricFitter(astrometric_solution_vector_components=[], central_epoch_dec=1.5, central_epoch_ra=1,
                                   epoch_times=np.array([1, 2, 3]), astrometric_chi_squared_matrices=[],
                                   fit_degree=1, use_parallax=False, normed=True)
        assert np.allclose(fitter.dec_epochs, [-1, 0, 1])
        assert np.allclose(fitter.ra_epochs, [-1, 0, 1])

    def test_ra_solution_vector(self):
        assert np.allclose([326, 2, 30, 60, 1050, 1800, 36750, 54000, 1286250],
                           2*ra_sol_vec(1, 10, 20, 5, 30, 35, 13, 10, vander=np.polynomial.polynomial.polyvander, deg=3))

    def test_dec_solution_vector(self):
        assert np.allclose([490, 30, 10, 900, 350, 27000, 12250, 810000, 428750],
                           2*dec_sol_vec(1, 10, 20, 5, 30, 35, 13, 10, vander=np.polynomial.polynomial.polyvander, deg=3))

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
        ])/2
        agreement = np.isclose(expected_chi2_matrix,
                               chi2_matrix(1, 10, 20, 5, 30, 35, 13, 10, vander=np.polynomial.polynomial.polyvander, deg=3))

        assert np.all(agreement)

    @mock.patch('htof.fit.chi2_matrix', return_value=np.ones((9, 9)))
    def test_chi2_matrix_many_epoch(self, fake_chi2_matrix_per_epoch):
        ivar = np.ones((2, 2))
        fitter = AstrometricFitter(inverse_covariance_matrices=[ivar, ivar, ivar], epoch_times=[1, 2, 3],
                                   astrometric_solution_vector_components=[], use_parallax=True, fit_degree=3)
        assert np.allclose(np.ones((9, 9)) * 3, fitter._init_astrometric_chi_squared_matrix(3)[0])
        fake_chi2_matrix_per_epoch.return_value = np.ones((7, 7))
        assert np.allclose(np.ones((7, 7)) * 3, fitter._init_astrometric_chi_squared_matrix(2)[0])

    @mock.patch('htof.fit.ra_sol_vec', return_value=np.ones(5))
    @mock.patch('htof.fit.dec_sol_vec', return_value=np.ones(5))
    def test_chi2_vector(self, mock_ra_vec, mock_dec_vec):
        covariance_matrix = np.array([[5, 1], [12, 2]])
        expected_c = 4 * np.ones(4)
        fitter = AstrometricFitter(inverse_covariance_matrices=np.array([np.linalg.pinv(covariance_matrix),
                                                                         np.linalg.pinv(covariance_matrix)]),
                                   epoch_times=np.array([1, 2]), astrometric_chi_squared_matrices=[],
                                   fit_degree=1, use_parallax=False)
        assert np.allclose(expected_c, fitter._chi2_vector(ra_vs_epoch=np.array([1, 1]),
                                                           dec_vs_epoch=np.array([1, 1])))

    @pytest.mark.parametrize('fitter_class', [AstrometricFitter, AstrometricFastFitter])
    def test_fitting_to_linear_astrometric_data(self, fitter_class):
        astrometric_data = generate_astrometric_data()
        fitter = fitter_class(inverse_covariance_matrices=astrometric_data['inverse_covariance_matrix'],
                              epoch_times=astrometric_data['epoch_delta_t'])
        assert np.allclose(fitter.fit_line(astrometric_data['ra'], astrometric_data['dec']),
                           astrometric_data['linear_solution'])

    def test_optimal_central_epoch_on_linear_data(self):
        astrometric_data = generate_astrometric_data()
        fitter = AstrometricFitter(inverse_covariance_matrices=astrometric_data['inverse_covariance_matrix'],
                                   epoch_times=astrometric_data['epoch_delta_t'])
        central_epoch_ra, central_epoch_dec = fitter.find_optimal_central_epoch('ra'), fitter.find_optimal_central_epoch('dec')
        cov_matrix = fitter.evaluate_cov_matrix(central_epoch_ra, central_epoch_dec)
        assert np.allclose([cov_matrix[0, 2], cov_matrix[1, 3]], 0)

    def test_errors_on_linear_astrometric_data(self):
        astrometric_data = generate_astrometric_data()
        fitter = AstrometricFitter(inverse_covariance_matrices=astrometric_data['inverse_covariance_matrix'],
                                   epoch_times=astrometric_data['epoch_delta_t'],
                                   normed=False)
        sol, errs, chisq = fitter.fit_line(astrometric_data['ra'], astrometric_data['dec'], return_all=True)
        assert errs.size == 4
        assert np.isclose(0, chisq, atol=1e-7)

    def test_fitting_with_nonzero_central_epoch(self):
        ra_cnt = np.random.randint(1, 100)
        dec_cnt = np.random.randint(1, 100)
        astrometric_data = generate_astrometric_data()
        expected_vec = astrometric_data['linear_solution']
        expected_vec[0] += ra_cnt * expected_vec[2]  # r0 = ra_central_time * mu_ra
        expected_vec[1] += dec_cnt * expected_vec[3]  # dec0 = dec_central_time * mu_dec
        fitter = AstrometricFitter(inverse_covariance_matrices=astrometric_data['inverse_covariance_matrix'],
                              epoch_times=astrometric_data['epoch_delta_t'],
                              central_epoch_dec=dec_cnt, central_epoch_ra=ra_cnt, normed=True)
        assert np.allclose(fitter.fit_line(astrometric_data['ra'], astrometric_data['dec']), expected_vec)

    def test_fitting_with_normalization(self):
        ra_cnt = np.random.randint(1, 100)
        dec_cnt = np.random.randint(1, 100)
        astrometric_data = generate_astrometric_data()
        expected_vec = astrometric_data['linear_solution']
        expected_vec[0] += ra_cnt * expected_vec[2]  # r0 = ra_central_time * mu_ra
        expected_vec[1] += dec_cnt * expected_vec[3]  # dec0 = dec_central_time * mu_dec
        fitter = AstrometricFitter(inverse_covariance_matrices=astrometric_data['inverse_covariance_matrix'],
                              epoch_times=astrometric_data['epoch_delta_t'],
                              central_epoch_dec=dec_cnt, central_epoch_ra=ra_cnt, normed=True)
        assert np.allclose(fitter.fit_line(astrometric_data['ra'], astrometric_data['dec']), expected_vec)

    @pytest.mark.parametrize('fitter_class', [AstrometricFitter, AstrometricFastFitter])
    def test_fitting_to_cubic_astrometric_data(self, fitter_class):
        astrometric_data = generate_astrometric_data(acc=True, jerk=True)
        fitter = fitter_class(inverse_covariance_matrices=astrometric_data['inverse_covariance_matrix'],
                              epoch_times=astrometric_data['epoch_delta_t'], use_parallax=False, fit_degree=3,
                              normed=False)
        assert np.allclose(fitter.fit_line(astrometric_data['ra'], astrometric_data['dec']),
                           astrometric_data['nonlinear_solution'], atol=0, rtol=1E-4)

    def test_optimal_central_epoch_on_cubic_data(self):
        astrometric_data = generate_astrometric_data(acc=True, jerk=True)
        fitter = AstrometricFitter(inverse_covariance_matrices=astrometric_data['inverse_covariance_matrix'],
                                   epoch_times=astrometric_data['epoch_delta_t'], use_parallax=False, fit_degree=3,
                                   normed=False)
        central_epoch_ra, central_epoch_dec = fitter.find_optimal_central_epoch('ra'), fitter.find_optimal_central_epoch('dec')
        cov_matrix = fitter.evaluate_cov_matrix(central_epoch_ra, central_epoch_dec)
        assert np.allclose([cov_matrix[0, 2], cov_matrix[1, 3]], 0)

    def test_fitting_to_linear_astrometric_data_with_parallax(self):
        real_plx = 100
        cntr_dec, cntr_ra = Angle(45, unit='degree'), Angle(45, unit='degree')
        astrometric_data = generate_astrometric_data(acc=False, jerk=False)
        jyear_epochs = Time(astrometric_data['epoch_delta_t'] + 2012, format='decimalyear').jyear
        ra_pert, dec_pert = parallactic_motion(jyear_epochs, cntr_dec.mas, cntr_ra.mas, 'mas', 2012, parallax=1)
        astrometric_data['dec'] += dec_pert * real_plx
        astrometric_data['ra'] += ra_pert * real_plx
        fitter = AstrometricFitter(inverse_covariance_matrices=astrometric_data['inverse_covariance_matrix'],
                                   epoch_times=astrometric_data['epoch_delta_t'], use_parallax=True,
                                   parallactic_pertubations={'ra_plx': ra_pert, 'dec_plx': dec_pert},
                                   fit_degree=1, normed=False)
        solution, errors, chisq = fitter.fit_line(astrometric_data['ra'], astrometric_data['dec'], return_all=True)
        assert np.allclose(solution[1:], astrometric_data['linear_solution'], atol=0, rtol=1E-6)
        assert np.allclose(solution[0], real_plx)

    def test_fitting_to_cubic_astrometric_data_with_parallax(self):
        real_plx = 100
        cntr_dec, cntr_ra = Angle(45, unit='degree'), Angle(45, unit='degree')
        astrometric_data = generate_astrometric_data(acc=True, jerk=True)
        jyear_epochs = Time(astrometric_data['epoch_delta_t'] + 2012, format='decimalyear').jyear
        ra_pert, dec_pert = parallactic_motion(jyear_epochs, cntr_dec.mas, cntr_ra.mas, 'mas', 2012, parallax=1)
        astrometric_data['dec'] += dec_pert * real_plx
        astrometric_data['ra'] += ra_pert * real_plx
        fitter = AstrometricFitter(inverse_covariance_matrices=astrometric_data['inverse_covariance_matrix'],
                                   epoch_times=astrometric_data['epoch_delta_t'], use_parallax=True,
                                   parallactic_pertubations={'ra_plx': ra_pert, 'dec_plx': dec_pert},
                                   fit_degree=3, normed=False)
        solution, errors, chisq = fitter.fit_line(astrometric_data['ra'], astrometric_data['dec'], return_all=True)
        assert np.allclose(solution[1:], astrometric_data['nonlinear_solution'], atol=0, rtol=1E-4)
        assert np.allclose(solution[0], real_plx)

    def test_solutions_equal_on_normed_and_unnormed(self):
        real_plx = 100
        cntr_dec, cntr_ra = Angle(45, unit='degree'), Angle(45, unit='degree')
        astrometric_data = generate_astrometric_data(acc=True, jerk=True)
        jyear_epochs = Time(astrometric_data['epoch_delta_t'] + 2012, format='decimalyear').jyear
        ra_pert, dec_pert = parallactic_motion(jyear_epochs, cntr_dec.mas, cntr_ra.mas, 'mas', 2012, parallax=1)
        astrometric_data['dec'] += dec_pert * real_plx
        astrometric_data['ra'] += ra_pert * real_plx
        fitters = []
        for normed in [True, False]:
            fitters.append(AstrometricFitter(inverse_covariance_matrices=astrometric_data['inverse_covariance_matrix'],
                                             epoch_times=astrometric_data['epoch_delta_t'], use_parallax=True,
                                             parallactic_pertubations={'ra_plx': ra_pert, 'dec_plx': dec_pert},
                                             fit_degree=3, normed=normed))
        solution1, errors1, chisq1 = fitters[0].fit_line(astrometric_data['ra'], astrometric_data['dec'], return_all=True)
        solution2, errors2, chisq2 = fitters[1].fit_line(astrometric_data['ra'], astrometric_data['dec'], return_all=True)
        assert np.allclose(errors2, errors1)
        assert np.isclose(chisq1, chisq2)
        assert np.allclose(solution1, solution2)

    def test_fitter_removes_parallax(self):
        astrometric_data = generate_astrometric_data()
        fitter = AstrometricFitter(inverse_covariance_matrices=astrometric_data['inverse_covariance_matrix'],
                                   epoch_times=astrometric_data['epoch_delta_t'], use_parallax=False, fit_degree=3)
        assert fitter._chi2_matrix.shape == (8, 8)
        assert fitter.astrometric_solution_vector_components['ra'][0].shape == (8,)
        assert fitter.astrometric_solution_vector_components['dec'][0].shape == (8,)


def test_fast_fitter_raises_on_normed():
    astrometric_data = generate_astrometric_data()
    with pytest.raises(NotImplementedError):
        fitter = AstrometricFastFitter(inverse_covariance_matrices=astrometric_data['inverse_covariance_matrix'],
                                   epoch_times=astrometric_data['epoch_delta_t'], use_parallax=False, fit_degree=3,
                                   normed=True)

def test_timing_of_fast_fitter():
    astrometric_data = generate_astrometric_data(acc=True, jerk=True)
    fitter = AstrometricFastFitter(inverse_covariance_matrices=astrometric_data['inverse_covariance_matrix'],
                                   epoch_times=astrometric_data['epoch_delta_t'], use_parallax=False, fit_degree=3,
                                   normed=False)
    assert np.allclose(fitter.fit_line(astrometric_data['ra'], astrometric_data['dec']),
                       astrometric_data['nonlinear_solution'], atol=0, rtol=1E-4)
    t = timeit.Timer(lambda: fitter.fit_line(astrometric_data['ra'], astrometric_data['dec']))
    num = int(1E4)
    runtime = t.timeit(number=num) / num * 1E6
    assert runtime < 10  # assert that the fast fitter fit_line time is less than 10 microseconds.


def test_unpack_elements_of_matrix():
    A = np.array([[0, 1], [2, 3]])
    assert np.allclose(np.arange(4), unpack_elements_of_matrix(A))


def test_transforming_from_unnormalized_domain():
    normed_coeffs = [50, 1, 1.1, .3, .4, .5, .6, .07, .08]  # parallax, ra0, dec0, mura, mudec, ara, adec, jra, jdec
    x = np.arange(50)
    ra = np.polynomial.polynomial.polyval(normalize(x, (np.min(x), np.max(x))), normed_coeffs[1:][::2])
    dec = np.polynomial.polynomial.polyval(normalize(x, (np.min(x), np.max(x))), normed_coeffs[1:][1::2])
    coeffs = transform_coefficients_to_unnormalized_domain(normed_coeffs, np.min(x), np.max(x), np.min(x), np.max(x),
                                                           True, basis=np.polynomial.polynomial.Polynomial)
    assert np.allclose(np.polynomial.polynomial.polyval(x, coeffs[1:][::2]), ra)
    assert np.allclose(np.polynomial.polynomial.polyval(x, coeffs[1:][1::2]), dec)
    coeffs = transform_coefficients_to_unnormalized_domain(normed_coeffs[1:], np.min(x), np.max(x), np.min(x), np.max(x),
                                                           False, basis=np.polynomial.polynomial.Polynomial)
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
    num_measurements = 20
    mu_ra, mu_dec = 20, 30
    acc_ra, acc_dec = acc * 5, acc * 10
    jerk_ra, jerk_dec = jerk * 2, jerk * 1
    ra0, dec0 = 1, 2
    epoch_start = -2
    epoch_end = 2
    t = np.linspace(epoch_start, epoch_end, num=num_measurements)
    astrometric_data['epoch_delta_t'] = t
    astrometric_data['dec'] = dec0 + t * mu_dec + 1/2*acc_dec * t ** 2 + 1/6*jerk_dec * t ** 3
    astrometric_data['ra'] = ra0 + t * mu_ra + 1/2*acc_ra * t ** 2 + 1/6*jerk_ra * t ** 3
    sigma_ra, sigma_dec, cc = 0.1, 0.1, 0
    astrometric_data['covariance_matrix'] = np.zeros((num_measurements, 2, 2))
    astrometric_data['inverse_covariance_matrix'] = np.zeros((num_measurements, 2, 2))
    astrometric_data['covariance_matrix'][:] = np.array([[sigma_ra**2, sigma_ra*sigma_dec*cc],
                                                       [sigma_ra*sigma_dec*cc, sigma_dec**2]])
    for i in range(num_measurements):
        astrometric_data['inverse_covariance_matrix'][i] = np.linalg.pinv(astrometric_data['covariance_matrix'][i])
    astrometric_data['linear_solution'] = np.array([ra0, dec0, mu_ra, mu_dec])
    astrometric_data['nonlinear_solution'] = np.array([ra0, dec0, mu_ra, mu_dec, acc_ra, acc_dec, jerk_ra, jerk_dec])

    return astrometric_data
