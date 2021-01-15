import pytest
import os
import numpy as np
from astropy.coordinates import Angle
from astropy.time import Time

from htof.parse import GaiaeDR3, GaiaData, HipparcosRereductionDVDBook, HipparcosOriginalData
from htof.fit import AstrometricFitter
from htof.validation.utils import refit_hip2_object, refit_hip1_object, load_hip2_catalog, load_hip2_seven_p_annex, load_hip2_nine_p_annex
from htof.validation.utils import load_hip1_dm_annex

from htof.main import Astrometry


@pytest.mark.integration
def test_parse_and_fit_to_line():
    """
    Tests fitting a line to fake RA and DEC data which has errors calculated from the real intermediate data
    from Hip1, Hip2, and GaiaDR2. This only fits a line to the first 11 points.
    """
    stars = ['049699', '027321', '027321']
    parsers = [GaiaeDR3, GaiaData, HipparcosOriginalData, HipparcosRereductionDVDBook]
    subdirectories = ['GaiaeDR3', 'GaiaDR2', 'Hip1', 'Hip2']
    base_directory = os.path.join(os.getcwd(), 'htof/test/data_for_tests')
    for star, parser, subdirectory in zip(stars, parsers, subdirectories):
        test_data_directory = os.path.join(base_directory, subdirectory)
        data = parser()
        data.parse(star_id=star,
                   intermediate_data_directory=test_data_directory)
        data.calculate_inverse_covariance_matrices()
        fitter = AstrometricFitter(inverse_covariance_matrices=data.inverse_covariance_matrix,
                                   epoch_times=np.linspace(0, 10, num=11))
        solution_vector = fitter.fit_line(ra_vs_epoch=np.linspace(30, 40, num=11),
                                          dec_vs_epoch=np.linspace(20, 30, num=11))
        ra0, dec0, mu_ra, mu_dec = solution_vector

        assert np.isclose(ra0, 30)
        assert np.isclose(dec0, 20)
        assert np.isclose(mu_ra, 1)
        assert np.isclose(mu_dec, 1)


class TestHipReReductionCDFits:
    CATALOG = load_hip2_catalog('htof/test/data_for_tests/Hip2/truncated_hip2dvd_Main_Cat.d')
    NINEP = load_hip2_nine_p_annex('htof/test/data_for_tests/Hip2/NineP_Cat.d')
    SEVENP = load_hip2_seven_p_annex('htof/test/data_for_tests/Hip2/SevenP_Cat.d')

    @pytest.mark.e2e
    @pytest.mark.parametrize("hip_id", ['70', '78999', '27321'])
    def test_Hip2_fit_5p_source(self, hip_id):
        diffs, error_diffs, chisq, chi2_partials, soltype = refit_hip2_object('htof/test/data_for_tests/Hip2',
                                                                         hip_id, catalog=self.CATALOG, use_parallax=True)
        assert np.allclose(diffs, 0, atol=0.02)
        assert np.allclose(error_diffs, 0, atol=0.1)

    @pytest.mark.e2e
    @pytest.mark.parametrize("hip_id", ['9631', '16468', '25838'])
    def test_Hip2_fit_7p9p_source(self, hip_id):
        diffs, error_diffs, chisq, chi2_partials, soltype = refit_hip2_object('htof/test/data_for_tests/Hip2/IntermediateData', hip_id,
                                                                              nine_p_annex=self.NINEP, seven_p_annex=self.SEVENP,
                                                                              catalog=self.CATALOG, use_parallax=True)
        assert np.allclose(diffs, 0, atol=0.02)
        # TODO test the error differences!
        #assert np.allclose(error_diffs, 0, atol=0.1)


class TestHip1Fits:
    SEVEN_NINEP_ANNEX = load_hip1_dm_annex('htof/test/data_for_tests/Hip1/hip_dm_g.dat')

    @pytest.mark.e2e
    @pytest.mark.parametrize("hip_id", ['5310', '5313', '46871', '50103', '46979'])
    def test_Hip1_fit_to_hip7p9p_source(self, hip_id):
        diffs, errors, chisq, chi2_partials, soltype = refit_hip1_object('htof/test/data_for_tests/Hip1/IntermediateData', hip_id,
                                                                         hip_dm_g=self.SEVEN_NINEP_ANNEX,
                                                                         use_parallax=True)
        assert np.allclose(diffs, 0, atol=0.07)

    @pytest.mark.e2e
    @pytest.mark.parametrize("hip_id", ['027321', '004391', '044801', '70000'])
    def test_Hip1_fit(self, hip_id):
        diffs, errors, chisq, chi2_partials, soltype = refit_hip1_object('htof/test/data_for_tests/Hip1/IntermediateData', hip_id,
                                                                         hip_dm_g=self.SEVEN_NINEP_ANNEX,
                                                                         use_parallax=True)
        assert np.allclose(diffs, 0, atol=0.02)


@pytest.mark.e2e
def test_Hip1_fit_to_hip27321():
    # Hip 27321 parameters from the Hipparcos 1 catalogue via Vizier
    cntr_ra, cntr_dec = Angle(86.82118054, 'degree'), Angle(-51.06671341, 'degree')
    plx = 51.87  # mas
    pmRA = 4.65  # mas/year
    pmDec = 81.96  # mas/year
    # generate fitter and parse intermediate data
    astro = Astrometry('Hip1', '27321', 'htof/test/data_for_tests/Hip1', central_epoch_ra=1991.25,
                       central_epoch_dec=1991.25, format='jyear', fit_degree=1, use_parallax=True,
                       central_ra=cntr_ra, central_dec=cntr_dec, normed=False)
    chisq = np.sum(astro.data.residuals ** 2 / astro.data.along_scan_errs ** 2)
    # generate ra and dec for each observation.
    year_epochs = Time(astro.data.julian_day_epoch(), format='jd', scale='tcb').jyear - \
                  Time(1991.25, format='decimalyear').jyear
    ra_motion = astro.fitter.parallactic_pertubations['ra_plx']
    dec_motion = astro.fitter.parallactic_pertubations['dec_plx']
    ra = Angle(ra_motion * plx + pmRA * year_epochs, unit='mas')
    dec = Angle(dec_motion * plx + pmDec * year_epochs, unit='mas')
    # add residuals
    ra += Angle(astro.data.residuals.values * np.sin(astro.data.scan_angle.values), unit='mas')
    dec += Angle(astro.data.residuals.values * np.cos(astro.data.scan_angle.values), unit='mas')
    #
    coeffs, errors, chisq_found = astro.fit(ra.mas, dec.mas, return_all=True)
    assert np.isclose(chisq, chisq_found, atol=1E-3)
    assert np.allclose([pmRA, pmDec], np.array([coeffs[3], coeffs[4]]).round(2))
    assert np.isclose(plx, coeffs[0].round(2), atol=0.01)
    assert np.allclose(errors.round(2), np.array([0.51, 0.45, 0.46, 0.53, 0.61]))


@pytest.mark.e2e
def test_Hip1_fit_to_hip44801():
    # Hip 44801 parameters from the intermediate data
    cntr_ra, cntr_dec = Angle(136.94995265, 'degree'), Angle(-9.85368471, 'degree')
    plx = 2.90  # mas
    pmRA = -10.91  # mas/year
    pmDec = 5.06  # mas/year
    # generate fitter and parse intermediate data
    astro = Astrometry('Hip1', '44801', 'htof/test/data_for_tests/Hip1', central_epoch_ra=1991.25,
                       central_epoch_dec=1991.25, format='jyear', fit_degree=1, use_parallax=True,
                       central_ra=cntr_ra, central_dec=cntr_dec, normed=False)
    chisq = np.sum(astro.data.residuals ** 2 / astro.data.along_scan_errs ** 2)
    # generate ra and dec for each observation.
    year_epochs = Time(astro.data.julian_day_epoch(), format='jd', scale='tcb').jyear - \
                  Time(1991.25, format='decimalyear').jyear
    ra_motion = astro.fitter.parallactic_pertubations['ra_plx']
    dec_motion = astro.fitter.parallactic_pertubations['dec_plx']
    ra = Angle(ra_motion * plx + pmRA * year_epochs, unit='mas')
    dec = Angle(dec_motion * plx + pmDec * year_epochs, unit='mas')
    # add residuals
    ra += Angle(astro.data.residuals.values * np.sin(astro.data.scan_angle.values), unit='mas')
    dec += Angle(astro.data.residuals.values * np.cos(astro.data.scan_angle.values), unit='mas')
    #
    coeffs, errors, chisq_found = astro.fit(ra.mas, dec.mas, return_all=True)
    assert np.isclose(chisq, chisq_found, atol=1E-3)
    assert np.allclose([pmRA, pmDec], np.array([coeffs[3], coeffs[4]]).round(2))
    assert np.isclose(plx, coeffs[0].round(2), atol=0.01)
    assert np.allclose(errors.round(2), np.array([1.09, 0.88, 0.77, 1.05, 0.80]))


@pytest.mark.e2e
def test_Hip1_fit_to_hip70000():
    # Hip 70000 parameters from the intermediate data
    cntr_ra, cntr_dec = Angle(214.85975459, 'degree'), Angle(14.93570946, 'degree')
    plx = 1.26  # mas
    pmRA = 1.01  # mas/year
    pmDec = 7.27  # mas/year
    # generate fitter and parse intermediate data
    astro = Astrometry('Hip1', '70000', 'htof/test/data_for_tests/Hip1', central_epoch_ra=1991.25,
                       central_epoch_dec=1991.25, format='jyear', fit_degree=1, use_parallax=True,
                       central_ra=cntr_ra, central_dec=cntr_dec, normed=False)
    chisq = np.sum(astro.data.residuals ** 2 / astro.data.along_scan_errs ** 2)
    # generate ra and dec for each observation.
    year_epochs = Time(astro.data.julian_day_epoch(), format='jd', scale='tcb').jyear - \
                  Time(1991.25, format='decimalyear').jyear
    ra_motion = astro.fitter.parallactic_pertubations['ra_plx']
    dec_motion = astro.fitter.parallactic_pertubations['dec_plx']
    ra = Angle(ra_motion * plx + pmRA * year_epochs, unit='mas')
    dec = Angle(dec_motion * plx + pmDec * year_epochs, unit='mas')
    # add residuals
    ra += Angle(astro.data.residuals.values * np.sin(astro.data.scan_angle.values), unit='mas')
    dec += Angle(astro.data.residuals.values * np.cos(astro.data.scan_angle.values), unit='mas')
    #
    coeffs, errors, chisq_found = astro.fit(ra.mas, dec.mas, return_all=True)
    assert np.isclose(chisq, chisq_found, atol=1E-3)
    assert np.allclose([pmRA, pmDec], np.array([coeffs[3], coeffs[4]]).round(2))
    assert np.isclose(plx, coeffs[0].round(2), atol=0.01)
    assert np.allclose(errors.round(2), np.array([1.11, 0.79, 0.62, 0.82, 0.64]))


@pytest.mark.e2e
def test_Hip1_fit_to_hip27321_no_parallax():
    # WARNING including the parallax component is important if you want to recover the catalog errors.
    # Hip 27321 parameters from the Hipparcos 1 catalogue via Vizier
    pmRA = 4.65  # mas/year
    pmDec = 81.96  # mas/year
    # generate fitter and parse intermediate data
    astro = Astrometry('Hip1', '27321', 'htof/test/data_for_tests/Hip1', central_epoch_ra=1991.25,
                       central_epoch_dec=1991.25, format='jyear', fit_degree=1,
                       use_parallax=False, normed=False)
    chisq = np.sum(astro.data.residuals ** 2 / astro.data.along_scan_errs ** 2)
    # generate ra and dec for each observation.
    year_epochs = Time(astro.data.julian_day_epoch(), format='jd', scale='tcb').jyear - \
                  Time(1991.25, format='decimalyear').jyear
    ra = Angle(pmRA * year_epochs, unit='mas')
    dec = Angle(pmDec * year_epochs, unit='mas')
    # add residuals
    ra += Angle(astro.data.residuals.values * np.sin(astro.data.scan_angle.values), unit='mas')
    dec += Angle(astro.data.residuals.values * np.cos(astro.data.scan_angle.values), unit='mas')
    #
    coeffs, errors, chisq_found = astro.fit(ra.mas, dec.mas, return_all=True)
    assert np.isclose(chisq, chisq_found, atol=1E-3)
    assert np.allclose([pmRA, pmDec], np.array([coeffs[2], coeffs[3]]).round(2))
    assert np.allclose(errors.round(2), np.array([0.45, 0.46, 0.53, 0.61]), atol=0.01)


@pytest.mark.e2e
def test_optimal_central_epochs_forHip1_hip27321():
    # Hip 27321 parameters from the Hipparcos 1 catalogue via Vizier
    cntr_ra, cntr_dec = Angle(86.82118054, 'degree'), Angle(-51.06671341, 'degree')
    # generate fitter and parse intermediate data
    astro = Astrometry('Hip1', '27321', 'htof/test/data_for_tests/Hip1', central_epoch_ra=1991.25,
                       central_epoch_dec=1991.25, format='jyear', fit_degree=1, use_parallax=True,
                       central_ra=cntr_ra, central_dec=cntr_dec, normed=False)
    central_epoch = astro.optimal_central_epochs()
    central_epoch_ra, central_epoch_dec = central_epoch['ra'], central_epoch['dec']
    #print(central_epoch_ra, central_epoch_dec)
    fitter = astro.fitter
    cov_matrix = fitter.evaluate_cov_matrix(central_epoch_ra, central_epoch_dec)
    ra_mura_cov, dec_mudec_cov = cov_matrix[1, 3], cov_matrix[2, 4]
    # do a brute force evaluation of all reasonable central epochs
    epoch_t = np.linspace(1991, 1992, 200)
    ra_vals, dec_vals = [], []
    for t in epoch_t:
        cov = fitter.evaluate_cov_matrix(t, t)
        ra_vals.append(cov[1, 3])
        dec_vals.append(cov[2, 4])
    # assert that the optimal central epochs give better covariances than all of those.
    assert np.all(np.abs(dec_vals) >= dec_mudec_cov)
    assert np.all(np.abs(ra_vals) >= ra_mura_cov)
    assert np.allclose([dec_mudec_cov, ra_mura_cov], 0, atol=1e-8)


@pytest.mark.e2e
def test_optimal_central_epochs_forHip1_hip27321_no_parallax():
    astro = Astrometry('Hip1', '27321', 'htof/test/data_for_tests/Hip1', central_epoch_ra=1991.25,
                       central_epoch_dec=1991.25, format='jyear', fit_degree=1,
                       use_parallax=False, normed=False)
    central_epoch = astro.optimal_central_epochs()
    central_epoch_ra, central_epoch_dec = central_epoch['ra'], central_epoch['dec']
    #print(central_epoch_ra, central_epoch_dec)
    fitter = astro.fitter
    cov_matrix = fitter.evaluate_cov_matrix(central_epoch_ra, central_epoch_dec)
    ra_mura_cov, dec_mudec_cov = cov_matrix[0, 2], cov_matrix[1, 3]
    # do a brute force evaluation of all reasonable central epochs
    epoch_t = np.linspace(1991, 1992, 200)
    ra_vals, dec_vals = [], []
    for t in epoch_t:
        cov = fitter.evaluate_cov_matrix(t, t)
        ra_vals.append(cov[0, 2])
        dec_vals.append(cov[1, 3])
    # assert that the optimal central epochs give better covariances than all of those.
    assert np.all(np.abs(dec_vals) >= dec_mudec_cov)
    assert np.all(np.abs(ra_vals) >= ra_mura_cov)
    assert np.allclose([dec_mudec_cov, ra_mura_cov], 0, atol=1e-9)
