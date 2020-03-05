import pytest
import os
import numpy as np
from astropy.coordinates import Angle
from astropy.time import Time

from htof.parse import GaiaData, HipparcosRereductionData, HipparcosOriginalData
from htof.fit import AstrometricFitter

from htof.main import Astrometry


@pytest.mark.integration
def test_parse_and_fit_to_line():
    """
    Tests fitting a line to fake RA and DEC data which has errors calculated from the real intermediate data
    from Hip1, Hip2, and GaiaDR2. This only fits a line to the first 11 points.
    """
    stars = ['049699', '027321', '027321']
    parsers = [GaiaData, HipparcosOriginalData, HipparcosRereductionData]
    subdirectories = ['GaiaDR2', 'Hip1', 'Hip2']
    base_directory = os.path.join(os.getcwd(), 'htof/test/data_for_tests')
    for star, parser, subdirectory in zip(stars, parsers, subdirectories):
        test_data_directory = os.path.join(base_directory, subdirectory)
        data = parser()
        data.parse(star_id=star,
                   intermediate_data_directory=test_data_directory)
        data.calculate_inverse_covariance_matrices(cross_scan_along_scan_var_ratio=1E5)
        fitter = AstrometricFitter(inverse_covariance_matrices=data.inverse_covariance_matrix,
                                   epoch_times=np.linspace(0, 10, num=11))
        solution_vector = fitter.fit_line(ra_vs_epoch=np.linspace(30, 40, num=11),
                                          dec_vs_epoch=np.linspace(20, 30, num=11))
        ra0, dec0, mu_ra, mu_dec = solution_vector

        assert np.isclose(ra0, 30)
        assert np.isclose(dec0, 20)
        assert np.isclose(mu_ra, 1)
        assert np.isclose(mu_dec, 1)


@pytest.mark.e2e
def test_Hip2_fit_to_hip27321():
    # Hip 27321 parameters from Snellen+Brown 2018: https://arxiv.org/pdf/1808.06257.pdf
    cntr_ra, cntr_dec = Angle(86.82118054, 'degree'), Angle(-51.06671341, 'degree')
    plx = 51.44  # mas
    pmRA = 4.65  # mas/year
    pmDec = 83.10  # mas/year
    # generate fitter and parse intermediate data
    astro = Astrometry('Hip2', '27321', 'htof/test/data_for_tests/Hip2', central_epoch_ra=1991.25,
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
    assert np.allclose(errors.round(2), np.array([0.11, 0.10, 0.11, 0.11, 0.15]))


@pytest.mark.e2e
def test_hip2_fit_to_hip78999():
    cntr_ra, cntr_dec = Angle(241.89259265, 'degree'), Angle(-5.70677966, 'degree')
    plx = 28.11  # mas
    pmRA = 156.38  # mas/year
    pmDec = -177.64  # mas/year
    # generate fitter and parse intermediate data
    astro = Astrometry('Hip2', '78999', 'htof/test/data_for_tests/Hip2', central_epoch_ra=1991.25,
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
    assert np.allclose([pmRA, pmDec], np.array([coeffs[3], coeffs[4]]).round(2), atol=0.01)
    assert np.isclose(plx, coeffs[0].round(2), atol=0.01)
    # testing the errors. Note these values must come from the CD if we are testing IAD from the CD.
    assert np.allclose(errors.round(2), np.array([2.4, 1.79, 0.94, 4.05, 2.2]))


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
def test_Hip1_fit_to_hip27321_no_parallax():
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
