import pytest
import os
import numpy as np

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
    for star, DataParser, subdirectory in zip(stars, parsers, subdirectories):
        test_data_directory = os.path.join(base_directory, subdirectory)
        data = DataParser()
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


class TestAstrometry:
    @pytest.mark.e2e
    def test_astrometric_fit(self):
        """
        Tests fitting a line to fake RA and DEC data which has errors calculated from the real intermediate data
        from Hip1, Hip2, and GaiaDR2. This only fits a line to the first 11 points.
        """
        stars = ['049699', '027321', '027321']
        data_choices = ['GaiaDR2', 'Hip1', 'Hip2']
        base_directory = os.path.join(os.getcwd(), 'htof/test/data_for_tests')
        for star_id, data_choice in zip(stars, data_choices):
            test_data_directory = os.path.join(base_directory, data_choice)

            fitter = Astrometry(data_choice, star_id, test_data_directory)
            num_pts = len(fitter.data.julian_day_epoch())
            ra0, dec0, mu_ra, mu_dec = fitter.fit(ra_vs_epoch=np.ones(num_pts),
                                                  dec_vs_epoch=np.ones(num_pts))
            assert True

    def test_shift_line_to_central_epoch(self):
        ra0, dec0, mu_ra, mu_dec = 1, 2, 10, 20
        solution = np.array([ra0, dec0, mu_ra, mu_dec])
        ra0s, dec0s, mu_ras, mu_decs = Astrometry._shift_to_central_epoch(solution, central_epoch_dec=1,
                                                            central_epoch_ra=1, central_epoch_fmt='MJD')
        assert np.allclose([mu_ras, mu_decs], [mu_ra, mu_dec])
        assert np.isclose(ra0s, ra0 + 1*mu_ra)
        assert np.isclose(dec0s, dec0 + 1 * mu_dec)

    def test_shift_warns_on_large_fractional_year(self):
        with pytest.warns(UserWarning):
            Astrometry._shift_to_central_epoch(np.array([0, 0, 0, 0]), central_epoch_dec=5000,
                                               central_epoch_ra=5000, central_epoch_fmt='frac_year')
