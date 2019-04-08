import pytest
import os
import numpy as np

from htof.main import GaiaData, HipparcosRereductionData, HipparcosOriginalData
from htof.main import AstrometricFitter


@pytest.mark.e2e
def test_parse_and_fit_to_line():
    stars = ['49699', '27321', '27321']
    parsers = [GaiaData, HipparcosOriginalData, HipparcosRereductionData]
    subdirectories = ['GaiaDR2', 'Hip1', 'Hip2']
    base_directory = os.path.join(os.getcwd(), 'htof/data_for_tests')
    for star, DataParser, subdirectory in zip(stars, parsers, subdirectories):
        test_data_directory = os.path.join(base_directory, subdirectory)
        data = DataParser()
        data.parse(star_hip_id=star,
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
