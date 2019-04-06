import pandas as pd
import numpy as np
import os

from HTOF.main import HipparcosOriginalData, HipparcosRereductionData, GaiaData, IntermediateDataParser
from HTOF.main import calculate_covariance_matrices


def test_parse_original_data():
    test_data_directory = os.path.join(os.getcwd(), 'data_for_tests/Hip1')
    data = HipparcosOriginalData()
    data.parse(star_hip_id='27321',
               intermediate_data_directory=test_data_directory,
               data_choice='FAST', convert_to_jd=False)
    assert len(data.epoch) == 32
    assert np.isclose(data.epoch[0], 1990.005772)
    assert np.isclose(data.scan_angle[0], -2.009532)
    assert np.isclose(data.epoch[17], 1990.779865)
    assert np.isclose(data.scan_angle[17], 2.769795)
    data.parse(star_hip_id='27321',
               intermediate_data_directory=test_data_directory,
               data_choice='NDAC', convert_to_jd=False)
    assert len(data.epoch) == 34
    assert np.isclose(data.epoch[1], 1990.005386)
    assert np.isclose(data.scan_angle[1], -2.009979)
    assert np.isclose(data.epoch[10], 1990.455515)
    assert np.isclose(data.scan_angle[10], 0.827485)


def test_parse_rereduced_data():
    test_data_directory = os.path.join(os.getcwd(), 'data_for_tests/Hip2')
    data = HipparcosRereductionData()
    data.parse(star_hip_id='27321',
               intermediate_data_directory=test_data_directory, convert_to_jd=False)
    assert len(data.epoch) == 111
    assert np.isclose(data.epoch[0], 1990.005)
    assert np.isclose(data.scan_angle[0], -2.006668)
    assert np.isclose(data.epoch[84], 1991.952)
    assert np.isclose(data.scan_angle[84], -0.941235)


def test_convert_dates_to_jd():
    parser = IntermediateDataParser()
    epochs = pd.DataFrame(data=[1990.0, 1990.25], index=[5, 6])
    jd_epochs = parser.convert_hip_style_epochs_to_julian_day(epochs)
    assert np.isclose(jd_epochs.values[0], 2447892.5)
    assert np.isclose(jd_epochs.values[1], 2447892.5 + 0.25*365.25)


def test_parse_gaia_data():
    test_data_directory = os.path.join(os.getcwd(), 'data_for_tests/GaiaDR2/IntermediateData')
    data = GaiaData()
    data.parse(intermediate_data_directory=test_data_directory,
               star_hip_id='49699')
    assert len(data.epoch) == 72
    assert np.isclose(data.epoch[0], 2456951.7659301492)
    assert np.isclose(data.scan_angle[0], -1.8904696884345342)
    assert np.isclose(data.epoch[70], 2458426.7784441216)
    assert np.isclose(data.scan_angle[70], 2.821818345385301)


def test_calculating_covariance_matrices():
    scan_angles = pd.DataFrame(data=np.linspace(0, 2 * np.pi, 10))
    covariances = calculate_covariance_matrices(scan_angles, cross_scan_var_to_along_scan_var_ratio=10)
    assert len(covariances) == len(scan_angles)
    assert np.allclose(covariances[-1], covariances[0])
    assert np.allclose(covariances[0], np.array([[10, 0], [0, 1]]))
    for cov_matrix, scan_angle in zip(covariances, scan_angles.values.flatten()):
        assert np.isclose(scan_angle % np.pi, angle_of_short_axis_of_error_ellipse(cov_matrix) % np.pi)
        # modulo pi since the scan angle and angle of short axis could differ in sign from one another.


def angle_of_short_axis_of_error_ellipse(cov_matrix):
    vals, vecs = np.linalg.eigh(cov_matrix)
    # Compute "tilt" of ellipse using first eigenvector
    x, y = vecs[:, 0]
    theta = np.arctan2(y, x) - np.pi/2
    return theta
