import pandas as pd
import numpy as np
import os

from HTOF.main import HipparcosOriginalData, HipparcosRereductionData, GaiaData


def test_parse_original_data():
    test_data_directory = os.path.join(os.getcwd(), 'data_for_tests/Hip1')
    data = HipparcosOriginalData()
    data.parse(star_hip_id='27321',
               intermediate_data_directory=test_data_directory,
               data_choice='FAST')
    assert len(data.epoch) == 32
    assert np.isclose(data.epoch[0], 1990.005772)
    assert np.isclose(data.scan_angle[0], -2.009532)
    assert np.isclose(data.epoch[17], 1990.779865)
    assert np.isclose(data.scan_angle[17], 2.769795)
    data.parse(star_hip_id='27321',
               intermediate_data_directory=test_data_directory,
               data_choice='NDAC')
    assert len(data.epoch) == 34
    assert np.isclose(data.epoch[1], 1990.005386)
    assert np.isclose(data.scan_angle[1], -2.009979)
    assert np.isclose(data.epoch[10], 1990.455515)
    assert np.isclose(data.scan_angle[10], 0.827485)


def test_parse_rereduced_data():
    test_data_directory = os.path.join(os.getcwd(), 'data_for_tests/Hip2')
    data = HipparcosRereductionData()
    data.parse(star_hip_id='27321',
               intermediate_data_directory=test_data_directory)
    assert len(data.epoch) == 111
    assert np.isclose(data.epoch[0], 1990.005)
    assert np.isclose(data.scan_angle[0], -2.006668)
    assert np.isclose(data.epoch[84], 1991.952)
    assert np.isclose(data.scan_angle[84], -0.941235)


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
    scan_angles = None
    assert True
