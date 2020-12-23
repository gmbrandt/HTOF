import pandas as pd
import numpy as np
import pytest
import mock
import os
import tempfile
from ast import literal_eval

from astropy.table import Table
from htof.parse import HipparcosOriginalData, HipparcosRereductionDVDBook,\
    GaiaData, DataParser, GaiaDR2, GaiaeDR3, DecimalYearData, HipparcosRereductionJavaTool, digits_only, \
    match_filename, get_nparam
from htof.parse import calc_inverse_covariance_matrices


class TestHipparcosOriginalData:
    def test_parse(self, hip_id='027321'):
        test_data_directory = os.path.join(os.getcwd(), 'htof/test/data_for_tests/Hip1')
        data = HipparcosOriginalData()
        data.parse(star_id=hip_id,
                   intermediate_data_directory=test_data_directory,
                   data_choice='FAST')
        assert len(data) == 32
        assert np.isclose(data._epoch[0], 1990.005772)
        assert np.isclose(np.sin(data.scan_angle[0]), -0.9053, rtol=.01)
        assert np.isclose(data._epoch[17], 1990.779865)
        assert np.isclose(np.sin(data.scan_angle[17]), 0.3633, rtol=.01)
        assert np.isclose(data.along_scan_errs.values[0], 2.21)
        data.parse(star_id=hip_id,
                   intermediate_data_directory=test_data_directory,
                   data_choice='NDAC')
        assert len(data) == 34
        assert np.isclose(data._epoch[1], 1990.005386)
        assert np.isclose(np.sin(data.scan_angle[1]), -0.9051, rtol=.01)
        assert np.isclose(data._epoch[10], 1990.455515)
        assert np.isclose(np.sin(data.scan_angle[10]), 0.7362, rtol=.01)
        data.parse(star_id=hip_id,
                   intermediate_data_directory=test_data_directory,
                   data_choice='MERGED')
        assert len(data) == 34
        assert np.isclose(data._epoch[0], 1990.005386)
        assert np.isclose(np.sin(data.scan_angle[0]), -0.9053, atol=.001)
        assert np.isclose(data._epoch[5], 1990.455515)
        assert np.isclose(np.sin(data.scan_angle[5]), 0.7364, atol=.001)
        assert np.isclose(data.along_scan_errs[5], 2.0814, atol=.0001)
        assert np.isclose(data.residuals[5], 1.1021, atol=.0001)
        data.parse(star_id=hip_id,
                   intermediate_data_directory=test_data_directory,
                   data_choice='BOTH')
        assert len(data) == 66

    @pytest.mark.integration
    def test_concatenation(self, hip_id='027321'):
        test_data_directory = os.path.join(os.getcwd(), 'htof/test/data_for_tests/Hip1')
        data = HipparcosOriginalData()
        data.parse(star_id=hip_id,
                   intermediate_data_directory=test_data_directory,
                   data_choice='FAST')
        data.calculate_inverse_covariance_matrices()
        covars = data.inverse_covariance_matrix
        data += data
        data.calculate_inverse_covariance_matrices()
        assert np.allclose(covars, data.inverse_covariance_matrix[:len(covars)])
        assert np.allclose(covars, data.inverse_covariance_matrix[len(covars):])

    def test_parse_IA3_eq_zero(self, hip_id='004391'):
        test_data_directory = os.path.join(os.getcwd(), 'htof/test/data_for_tests/Hip1')
        data = HipparcosOriginalData()
        data.parse(star_id=hip_id,
                   intermediate_data_directory=test_data_directory,
                   data_choice='NDAC')
        # IA4 is larger than IA3 -> more precise answer from IA7/IA4 (differs by .0103538462)
        assert np.isclose(data._epoch[38], 1992.9142, rtol=1e-8)
        # IA3 is larger than IA4 -> more precise answer from IA6/IA3 (differs by .0001434115)
        assert np.isclose(data._epoch[5], 1990.4227657727, rtol=1e-8)
        # IA3 is exactly 0 which would result in NaN. Must use IA7/IA4 to get a valid result. 
        # IA4 is negative. This case thus also checks that the absolute value is correctly used when comparing IA3 and IA4.
        assert np.isclose(data._epoch[30], 1992.407684232, rtol=1e-8)

    def test_merged_parse_removes_flagged_observations(self):
        test_data_directory = os.path.join(os.getcwd(), 'htof/test/data_for_tests/Hip1')
        data = HipparcosOriginalData()
        data.parse(star_id='999999',
                   intermediate_data_directory=test_data_directory,
                   data_choice='MERGED')
        assert len(data) == 32

    def test_raises_exception_on_bad_data_choice(self):
        test_data_directory = os.path.join(os.getcwd(), 'htof/test/data_for_tests/Hip1')
        data = HipparcosOriginalData()
        with pytest.raises(Exception):
            data.parse(star_id='027321',
                       intermediate_data_directory=test_data_directory,
                       data_choice='something')


class TestHipparcosRereductionDVDBook:
    def test_error_inflation_factor(self):
        u = 0.875291 # D. Michalik et al. 2014 Q factor for Hip 27321, calculated by hand
        assert np.isclose(HipparcosRereductionDVDBook.error_inflation_factor(111, 5, -1.81), u, rtol=1e-5)

    def test_parse(self):
        test_data_directory = os.path.join(os.getcwd(), 'htof/test/data_for_tests/Hip2')
        data = HipparcosRereductionDVDBook()
        data.parse(star_id='027321', intermediate_data_directory=test_data_directory)
        nu = 111 - 5
        Q = nu * (np.sqrt(2/(9*nu))*-1.81 + 1 - 2/(9*nu))**3  # D. Michalik et al. 2014 Q factor for Hip 27321
        # Note that F2 = -1.81 in the CD intermediate data, while F2 = -1.63 on Vizier.
        u = np.sqrt(Q/nu)  # error inflation factor.
        assert len(data) == 111
        assert np.isclose(data._epoch[0], 1990.005)
        assert np.isclose(np.sin(data.scan_angle[0]), -0.9065, rtol=.01)
        assert np.isclose(data.along_scan_errs.values[0], 0.81 * u)
        assert np.isclose(data._epoch[84], 1991.952)
        assert np.isclose(np.sin(data.scan_angle[84]), -0.8083, rtol=.01)

    def test_reject_obs(self):
        test_data_directory = os.path.join(os.getcwd(), 'htof/test/data_for_tests/Hip2')
        data = HipparcosRereductionDVDBook()
        data.parse(star_id='27321', intermediate_data_directory=test_data_directory)
        assert len(data) == 111 - 0
        assert data.rejected_epochs == []

        data.parse(star_id='84', intermediate_data_directory=test_data_directory)
        assert len(data) == 96 - 1
        assert np.allclose(np.sort(data.rejected_epochs), [69])

        data.parse(star_id='70', intermediate_data_directory=test_data_directory)
        assert len(data) == 112 - 5
        assert np.allclose(np.sort(data.rejected_epochs), [36, 55, 59, 64, 105])

        data.parse(star_id='40', intermediate_data_directory=test_data_directory)
        assert len(data) == 134 - 2
        assert np.allclose(np.sort(data.rejected_epochs), [6, 53])

        data.parse(star_id='072477', intermediate_data_directory=test_data_directory)
        assert len(data) == 64 - 0
        assert data.rejected_epochs == []



class TestHipparcosRereductionJavaTool:
    test_data_directory = os.path.join(os.getcwd(), 'htof/test/data_for_tests/Hip21')

    def test_parse(self):
        data = HipparcosRereductionJavaTool()
        data.parse(star_id='27321', intermediate_data_directory=self.test_data_directory)
        u = 1  # Error inflation factor
        assert len(data) == 111
        assert np.isclose(data._epoch[0], 1990.0055)
        assert np.isclose(np.sin(data.scan_angle[0]), -0.9050, rtol=.01)
        assert np.isclose(data.along_scan_errs.values[0], 0.80 * u)
        assert np.isclose(data._epoch[84], 1991.9523)
        assert np.isclose(np.sin(data.scan_angle[84]), -0.8083, rtol=.01)

    def test_outlier_reject(self):
        data = HipparcosRereductionJavaTool()
        data.parse(star_id='27100', intermediate_data_directory=self.test_data_directory)
        assert len(data) == 147 - 2  # num entries - num outliers
        # outliers are marked with negative AL errors. Assert outliers are gone.
        assert np.all(data.along_scan_errs > 0)


class TestDataParser:
    def test_parse_raises_file_not_found_error(self):
        with pytest.raises(FileNotFoundError):
            test_data_directory = os.path.join(os.getcwd(), 'htof/test/data_for_tests/Hip2')
            data = HipparcosRereductionDVDBook()
            data.parse(star_id='12gjas2',
                       intermediate_data_directory=test_data_directory)

    @mock.patch('htof.parse.glob.glob', return_value=['path/027321.dat', 'path/027321.dat'])
    def test_parse_raises_error_on_many_files_found(self, fake_glob):
        test_data_directory = os.path.join(os.getcwd(), 'htof/test/data_for_tests/Hip2')
        data = HipparcosRereductionDVDBook()
        with pytest.raises(FileNotFoundError):
            data.parse(star_id='027321',
                       intermediate_data_directory=test_data_directory)

    @mock.patch('htof.parse.pd.read_csv', return_value=None)
    @mock.patch('htof.parse.glob.glob', return_value=['path/127321.dat', 'path/27321.dat'])
    def test_read_on_couple_similar_filepaths(self, fake_glob, fake_load):
        test_data_directory = os.path.join(os.getcwd(), 'path/')
        DataParser().read_intermediate_data_file('27321', test_data_directory, None, None, None)
        fake_load.assert_called_with('path/27321.dat', sep=None, skiprows=None, header=None, engine='python')

    @mock.patch('htof.parse.pd.read_csv', return_value=None)
    @mock.patch('htof.parse.glob.glob')
    def test_read_on_many_filepaths(self, fake_glob, fake_load):
        test_data_directory = os.path.join(os.getcwd(), 'path/')
        fake_glob.return_value = ['/fake/path/1232.dat', '/fake/path/23211.dat', '/fake/path/232.dat']
        DataParser().read_intermediate_data_file('232', test_data_directory, None, None, None)
        fake_load.assert_called_with('/fake/path/232.dat', sep=None, skiprows=None, header=None, engine='python')
        # Test hip2 style names
        fake_glob.return_value = ['H075/HIP075290.d', 'H075/HIP075293.d', 'H107/HIP107529.d', 'H007/HIP007529.d', 'H117/HIP117529.d']
        DataParser().read_intermediate_data_file('7529', test_data_directory, None, None, None)
        fake_load.assert_called_with('H007/HIP007529.d', sep=None, skiprows=None, header=None, engine='python')

    def test_match_filename(self):
        paths = ['/hip/24374.txt', 'hip/43749.txt', 'hip/43740.txt', 'hip/4374.txt']
        assert 'hip/4374.txt' == match_filename(paths, '4374')[0]

    @mock.patch('htof.parse.pd.read_csv', return_value=None)
    @mock.patch('htof.parse.glob.glob')
    def test_read_on_irregular_paths(self, fake_glob, fake_load):
        test_data_directory = os.path.join(os.getcwd(), 'path/')
        fake_glob.return_value = ['/fake/path/SSADSx1232.dat', '/fake/path/WDASxs23211.dat', '/fake/path/OMGH232.dat']
        DataParser().read_intermediate_data_file('232', test_data_directory, None, None, None)
        fake_load.assert_called_with('/fake/path/OMGH232.dat', sep=None, skiprows=None, header=None, engine='python')

    def test_len(self):
        assert len(DataParser()) == 0
        assert len(DataParser(epoch=pd.DataFrame(np.arange(4)))) == 4


def test_convert_dates_to_jd():
    epoch = pd.DataFrame(data=[1990.0, 1990.25], index=[5, 6])
    parser = DecimalYearData(epoch=epoch)
    jd_epochs = parser.julian_day_epoch()
    assert np.isclose(jd_epochs[0], 2447892.5)
    assert np.isclose(jd_epochs[1], 2447892.5 + 0.25*365.25)


def test_call_jd_dates_hip():
    parser = DecimalYearData()
    parser._epoch = pd.DataFrame(data=[1990.0, 1990.25], index=[5, 6])
    jd_epochs = parser.julian_day_epoch()
    assert np.isclose(jd_epochs[0], 2447892.5)
    assert np.isclose(jd_epochs[1], 2447892.5 + 0.25*365.25)


def test_call_jd_dates_gaia():
    parser = DataParser()
    parser._epoch = pd.DataFrame(data=[2447892.5, 2447893], index=[5, 6])
    jd_epochs = parser.julian_day_epoch()
    assert np.isclose(jd_epochs[0], 2447892.5)
    assert np.isclose(jd_epochs[1], 2447893)


def test_trim_gaia_data():
    parser = GaiaData()
    datemin, datemax = 3, 5
    epochs = pd.DataFrame(data=[datemin - 1, datemin, datemax, datemax + 1], index=[3, 4, 5, 6])
    data = pd.DataFrame(data=[datemin - 1, datemin, datemax, datemax + 1], index=[3, 4, 5, 6])
    data = parser.trim_data(data, epochs, datemin, datemax)
    assert np.allclose(data.values.flatten(), [datemin, datemax])


@mock.patch('htof.parse.calc_inverse_covariance_matrices', return_value=np.array([np.ones((2, 2))]))
def test_calculate_inverse_covariances(mock_cov_matrix):
    parser = DataParser()
    parser.calculate_inverse_covariance_matrices()
    assert np.allclose(parser.inverse_covariance_matrix[0], np.ones((2, 2)))


class TestParseGaiaData:
    @pytest.mark.integration
    def test_parse_all_epochs(self):
        test_data_directory = os.path.join(os.getcwd(), 'htof/test/data_for_tests/GaiaDR2/IntermediateData')
        data = GaiaData(max_epoch=np.inf, min_epoch=-np.inf)
        data.parse(intermediate_data_directory=test_data_directory,
                   star_id='049699')
        assert len(data._epoch) == 72
        assert np.isclose(data._epoch[0], 2456951.7659301492)
        assert np.isclose(data.scan_angle[0], -1.8904696884345342)
        assert np.isclose(data._epoch[70], 2458426.7784441216)
        assert np.isclose(data.scan_angle[70], 2.821818345385301)

    @pytest.mark.integration
    def test_parse_selects_valid_epochs(self):
        test_data_directory = os.path.join(os.getcwd(), 'htof/test/data_for_tests/GaiaDR2/IntermediateData')
        data = GaiaDR2(max_epoch=2458426.7784441218, min_epoch=2457143.4935643710)
        data.DEAD_TIME_TABLE_NAME = None  # Do not reject dead times for this test
        data.parse(intermediate_data_directory=test_data_directory,
                   star_id='049699')

        assert len(data._epoch) == 68
        assert np.isclose(data._epoch.iloc[0], 2457143.4935643715)
        assert np.isclose(data.scan_angle.iloc[0], -0.3066803677989655)
        assert np.isclose(data._epoch.iloc[67], 2458426.7784441216)
        assert np.isclose(data.scan_angle.iloc[67], 2.821818345385301)

    @pytest.mark.integration
    def test_DR2parse_removes_dead_times(self):
        # tests the dead time table against a fake source which has the first observation fall in a data gap
        # and has the second observation not fall into any data gap.
        test_data_directory = os.path.join(os.getcwd(), 'htof/test/data_for_tests/GaiaDR2/IntermediateData')
        data = GaiaDR2(max_epoch=np.inf, min_epoch=-np.inf)
        data.parse(intermediate_data_directory=test_data_directory,
                   star_id='gaiatestsource01')

        assert len(data._epoch) == 1
        assert np.isclose(data._epoch.iloc[0], 2456893.28785)
        assert np.isclose(data.scan_angle.iloc[0], -1.7804696884345342)

    @pytest.mark.integration
    def test_eDR3parse_removes_dead_times(self):
        # tests the dead time table against a fake source which has the first observation fall in a data gap
        # and has the second observation not fall into any data gap.
        test_data_directory = os.path.join(os.getcwd(), 'htof/test/data_for_tests/GaiaeDR3/IntermediateData')
        data = GaiaeDR3(max_epoch=np.inf, min_epoch=-np.inf)
        data.parse(intermediate_data_directory=test_data_directory,
                   star_id='gaiatestsource01')

        assert len(data._epoch) == 1
        assert np.isclose(data._epoch.iloc[0], 2456893.28785)
        assert np.isclose(data.scan_angle.iloc[0], -1.7804696884345342)


def test_write_with_missing_info():
    data = DataParser(scan_angle=np.arange(3), epoch=np.arange(1991, 1994),
                      residuals=np.arange(2, 5),
                      inverse_covariance_matrix=None,
                      along_scan_errs=None)
    with tempfile.TemporaryDirectory() as tmp_dir:
        data.write(os.path.join(tmp_dir, 'out.csv'))
        t = Table.read(os.path.join(tmp_dir, 'out.csv'))
        assert np.allclose(t['residuals'], data.residuals)
        assert np.allclose(t['julian_day_epoch'], data.julian_day_epoch())
        assert np.allclose(t['scan_angle'], data.scan_angle)
        assert len(t.colnames) == 5


def test_write():
    data = DataParser(scan_angle=np.arange(3), epoch=np.arange(1991, 1994),
                      residuals=np.arange(2, 5),
                      inverse_covariance_matrix=np.array([[1, 2], [3, 4]]) * np.ones((3, 2, 2)),
                      along_scan_errs=np.arange(3, 6))
    with tempfile.TemporaryDirectory() as tmp_dir:
        data.write(os.path.join(tmp_dir, 'out.csv'))
        t = Table.read(os.path.join(tmp_dir, 'out.csv'))
        assert np.allclose(t['residuals'], data.residuals)
        assert np.allclose(t['julian_day_epoch'], data.julian_day_epoch())
        assert np.allclose(t['scan_angle'], data.scan_angle)
        assert np.allclose(t['along_scan_errs'], data.along_scan_errs)
        icovs = [np.array(literal_eval(icov)) for icov in t['icov']]
        assert np.allclose(icovs, data.inverse_covariance_matrix)
        assert len(t.colnames) == 5


def test_calculating_covariance_matrices():
    scan_angles = pd.DataFrame(data=np.linspace(0, 2 * np.pi, 5))
    icovariances = calc_inverse_covariance_matrices(scan_angles, cross_scan_along_scan_var_ratio=10)
    assert len(icovariances) == len(scan_angles)
    # calculate the covariance matrices so that we can compute the scan angle from the matrix.
    covariances = np.zeros_like(icovariances)
    for i in range(len(icovariances)):
        covariances[i] = np.linalg.pinv(icovariances[i])
    assert np.allclose(covariances[-1], covariances[0])  # check 2pi is equivalent to 0.
    assert np.allclose(covariances[0], np.array([[10, 0], [0, 1]]))  # angle of 0 has AL parallel with DEC.
    for cov_matrix, scan_angle in zip(covariances, scan_angles.values.flatten()):
        assert np.isclose(scan_angle % np.pi, angle_of_short_axis_of_error_ellipse(cov_matrix) % np.pi)
        # modulo pi since the scan angle and angle of short axis could differ in sign from one another.


def test_concatenating_data():
    data = DataParser(scan_angle=np.arange(3), epoch=np.arange(1991, 1994),
                      residuals=np.arange(2, 5),
                      inverse_covariance_matrix=np.array([[1, 2], [3, 4]]) * np.ones((3, 2, 2)),
                      along_scan_errs=np.arange(3, 6))
    new_data = sum([data, data])
    assert np.allclose(new_data.scan_angle, [*data.scan_angle, *data.scan_angle])
    assert np.allclose(new_data.residuals, [*data.residuals, *data.residuals])
    assert len(new_data.inverse_covariance_matrix) == 2*len(data.inverse_covariance_matrix)
    data += data
    assert np.allclose(new_data.scan_angle, data.scan_angle)
    assert np.allclose(new_data.residuals, data.residuals)
    data.calculate_inverse_covariance_matrices()


def test_add_to_empty():
    data = DataParser(scan_angle=np.arange(3), epoch=pd.DataFrame(np.arange(1991, 1994)),
                      residuals=np.arange(2, 5),
                      inverse_covariance_matrix=np.array([[1, 2], [3, 4]]) * np.ones((3, 2, 2)),
                      along_scan_errs=np.arange(3, 6))
    new_data = DataParser()
    new_data += data
    assert np.allclose(new_data.scan_angle, data.scan_angle)
    assert len(new_data) == len(data)


def test_concatenating_data_with_missing():
    data = DataParser(scan_angle=np.arange(3), epoch=pd.DataFrame(np.arange(1991, 1994)),
                      residuals=np.arange(2, 5))
    new_data = sum([data, data])
    assert np.allclose(new_data.scan_angle, [*data.scan_angle, *data.scan_angle])
    assert np.allclose(new_data.residuals, [*data.residuals, *data.residuals])
    data += data
    assert np.allclose(new_data.scan_angle, data.scan_angle)
    assert np.allclose(new_data.residuals, data.residuals)


@pytest.mark.integration
def test_two_concatenate_decyear_and_jd():
    data = DecimalYearData(scan_angle=np.arange(3), epoch=pd.DataFrame(np.arange(1991, 1994)),
                           residuals=np.arange(2, 5))
    data2 = DataParser(scan_angle=np.arange(3), epoch=pd.DataFrame(np.arange(2456951, 2456954)),
                       residuals=np.arange(2, 5))
    data3 = data + data2
    assert np.allclose(data3.julian_day_epoch()[:len(data)], data.julian_day_epoch())
    assert np.allclose(data3.julian_day_epoch()[len(data):], data2.julian_day_epoch())


def angle_of_short_axis_of_error_ellipse(cov_matrix):
    vals, vecs = np.linalg.eigh(cov_matrix)
    # Compute "tilt" of ellipse using first eigenvector
    x, y = vecs[:, 0]
    theta = np.arctan2(y, x) - np.pi/2
    return theta


def test_digits_only():
    assert '987978098098098' == digits_only("sdkjh987978asd098as0980a98sd")


def test_get_nparam():
    assert 5 == get_nparam('23115')
    assert 5 == get_nparam('95')
    assert 7 == get_nparam('97')
