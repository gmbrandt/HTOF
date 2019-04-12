import numpy as np
import pandas as pd
import os
import glob
import datetime
import warnings

from astropy.time import Time

import abc


class IntermediateDataParser(object):
    """
    Base class for parsing Hip1 and Hip2 data. self.epoch, self.covariance_matrix and self.scan_angle are saved
    as panda dataframes. use .values (e.g. self.epoch.values) to call the ndarray version.
    """
    def __init__(self, scan_angle=None, epoch=None, residuals=None, inverse_covariance_matrix=None):
        self.scan_angle = scan_angle
        self._epoch = epoch
        self.residuals = residuals
        self.inverse_covariance_matrix = inverse_covariance_matrix

    @staticmethod
    def read_intermediate_data_file(star_hip_id, intermediate_data_directory, skiprows, header, sep):
        if len(star_hip_id) < 6:
            warnings.warn("Hip ID has not been fully specified (e.g. 3865 instead of 003865). Search may fail.",
                          SyntaxWarning)
        filepath = os.path.join(os.path.join(intermediate_data_directory, '**/'), '*' + star_hip_id + '*')
        filepath_list = glob.glob(filepath, recursive=True)
        if len(filepath_list) > 1:
            raise ValueError('More than one input file with hip id {0} found'.format(star_hip_id))
        data = pd.read_csv(filepath_list[0], sep=sep, skiprows=skiprows, header=header, engine='python')
        return data

    @abc.abstractmethod
    def parse(self, star_id, intermediate_data_parent_directory, **kwargs):
        pass

    def julian_day_epoch(self):
        return self.convert_hip_style_epochs_to_julian_day(self._epoch)

    @staticmethod
    def convert_hip_style_epochs_to_julian_day(epochs, half_day_correction=True):
        jd_epochs = []
        for epoch in epochs.values:
            epoch_year = int(epoch)
            fraction = epoch - int(epoch)
            utc_time = datetime.datetime(year=epoch_year, month=1, day=1) + datetime.timedelta(days=365.25) * fraction
            if half_day_correction:
                utc_time += datetime.timedelta(days=0.5)
            jd_epochs.append(Time(utc_time).jd)
        return np.array(jd_epochs)

    def calculate_inverse_covariance_matrices(self, cross_scan_along_scan_var_ratio=1E5):
        cov_matrices = calculate_covariance_matrices(self.scan_angle,
                                                     cross_scan_along_scan_var_ratio=cross_scan_along_scan_var_ratio)
        icov_matrices = np.zeros_like(cov_matrices)
        for i in range(len(cov_matrices)):
            icov_matrices[i] = np.linalg.pinv(cov_matrices[i])
        self.inverse_covariance_matrix = icov_matrices


def calculate_covariance_matrices(scan_angles, cross_scan_along_scan_var_ratio=1E5):
    """
    :param scan_angles: pandas DataFrame with scan angles, e.g. as-is from the data parsers. scan_angles.values is a
                        numpy array with the scan angles
    :param cross_scan_along_scan_var_ratio: var_cross_scan / var_along_scan
    :return An ndarray with shape (len(scan_angles), 2, 2), e.g. an array of covariance matrices in the same order
    as the scan angles
    """
    covariance_matrices = []
    cov_matrix_in_scan_basis = np.array([[cross_scan_along_scan_var_ratio, 0],
                                         [0, 1]])
    # we define the along scan to be 'y' in the scan basis.
    for theta in scan_angles.values.flatten():
        # for Hipparcos, theta is measured against north, specifically east of the north equatorial pole
        c, s = np.cos(theta), np.sin(theta)
        Rccw = np.array([[c, -s], [s, c]])
        cov_matrix_in_ra_dec_basis = np.matmul(np.matmul(Rccw, cov_matrix_in_scan_basis), Rccw.T)
        covariance_matrices.append(cov_matrix_in_ra_dec_basis)
    return np.array(covariance_matrices)


class HipparcosOriginalData(IntermediateDataParser):
    def __init__(self, scan_angle=None, epoch=None, residuals=None, inverse_covariance_matrix=None):
        super(HipparcosOriginalData, self).__init__(scan_angle=scan_angle,
                                                    epoch=epoch, residuals=residuals,
                                                    inverse_covariance_matrix=inverse_covariance_matrix)

    def parse(self, star_hip_id, intermediate_data_directory, data_choice='NDAC'):
        """
        :param star_hip_id: a string which is just the number for the HIP ID.
        :param intermediate_data_directory: the path (string) to the place where the intermediate data is stored, e.g.
                Hip2/IntermediateData/resrec
                note you have to specify the file resrec or absrec. We use the residual records, so specify resrec.
        :param data_choice: 'FAST' or 'NDAC'. This slightly affects the scan angles. This mostly affects
        the residuals which are not used.
        """
        if (data_choice is not 'NDAC') and (data_choice is not 'FAST'):
            raise ValueError('data choice has to be either NDAC or FAST')
        data = self.read_intermediate_data_file(star_hip_id, intermediate_data_directory,
                                                skiprows=0, header='infer', sep='\s*\|\s*')
        # select either the data from the NDAC or the FAST consortium.
        data = data[data['IA2'] == data_choice[0]]
        # compute scan angles and observations epochs according to van Leeuwen & Evans 1997, eq. 11 & 12.
        self.scan_angle = np.arctan2(data['IA3'], data['IA4'])  # unit radians
        self._epoch = data['IA6'] / data['IA3'] + 1991.25
        self.residuals = data['IA8']  # unit milli-arcseconds (mas)


class HipparcosRereductionData(IntermediateDataParser):
    def __init__(self, scan_angle=None, epoch=None, residuals=None, inverse_covariance_matrix=None):
        super(HipparcosRereductionData, self).__init__(scan_angle=scan_angle,
                                                       epoch=epoch, residuals=residuals,
                                                       inverse_covariance_matrix=inverse_covariance_matrix)

    def parse(self, star_hip_id, intermediate_data_directory, **kwargs):
        data = self.read_intermediate_data_file(star_hip_id, intermediate_data_directory,
                                                skiprows=1, header=None, sep='\s+')
        # compute scan angles and observations epochs from van Leeuwen 2007, table G.8
        # see also Figure 2.1, section 2.5.1, and section 4.1.2
        self.scan_angle = np.arctan2(data[3], data[4])  # data[3] = cos(psi), data[4] = sin(psi)
        self._epoch = data[1] + 1991.25
        self.residuals = data[5]  # unit milli-arcseconds (mas)


class GaiaData(IntermediateDataParser):
    def __init__(self, scan_angle=None, epoch=None, residuals=None, inverse_covariance_matrix=None):
        super(GaiaData, self).__init__(scan_angle=scan_angle,
                                       epoch=epoch, residuals=residuals,
                                       inverse_covariance_matrix=inverse_covariance_matrix)

    def parse(self, star_hip_id, intermediate_data_directory, **kwargs):
        data = self.read_intermediate_data_file(star_hip_id, intermediate_data_directory,
                                                skiprows=0, header='infer', sep='\s*,\s*')
        self._epoch = data['ObservationTimeAtBarycentre[BarycentricJulianDateInTCB]']
        self.scan_angle = data['scanAngle[rad]']

    def julian_day_epoch(self):
        return self._epoch.values.flatten()
