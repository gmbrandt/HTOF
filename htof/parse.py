"""
  Module for parsing intermediate data from Hipparcos and Gaia.
  For Hipparcos (both reductions) and Gaia, the scan angle theta is the angle between the north
  equitorial pole (declination) and the along-scan axis, defined as positive if east of the north pole
  (positive for increasing RA).

  Author:
    G. Mirek Brandt
    Daniel Michalik
"""

import numpy as np
import pandas as pd
import os
import glob

from astropy.time import Time
from htof import settings as st

import abc


class IntermediateDataParser(object):
    """
    Base class for parsing Hip1 and Hip2 data. self.epoch, self.covariance_matrix and self.scan_angle are saved
    as panda dataframes. use .values (e.g. self.epoch.values) to call the ndarray version.
    """
    def __init__(self, scan_angle=None, epoch=None, residuals=None, inverse_covariance_matrix=None,
                 along_scan_errs=None):
        self.scan_angle = scan_angle
        self._epoch = epoch
        self.residuals = residuals
        self.inverse_covariance_matrix = inverse_covariance_matrix
        self.along_scan_errs = along_scan_errs

    @staticmethod
    def read_intermediate_data_file(star_id, intermediate_data_directory, skiprows, header, sep):
        filepath = os.path.join(os.path.join(intermediate_data_directory, '**/'), '*' + star_id + '*')
        filepath_list = glob.glob(filepath, recursive=True)
        if len(filepath_list) == 0:
            filepath = os.path.join(os.path.join(intermediate_data_directory, '**/'), '*' + star_id.lstrip('0') + '*')
            filepath_list = glob.glob(filepath, recursive=True)
        if len(filepath_list) == 0:
            raise FileNotFoundError('No file with name containing {0} or {1}'
                                    ' found in {2}'.format(str(star_id), str(star_id).lstrip('0'), intermediate_data_directory))
        if len(filepath_list) > 1:
            filepath_list = _match_filename_to_star_id(star_id, filepath_list)
        if len(filepath_list) > 1:
            raise ValueError('More than one filename containing {0}'
                             'found in {1}'.format(str(star_id), intermediate_data_directory))
        data = pd.read_csv(filepath_list[0], sep=sep, skiprows=skiprows, header=header, engine='python')
        return data

    @abc.abstractmethod
    def parse(self, star_id, intermediate_data_parent_directory, **kwargs):
        pass  # pragma: no cover

    def julian_day_epoch(self):
        return fractional_year_epoch_to_jd(self._epoch.values.flatten(), half_day_correction=True)

    def calculate_inverse_covariance_matrices(self, cross_scan_along_scan_var_ratio=1E5):
        cov_matrices = calculate_covariance_matrices(self.scan_angle,
                                                     cross_scan_along_scan_var_ratio=cross_scan_along_scan_var_ratio,
                                                     along_scan_errs=self.along_scan_errs)
        icov_matrices = np.zeros_like(cov_matrices)
        for i in range(len(cov_matrices)):
            icov_matrices[i] = np.linalg.pinv(cov_matrices[i])
        self.inverse_covariance_matrix = icov_matrices


def fractional_year_epoch_to_jd(epoch, half_day_correction=True):
    return Time(epoch, format='decimalyear').jd + half_day_correction * 0.5


def _match_filename_to_star_id(star_id, filepath_list):
    return [path for path in filepath_list if os.path.basename(path).split('.')[0] == str(star_id)]


def calculate_covariance_matrices(scan_angles, cross_scan_along_scan_var_ratio=1E5,
                                  along_scan_errs=None):
    """
    :param scan_angles: pandas DataFrame with scan angles, e.g. as-is from the data parsers. scan_angles.values is a
                        numpy array with the scan angles
    :param cross_scan_along_scan_var_ratio: var_cross_scan / var_along_scan
    :return An ndarray with shape (len(scan_angles), 2, 2), e.g. an array of covariance matrices in the same order
    as the scan angles
    """
    if along_scan_errs is None:
        along_scan_errs = np.ones_like(scan_angles.values.flatten())
    covariance_matrices = []
    cov_matrix_in_scan_basis = np.array([[1, 0],
                                         [0, cross_scan_along_scan_var_ratio]])
    for theta, err in zip(scan_angles.values.flatten(), along_scan_errs):
        c, s = np.cos(theta), np.sin(theta)
        Rot = np.array([[s, -c], [c, s]])
        cov_matrix_in_ra_dec_basis = np.matmul(np.matmul(Rot, (err ** 2) * cov_matrix_in_scan_basis), Rot.T)
        covariance_matrices.append(cov_matrix_in_ra_dec_basis)
    return np.array(covariance_matrices)


class HipparcosOriginalData(IntermediateDataParser):
    def __init__(self, scan_angle=None, epoch=None, residuals=None, inverse_covariance_matrix=None):
        super(HipparcosOriginalData, self).__init__(scan_angle=scan_angle,
                                                    epoch=epoch, residuals=residuals,
                                                    inverse_covariance_matrix=inverse_covariance_matrix)

    def parse(self, star_id, intermediate_data_directory, data_choice='NDAC'):
        """
        :param star_id: a string which is just the number for the HIP ID.
        :param intermediate_data_directory: the path (string) to the place where the intermediate data is stored, e.g.
                Hip2/IntermediateData/resrec
                note you have to specify the file resrec or absrec. We use the residual records, so specify resrec.
        :param data_choice: 'FAST' or 'NDAC'. This slightly affects the scan angles. This mostly affects
        the residuals which are not used.
        """
        if (data_choice is not 'NDAC') and (data_choice is not 'FAST'):
            raise ValueError('data choice has to be either NDAC or FAST')
        data = self.read_intermediate_data_file(star_id, intermediate_data_directory,
                                                skiprows=10, header='infer', sep='\s*\|\s*')
        data = self._fix_unnamed_column(data)
        # select either the data from the NDAC or the FAST consortium.
        data = data[data['IA2'] == data_choice[0]]
        # compute scan angles and observations epochs according to van Leeuwen & Evans 1997, eq. 11 & 12.
        self.scan_angle = np.arctan2(data['IA3'], data['IA4'])  # unit radians, arctan2(sin, cos)
        self._epoch = data['IA6'] / data['IA3'] + 1991.25
        self.residuals = data['IA8']  # unit milli-arcseconds (mas)
        self.along_scan_errs = data['IA9']  # unit milli-arcseconds (mas)

    @staticmethod
    def _fix_unnamed_column(data_frame, correct_key='IA2', col_idx=1):
        data_frame.rename(columns={data_frame.columns[col_idx]: correct_key}, inplace=True)
        return data_frame


class HipparcosRereductionData(IntermediateDataParser):
    def __init__(self, scan_angle=None, epoch=None, residuals=None, inverse_covariance_matrix=None):
        super(HipparcosRereductionData, self).__init__(scan_angle=scan_angle,
                                                       epoch=epoch, residuals=residuals,
                                                       inverse_covariance_matrix=inverse_covariance_matrix)

    def parse(self, star_id, intermediate_data_directory, **kwargs):
        """
        Compute scan angles and observations epochs from van Leeuwen 2007, table G.8
        see also Figure 2.1, section 2.5.1, and section 4.1.2
        NOTE: that the Hipparcos re-reduction book and the figures therein describe the
        scan angle against the north ecliptic pole.
        NOTE: In the actual intermediate astrometry data on the CD the scan angle
        is given as east of the north equatorial pole, as for the original
        Hipparcos and Gaia (Source: private communication between Daniel
        Michalik and Floor van Leeuwen, April 2019).
        """
        data = self.read_intermediate_data_file(star_id, intermediate_data_directory,
                                                skiprows=1, header=None, sep='\s+')
        self.scan_angle = np.arctan2(data[3], data[4])  # data[3] = sin(psi), data[4] = cos(psi)
        self._epoch = data[1] + 1991.25
        self.residuals = data[5]  # unit milli-arcseconds (mas)
        self.along_scan_errs = data[6]  # unit milli-arcseconds (mas)


class GaiaData(IntermediateDataParser):
    def __init__(self, scan_angle=None, epoch=None, residuals=None, inverse_covariance_matrix=None,
                 min_epoch=st.GaiaDR2_min_epoch, max_epoch=st.GaiaDR2_max_epoch):
        super(GaiaData, self).__init__(scan_angle=scan_angle,
                                       epoch=epoch, residuals=residuals,
                                       inverse_covariance_matrix=inverse_covariance_matrix)
        self.min_epoch = min_epoch
        self.max_epoch = max_epoch

    def parse(self, star_id, intermediate_data_directory, **kwargs):
        data = self.read_intermediate_data_file(star_id, intermediate_data_directory,
                                                skiprows=0, header='infer', sep='\s*,\s*')
        self._epoch = data['ObservationTimeAtBarycentre[BarycentricJulianDateInTCB]']
        self.scan_angle = data['scanAngle[rad]']
        self._epoch, self.scan_angle = self.trim_data(self._epoch, self.min_epoch,
                                                      self.max_epoch, [self.scan_angle])

    def julian_day_epoch(self):
        return self._epoch.values.flatten()

    def trim_data(self, epochs, min_mjd, max_mjd, other_data=()):
        valid = np.logical_and(epochs >= min_mjd, epochs <= max_mjd)
        return tuple(data[valid].dropna() for data in [epochs, *other_data])
