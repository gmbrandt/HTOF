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
import pkg_resources

from astropy.time import Time
from astropy.table import QTable, Column, Table

from htof import settings as st
from htof.utils.data_utils import merge_consortia, safe_concatenate
from htof.utils.parse_utils import gaia_obmt_to_tcb_julian_year

import abc


class DataParser(object):
    """
    Base class for parsing Hip1, Hip2 and Gaia data. self.epoch, self.covariance_matrix and self.scan_angle are saved
    as pandas.DataFrame. use .values (e.g. self.epoch.values) to call the ndarray version.
    """
    def __init__(self, scan_angle=None, epoch=None, residuals=None, inverse_covariance_matrix=None,
                 along_scan_errs=None):
        self.scan_angle = pd.Series(scan_angle)
        self._epoch = pd.DataFrame(epoch)
        self.residuals = pd.Series(residuals)
        self.along_scan_errs = pd.Series(along_scan_errs)
        self.inverse_covariance_matrix = inverse_covariance_matrix

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
        pass    # pragma: no cover

    def julian_day_epoch(self):
        return self._epoch.values.flatten()

    def calculate_inverse_covariance_matrices(self, cross_scan_along_scan_var_ratio=1E5):
        cov_matrices = calculate_covariance_matrices(self.scan_angle,
                                                     cross_scan_along_scan_var_ratio=cross_scan_along_scan_var_ratio,
                                                     along_scan_errs=self.along_scan_errs)
        icov_matrices = np.zeros_like(cov_matrices)
        for i in range(len(cov_matrices)):
            icov_matrices[i] = np.linalg.pinv(cov_matrices[i])
        self.inverse_covariance_matrix = icov_matrices

    def write(self, path: str, *args, **kwargs):
        """
        :param path: str. filepath to write out the processed data.
        :param args: arguments for astropy.table.Table.write()
        :param kwargs: keyword arguments for astropy.table.Table.write()
        :return: None

        Note: The IntermediateDataParser.inverse_covariance_matrix are added to the table as strings
        so that they are easily writable. The icov matrix is saved a string.
        Each element of t['icov'] can be recovered with ast.literal_eval(t['icov'][i])
        where i is the index. ast.literal_eval(t['icov'][i]) will return a 2x2 list.
        """
        t = self.as_table()
        # fix icov matrices as writable strings.
        t['icov'] = [str(icov.tolist()) for icov in t['icov']]
        t.write(path, fast_writer=False, *args, **kwargs)

    def as_table(self):
        """
        :return: astropy.table.QTable
                 The IntermediateDataParser object tabulated.
                 This table has as columns all of the attributes of IntermediateDataParser.

                 For any attribute which is empty or None, the column will contain zeros.
        """
        cols = [self.scan_angle, self.julian_day_epoch(), self.residuals, self.along_scan_errs, self.inverse_covariance_matrix]
        cols = [Column(col) for col in cols]
        # replacing incorrect length columns with empties.
        cols = [col if len(col) == len(self) else Column(None, length=len(self)) for col in cols]

        t = QTable(cols, names=['scan_angle', 'julian_day_epoch', 'residuals', 'along_scan_errs', 'icov'])
        return t

    def __add__(self, other):
        all_scan_angles = pd.concat([self.scan_angle, other.scan_angle])
        all_epoch = pd.concat([pd.DataFrame(self.julian_day_epoch()), pd.DataFrame(other.julian_day_epoch())])
        all_residuals = pd.concat([self.residuals, other.residuals])
        all_along_scan_errs = pd.concat([self.along_scan_errs, other.along_scan_errs])

        all_inverse_covariance_matrix = safe_concatenate(self.inverse_covariance_matrix,
                                                         other.inverse_covariance_matrix)

        return DataParser(scan_angle=all_scan_angles, epoch=all_epoch, residuals=all_residuals,
                          inverse_covariance_matrix=all_inverse_covariance_matrix,
                          along_scan_errs=all_along_scan_errs)

    def __radd__(self, other):
        if other == 0:
            return self
        return self.__add__(other)

    def __len__(self):
        return len(self._epoch)


class GaiaData(DataParser):
    DEAD_TIME_TABLE_NAME = None

    def __init__(self, scan_angle=None, epoch=None, residuals=None, inverse_covariance_matrix=None,
                 min_epoch=-np.inf, max_epoch=np.inf, along_scan_errs=None):
        super(GaiaData, self).__init__(scan_angle=scan_angle, along_scan_errs=along_scan_errs,
                                       epoch=epoch, residuals=residuals,
                                       inverse_covariance_matrix=inverse_covariance_matrix)
        self.min_epoch = min_epoch
        self.max_epoch = max_epoch

    def parse(self, star_id, intermediate_data_directory, **kwargs):
        data = self.read_intermediate_data_file(star_id, intermediate_data_directory,
                                                skiprows=0, header='infer', sep='\s*,\s*')
        data = self.trim_data(data['ObservationTimeAtBarycentre[BarycentricJulianDateInTCB]'],
                              data, self.min_epoch, self.max_epoch)
        data = self.reject_dead_times(data['ObservationTimeAtBarycentre[BarycentricJulianDateInTCB]'], data)
        self._epoch = data['ObservationTimeAtBarycentre[BarycentricJulianDateInTCB]']
        self.scan_angle = data['scanAngle[rad]']

    def trim_data(self, epochs, data, min_mjd, max_mjd):
        valid = np.logical_and(epochs >= min_mjd, epochs <= max_mjd)
        return data[valid].dropna()

    def reject_dead_times(self, epochs, data):
        # there will be different astrometric gaps for gaia DR2 and DR3 because rejection criteria may change.
        # hence we have the appropriate parsers have different values for DEAD_TIME_TABLE_NAME.
        if self.DEAD_TIME_TABLE_NAME is None:
            # return the data if there is no dead time table specified.
            return data
        dead_time_table = Table.read(pkg_resources.resource_filename('htof', self.DEAD_TIME_TABLE_NAME))
        # convert on board mission time (OBMT) to julian day
        for col, newcol in zip(['start', 'end'], ['start_tcb_jd', 'end_tcb_jd']):
            dead_time_table[newcol] = gaia_obmt_to_tcb_julian_year(dead_time_table[col]).jd
        # make a mask of the epochs. Those that are within a dead time window have a value of 0 (masked)
        valid = np.ones(len(data), dtype=bool)
        for entry in dead_time_table:
            valid[np.logical_and(epochs >= entry['start_tcb_jd'], epochs <= entry['end_tcb_jd'])] = 0
        # reject the epochs which fall within a dead time window
        data = data[valid].dropna()
        return data


class DecimalYearData(DataParser):
    def __init__(self, scan_angle=None, epoch=None, residuals=None, inverse_covariance_matrix=None,
                 along_scan_errs=None):
        super(DecimalYearData, self).__init__(scan_angle=scan_angle, along_scan_errs=along_scan_errs,
                                              epoch=epoch, residuals=residuals,
                                              inverse_covariance_matrix=inverse_covariance_matrix)

    def parse(self, star_id, intermediate_data_parent_directory, **kwargs):
        pass  # pragma: no cover

    def julian_day_epoch(self):
        return Time(self._epoch.values.flatten(), format='decimalyear').jd


def _match_filename_to_star_id(star_id, filepath_list):
    return [path for path in filepath_list if os.path.basename(path).split('.')[0] == str(star_id)]


def calculate_covariance_matrices(scan_angles, cross_scan_along_scan_var_ratio=1E5,
                                  along_scan_errs=None):
    """
    :param scan_angles: pandas.DataFrame.
            data frame with scan angles, e.g. as-is from IntermediateDataParser.read_intermediate_data_file.
            scan_angles.values is a numpy array with the scan angles
    :param cross_scan_along_scan_var_ratio: var_cross_scan / var_along_scan
    :return An ndarray with shape (len(scan_angles), 2, 2), e.g. an array of covariance matrices in the same order
    as the scan angles
    """
    if along_scan_errs is None or len(along_scan_errs) == 0:
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


class HipparcosOriginalData(DecimalYearData):
    def __init__(self, scan_angle=None, epoch=None, residuals=None, inverse_covariance_matrix=None,
                 along_scan_errs=None):
        super(HipparcosOriginalData, self).__init__(scan_angle=scan_angle, along_scan_errs=along_scan_errs,
                                                    epoch=epoch, residuals=residuals,
                                                    inverse_covariance_matrix=inverse_covariance_matrix)

    def parse(self, star_id, intermediate_data_directory, data_choice='MERGED'):
        """
        :param star_id: a string which is just the number for the HIP ID.
        :param intermediate_data_directory: the path (string) to the place where the intermediate data is stored, e.g.
                Hip2/IntermediateData/resrec
                note you have to specify the file resrec or absrec. We use the residual records, so specify resrec.
        :param data_choice: 'FAST' or 'NDAC'. This slightly affects the scan angles. This mostly affects
        the residuals which are not used.
        """
        if (data_choice is not 'NDAC') and (data_choice is not 'FAST') and (data_choice is not 'MERGED'):
            raise ValueError('data choice has to be either NDAC or FAST or MERGED.')
        data = self.read_intermediate_data_file(star_id, intermediate_data_directory,
                                                skiprows=10, header='infer', sep='\s*\|\s*')
        data = self._fix_unnamed_column(data)
        data = self._select_data(data, data_choice)
        # compute scan angles and observations epochs according to van Leeuwen & Evans 1997, eq. 11 & 12.
        self.scan_angle = np.arctan2(data['IA3'], data['IA4'])  # unit radians, arctan2(sin, cos)
        # Use the larger denominator when computing the epoch offset. 
        # This increases numerical precision and avoids NaNs if one of the two fields (IA3, IA4) is exactly zero.
        self._epoch = 1991.25 + (data['IA6'] / data['IA3']).where(abs(data['IA3']) > abs(data['IA4']), (data['IA7'] / data['IA4']))
        self.residuals = data['IA8']  # unit milli-arcseconds (mas)
        self.along_scan_errs = data['IA9']  # unit milli-arcseconds

    @staticmethod
    def _select_data(data, data_choice):
        # restrict intermediate data to either NDAC, FAST, or merge the NDAC and FAST results.
        if data_choice is 'MERGED':
            data = merge_consortia(data)
        else:
            data = data[data['IA2'].str.upper() == {'NDAC': 'N', 'FAST': 'F'}[data_choice]]
        return data

    @staticmethod
    def _fix_unnamed_column(data, correct_key='IA2', col_idx=1):
        data.rename(columns={data.columns[col_idx]: correct_key}, inplace=True)
        return data


class HipparcosRereductionData(DecimalYearData):
    def __init__(self, scan_angle=None, epoch=None, residuals=None, inverse_covariance_matrix=None,
                 along_scan_errs=None):
        super(HipparcosRereductionData, self).__init__(scan_angle=scan_angle, along_scan_errs=along_scan_errs,
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
        header = self.read_intermediate_data_file(star_id, intermediate_data_directory,
                                                  skiprows=0, header=None, sep='\s+').iloc[0]
        data = self.read_intermediate_data_file(star_id, intermediate_data_directory,
                                                skiprows=1, header=None, sep='\s+')
        self.scan_angle = np.arctan2(data[3], data[4])  # data[3] = sin(psi), data[4] = cos(psi)
        self._epoch = data[1] + 1991.25
        self.residuals = data[5]  # unit milli-arcseconds (mas)
        self.along_scan_errs = data[6]  # unit milli-arcseconds (mas)
        self.along_scan_errs *= self.error_inflation_factor(header[2], header[4], header[6])

    @staticmethod
    def error_inflation_factor(ntr, nparam, f2):
        """
        :param ntr: int. Number of transits
        :param nparam: int. Number of parameters used in the solution (e.g. 5, 7, 9..)
        :param f2: float. Goodness of fit metric. field F2 in the Hipparcos Re-reduction catalog.
        :return: u. float.
        The errors are to be scaled by u = Sqrt(Q/v) in equation B.4 of D. Michalik et al. 2014.
        (Title: Joint astrometric solution of Hipparcos and Gaia)
        NOTE: ntr (the number of transits) given in the header of the Hip2 IAD, is not necessarily
        the number of transits used.
        """
        num_transits_used = ntr
        nu = num_transits_used - nparam  # equation B.1 of D. Michalik et al. 2014
        Q = nu * (np.sqrt(2/(9*nu))*f2 + 1 - 2/(9*nu))**3  # equation B.3
        u = np.sqrt(Q/nu)  # equation B.4. This is the chi squared statistic of the fit.
        return u


class GaiaDR2(GaiaData):
    DEAD_TIME_TABLE_NAME = 'data/astrometric_gaps_gaiadr2_08252020.csv'

    def __init__(self, scan_angle=None, epoch=None, residuals=None, inverse_covariance_matrix=None,
                 min_epoch=st.GaiaDR2_min_epoch, max_epoch=st.GaiaDR2_max_epoch, along_scan_errs=None):
        super(GaiaDR2, self).__init__(scan_angle=scan_angle, along_scan_errs=along_scan_errs,
                                      epoch=epoch, residuals=residuals,
                                      inverse_covariance_matrix=inverse_covariance_matrix,
                                      min_epoch=min_epoch, max_epoch=max_epoch)
