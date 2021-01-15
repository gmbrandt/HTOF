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
import re
import glob
import itertools
from math import ceil, floor
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
        self._rejected_epochs = []

    @staticmethod
    def read_intermediate_data_file(star_id: str, intermediate_data_directory: str, skiprows, header, sep):
        star_id = str(star_id)
        filepath = os.path.join(os.path.join(intermediate_data_directory, '**/'), '*' + star_id + '*')
        filepath_list = glob.glob(filepath, recursive=True)
        if len(filepath_list) != 1:
            # search for the star id with leading zeros stripped
            filepath = os.path.join(os.path.join(intermediate_data_directory, '**/'), '*' + star_id.lstrip('0') + '*')
            filepath_list = glob.glob(filepath, recursive=True)
        if len(filepath_list) != 1:
            # search for files with the full 6 digit hipparcos string
            filepath = os.path.join(os.path.join(intermediate_data_directory, '**/'), '*' + star_id.zfill(6) + '*')
            filepath_list = glob.glob(filepath, recursive=True)
        if len(filepath_list) != 1:
            # take the file with which contains only the hip id if there are multiple matches
            filepath = os.path.join(os.path.join(intermediate_data_directory, '**/'), '*' + star_id.lstrip('0') + '*')
            filepath_list = match_filename(glob.glob(filepath, recursive=True), star_id)
        if len(filepath_list) == 0:
            raise FileNotFoundError('No file with name containing {0} or {1} or {2} found in {3}'
                                    ''.format(star_id, star_id.lstrip('0'), star_id.zfill(6), intermediate_data_directory))
        if len(filepath_list) > 1:
            raise FileNotFoundError('Unable to find the correct file among the {0} files containing {1}'
                                    'found in {2}'.format(len(filepath_list), star_id, intermediate_data_directory))
        data = pd.read_csv(filepath_list[0], sep=sep, skiprows=skiprows, header=header, engine='python')
        return data

    @abc.abstractmethod
    def parse(self, star_id: str, intermediate_data_parent_directory: str, **kwargs):
        pass    # pragma: no cover

    def julian_day_epoch(self):
        return self._epoch.values.flatten()

    @property
    def epoch(self):
        return self._epoch.values.flatten()

    def calculate_inverse_covariance_matrices(self, cross_scan_along_scan_var_ratio=np.inf):
        self.inverse_covariance_matrix = calc_inverse_covariance_matrices(self.scan_angle,
                                                                          cross_scan_along_scan_var_ratio=cross_scan_along_scan_var_ratio,
                                                                          along_scan_errs=self.along_scan_errs)

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
        # transform icov matrices as writable strings.
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

    @property
    def rejected_epochs(self):
        return self._rejected_epochs

    @rejected_epochs.setter
    def rejected_epochs(self, value):
        not_outlier = np.ones(len(self), dtype=bool)
        np.put(not_outlier, value, False)
        self.along_scan_errs, self.scan_angle = self.along_scan_errs[not_outlier], self.scan_angle[not_outlier]
        self.residuals, self._epoch = self.residuals[not_outlier], self._epoch[not_outlier]
        self._rejected_epochs = value

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
        dead_time_table = Table.read(self.DEAD_TIME_TABLE_NAME)
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


def calc_inverse_covariance_matrices(scan_angles, cross_scan_along_scan_var_ratio=np.inf,
                                     along_scan_errs=None):
    """
    :param scan_angles: pandas.DataFrame.
            data frame with scan angles, e.g. as-is from IntermediateDataParser.read_intermediate_data_file.
            scan_angles.values is a numpy array with the scan angles
    :param cross_scan_along_scan_var_ratio: var_cross_scan / var_along_scan
    :param along_scan_errs: array. array of len(scan_angles), the errors in the along scan direction, one for each
    scan in scan_angles.
    :return An ndarray with shape (len(scan_angles), 2, 2), e.g. an array of covariance matrices in the same order
    as the scan angles
    """
    if along_scan_errs is None or len(along_scan_errs) == 0:
        along_scan_errs = np.ones_like(scan_angles.values.flatten())
    icovariance_matrices = []
    icov_matrix_in_scan_basis = np.array([[1, 0],
                                         [0, 1/cross_scan_along_scan_var_ratio]])
    for theta, err in zip(scan_angles.values.flatten(), along_scan_errs):
        c, s = np.cos(theta), np.sin(theta)
        Rot = np.array([[s, -c], [c, s]])
        icov_matrix_in_ra_dec_basis = np.matmul(np.matmul(1/(err ** 2) * Rot, icov_matrix_in_scan_basis), Rot.T)
        icovariance_matrices.append(icov_matrix_in_ra_dec_basis)
    return np.array(icovariance_matrices)


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
        :param data_choice: 'FAST' or 'NDAC', 'BOTH', or 'MERGED. The standard is 'MERGED' which does a merger
        of the 'NDAC' and 'FAST' data reductions in the same way as the hipparcos 1991.25 catalog. 'BOTH' keeps
        both consortia's data in the IAD, which would be unphysical and is just for debugging. 'FAST' would keep
        only the FAST consortia data, likewise only NDAC would be kept if you selected 'NDAC'.
        """
        if (data_choice is not 'NDAC') and (data_choice is not 'FAST') and (data_choice is not 'MERGED')\
                and (data_choice is not 'BOTH'):
            raise ValueError('data choice has to be either NDAC or FAST or MERGED or BOTH.')
        data = self.read_intermediate_data_file(star_id, intermediate_data_directory,
                                                skiprows=10, header='infer', sep='\s*\|\s*')
        data = self._fix_unnamed_column(data)
        data = self._select_data(data, data_choice)
        # compute scan angles and observations epochs according to van Leeuwen & Evans 1998
        #  10.1051/aas:1998218, eq. 11 & 12.
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
        elif data_choice is not 'BOTH':
            data = data[data['IA2'].str.upper() == {'NDAC': 'N', 'FAST': 'F'}[data_choice]]
        return data

    @staticmethod
    def _fix_unnamed_column(data, correct_key='IA2', col_idx=1):
        data.rename(columns={data.columns[col_idx]: correct_key}, inplace=True)
        return data


class HipparcosRereductionDVDBook(DecimalYearData):
    def __init__(self, scan_angle=None, epoch=None, residuals=None, inverse_covariance_matrix=None,
                 along_scan_errs=None):
        super(HipparcosRereductionDVDBook, self).__init__(scan_angle=scan_angle, along_scan_errs=along_scan_errs,
                                                          epoch=epoch, residuals=residuals,
                                                          inverse_covariance_matrix=inverse_covariance_matrix)

    def parse(self, star_id, intermediate_data_directory, error_inflate=True, header_rows=1, attempt_adhoc_rejection=True, **kwargs):
        """
        :param: star_id:
        :param: intermediate_data_directory:
        :param: error_inflate: True if the along-scan errors are to be corrected by the inflation factor
        according to equation B.1 of D. Michalik et al. 2014. Only turn this off for tests, or if the parameters
        required to compute the error inflation are unavailable.
        :param: header_rows: int.
        :return:

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
                                                skiprows=header_rows, header=None, sep='\s+')
        self.scan_angle = np.arctan2(data[3], data[4])  # data[3] = sin(psi), data[4] = cos(psi)
        self._epoch = data[1] + 1991.25
        self.residuals = data[5]  # unit milli-arcseconds (mas)
        self.along_scan_errs = data[6]  # unit milli-arcseconds (mas)
        n_transits, nparam, catalog_f2, percent_rejected = header[2], get_nparam(header[4]), header[6], header[7]
        if attempt_adhoc_rejection:
            # must reject before inflating errors, otherwise F2 is around zero.
            epochs_to_reject = find_epochs_to_reject(self, catalog_f2, n_transits, nparam, percent_rejected)
            self.rejected_epochs = epochs_to_reject  # setting rejected_epochs also rejects the epochs (see the @setter)
        if error_inflate:
            # adjust the along scan errors so that the errors on the best fit parameters match the catalog.
            # TODO check that oyu sohuld incorporate n_reject as well.
            self.along_scan_errs *= self.error_inflation_factor(n_transits, nparam, catalog_f2)

    @staticmethod
    def error_inflation_factor(ntr, nparam, f2):
        """
        :param ntr: int. Number of transits used in the catalog solution. I.e. this should be
        N_transit_total - N_reject. So if N_reject is unknown, then the error inflation factor will be slightly wrong.
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


class HipparcosRereductionJavaTool(HipparcosRereductionDVDBook):
    def __init__(self, scan_angle=None, epoch=None, residuals=None, inverse_covariance_matrix=None,
                 along_scan_errs=None):
        super(HipparcosRereductionJavaTool, self).__init__(scan_angle=scan_angle, along_scan_errs=along_scan_errs,
                                                         epoch=epoch, residuals=residuals,
                                                         inverse_covariance_matrix=inverse_covariance_matrix)

    def parse(self, star_id, intermediate_data_directory, **kwargs):
        # TODO set error error_inflate=True when the F2 value is available in the headers of 2.1 data.
        super(HipparcosRereductionJavaTool, self).parse(star_id, intermediate_data_directory,
                                                        error_inflate=False, header_rows=5, attempt_adhoc_rejection=False)
        epochs_to_reject = np.where(self.along_scan_errs < 0)[0]
        self.rejected_epochs = epochs_to_reject  # setting rejected_epochs also rejects the epochs (see the @setter)


class GaiaDR2(GaiaData):
    DEAD_TIME_TABLE_NAME = pkg_resources.resource_filename('htof', 'data/astrometric_gaps_gaiadr2_08252020.csv')

    def __init__(self, scan_angle=None, epoch=None, residuals=None, inverse_covariance_matrix=None,
                 min_epoch=st.GaiaDR2_min_epoch, max_epoch=st.GaiaDR2_max_epoch, along_scan_errs=None):
        super(GaiaDR2, self).__init__(scan_angle=scan_angle, along_scan_errs=along_scan_errs,
                                      epoch=epoch, residuals=residuals,
                                      inverse_covariance_matrix=inverse_covariance_matrix,
                                      min_epoch=min_epoch, max_epoch=max_epoch)


class GaiaeDR3(GaiaData):
    DEAD_TIME_TABLE_NAME = pkg_resources.resource_filename('htof', 'data/astrometric_gaps_gaiaedr3_12232020.csv')

    def __init__(self, scan_angle=None, epoch=None, residuals=None, inverse_covariance_matrix=None,
                 min_epoch=st.GaiaeDR3_min_epoch, max_epoch=st.GaiaeDR3_max_epoch, along_scan_errs=None):
        super(GaiaeDR3, self).__init__(scan_angle=scan_angle, along_scan_errs=along_scan_errs,
                                      epoch=epoch, residuals=residuals,
                                      inverse_covariance_matrix=inverse_covariance_matrix,
                                      min_epoch=min_epoch, max_epoch=max_epoch)


def digits_only(x: str):
    return re.sub("[^0-9]", "", x)


def match_filename(paths, star_id):
    return [f for f in paths if digits_only(os.path.basename(f).split('.')[0]).zfill(6) == star_id.zfill(6)]


def find_epochs_to_reject(data: DataParser, catalog_f2, n_transits, nparam, percent_rejected):
    atol_f2 = 0.1  # f2 must match to the catalog within this to be considered valid.
    # Calculate how many observations were probably rejected
    n_reject = max(floor((percent_rejected - 1) / 100 * n_transits), 0)
    max_n_reject = max(ceil((percent_rejected + 1) / 100 * n_transits), 1)
    # Calculate z_score and grab the worst max_n_reject observations plus a few for wiggle room.
    z_score = np.abs(data.residuals.values/data.along_scan_errs.values)
    possible_rejects = np.arange(len(data))[np.argsort(z_score)[::-1]][:max_n_reject + 3]
    # Calculate f2 without rejecting any observations
    chisquared = np.sum((data.residuals.values/data.along_scan_errs.values)**2)
    f2 = compute_f2(n_transits - nparam, chisquared)
    # Check if f2 agrees with the catalog.
    f2_matches_catalog = np.isclose(catalog_f2, f2, atol=atol_f2)
    #
    reject_idx = []
    if not f2_matches_catalog:
        # If f2 does not agree with the catalog, reject observations trying to match the f2 value.
        reject_idx_per_possibility = []
        f2_per_possibility = []
        while n_reject <= max_n_reject:
            # calculate f2 given sets of rejected observations of n_reject.
            n_reject += 1
            combinations = set(itertools.combinations(possible_rejects, n_reject))
            idx = np.ones((len(combinations), len(data)), dtype=bool)
            idx[[(i,) for i in range(len(combinations))], list(combinations)] = False
            chisquareds = np.nansum((data.residuals.values * idx / (data.along_scan_errs.values * idx))**2, axis=1)
            f2_trials = compute_f2(n_transits - n_reject - nparam, chisquareds)
            best_trial = np.nanargmin(np.abs(f2_trials - catalog_f2))
            f2_per_possibility.append(f2_trials[best_trial])
            reject_idx_per_possibility.append(list(list(combinations)[best_trial]))

        # see which reject combinations give an f2 value that is close to the catalog value (with wiggle room)
        reject_idx_viable = np.asarray(reject_idx_per_possibility)[np.isclose(catalog_f2, f2_per_possibility, atol=3 * atol_f2)]
        if len(reject_idx_viable) == 0:
            print('Could not find a set of epochs to reject that yielded an f2 value within {0} of the catalog value '
                  'for f2. Will proceed without rejecting ANY observations'.format(atol_f2))
            reject_idx = []
        elif len(reject_idx_viable) == 1:
            # if there is one viable combination, take that one
            reject_idx = reject_idx_viable[0]
        else:
            # if there are more than one viable combinations, take the combination that allows the catalog solution to be a stationary point.
            sum_squared_chisq_partials = []
            # evaluate the contributions to the different chisquared components
            sin_scan = np.sin(data.scan_angle.values)
            cos_scan = np.cos(data.scan_angle.values)
            dt = data.epoch - 1991.25
            chi2_vector = (2 * data.residuals.values / data.along_scan_errs.values ** 2 * np.array(
                [sin_scan, cos_scan, dt * sin_scan, dt * cos_scan])).T
            idx = np.ones(len(data), dtype=bool)
            for reject_idx in reject_idx_viable:
                # set the observations to be rejected.
                np.put(idx, reject_idx, False)
                # calculate the sum of the chisquared partials without the rejected observations
                sum_squared_chisq_partials.append(np.sum(np.sum(chi2_vector[idx], axis=0) ** 2))
                # reset the rejected observations for the next loop
                np.put(idx, reject_idx, True)
            reject_idx = reject_idx_viable[np.argmin(sum_squared_chisq_partials)]
        # Done trying to outlier reject.
        # Compute chisquared and f2 for final status check.
        idx = np.ones(len(data), dtype=bool)
        np.put(idx, reject_idx, False)
        chisquared = np.sum((data.residuals.values[idx] / data.along_scan_errs.values[idx]) ** 2)
        f2 = compute_f2(n_transits - nparam - len(reject_idx), chisquared)
        if not np.isclose(catalog_f2, f2, atol=0.05):
            print('catalog f2 value is {0} while the found value is {1}. It is possible that the '
                  'rejected observations numbered {2} are not the correct '
                  'rejections to make.'.format(catalog_f2, round(f2, 2), reject_idx))
    return reject_idx


def get_nparam(nparam_header_val):
    # strip the solution type (5, 7, or 9) from the solution type, which is a number 10xd+s consisting of
    # two parts: d and s. see Note 1 on Vizier for the Hipparcos re-reduction.
    return int(str(int(nparam_header_val))[-1])


def compute_f2(nu, chisquared):
    # equation B.2 of D. Michalik et al. 2014. Joint astrometric solution of Hipparcos and Gaia
    return (9*nu/2)**(1/2)*((chisquared/nu)**(1/3) + 2/(9*nu) - 1)
