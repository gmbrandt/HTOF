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
from math import ceil

from astropy.time import Time
from astropy.table import QTable, Column
from astropy.coordinates import Angle

from htof import settings as st
from htof.utils.data_utils import merge_consortia, safe_concatenate
from htof.fit import AstrometricFitter
from htof.utils.fit_utils import chisq_of_fit

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
        self._epoch = data['ObservationTimeAtBarycentre[BarycentricJulianDateInTCB]']
        self.scan_angle = data['scanAngle[rad]']

    def trim_data(self, epochs, data, min_mjd, max_mjd):
        valid = np.logical_and(epochs >= min_mjd, epochs <= max_mjd)
        return data[valid].dropna()


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


class HipparcosRereductionCDBook(DecimalYearData):
    def __init__(self, scan_angle=None, epoch=None, residuals=None, inverse_covariance_matrix=None,
                 along_scan_errs=None):
        super(HipparcosRereductionCDBook, self).__init__(scan_angle=scan_angle, along_scan_errs=along_scan_errs,
                                                         epoch=epoch, residuals=residuals,
                                                         inverse_covariance_matrix=inverse_covariance_matrix)

    def parse(self, star_id, intermediate_data_directory, error_inflate=True, header_rows=1, reject_obs=True, **kwargs):
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
        n_transits, nparam, f2, percent_rejected = header[2], header[4], header[6], header[7]
        if reject_obs:
            print(star_id)
            n_reject = ceil((percent_rejected + 1)/100 * n_transits)
            epochs_to_reject = find_epochs_to_reject(self, f2, n_transits, nparam, n_reject)
            print(epochs_to_reject)
        if error_inflate:
            self.along_scan_errs *= self.error_inflation_factor(n_transits, nparam, f2)


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
        # strip the solution type (5, 7, or 9) from the solution type, which is a number 10xd+s consisting of
        # two parts: d and s. see Note 1 on Vizier for the Hipparcos re-reduction.
        nparam = int(str(int(nparam))[-1])
        #
        num_transits_used = ntr  # TODO take into account the n_rejected_obs when calculating num_transits_used
        nu = num_transits_used - nparam  # equation B.1 of D. Michalik et al. 2014
        Q = nu * (np.sqrt(2/(9*nu))*f2 + 1 - 2/(9*nu))**3  # equation B.3
        u = np.sqrt(Q/nu)  # equation B.4. This is the chi squared statistic of the fit.
        return u


class HipparcosRereductionJavaTool(HipparcosRereductionCDBook):
    def __init__(self, scan_angle=None, epoch=None, residuals=None, inverse_covariance_matrix=None,
                 along_scan_errs=None):
        super(HipparcosRereductionJavaTool, self).__init__(scan_angle=scan_angle, along_scan_errs=along_scan_errs,
                                                         epoch=epoch, residuals=residuals,
                                                         inverse_covariance_matrix=inverse_covariance_matrix)

    def parse(self, star_id, intermediate_data_directory, **kwargs):
        # TODO set error error_inflate=True when the F2 value is available in the headers of 2.1 data.
        super(HipparcosRereductionJavaTool, self).parse(star_id, intermediate_data_directory,
                                                        error_inflate=False, header_rows=5, reject_obs=False)
        # remove outliers. Outliers have negative along scan errors.
        not_outlier = (self.along_scan_errs > 0)
        self.along_scan_errs, self.scan_angle = self.along_scan_errs[not_outlier], self.scan_angle[not_outlier]
        self.residuals, self._epoch = self.residuals[not_outlier], self._epoch[not_outlier]


class GaiaDR2(GaiaData):
    def __init__(self, scan_angle=None, epoch=None, residuals=None, inverse_covariance_matrix=None,
                 min_epoch=st.GaiaDR2_min_epoch, max_epoch=st.GaiaDR2_max_epoch, along_scan_errs=None):
        super(GaiaDR2, self).__init__(scan_angle=scan_angle, along_scan_errs=along_scan_errs,
                                      epoch=epoch, residuals=residuals,
                                      inverse_covariance_matrix=inverse_covariance_matrix,
                                      min_epoch=min_epoch, max_epoch=max_epoch)


def digits_only(x: str):
    return re.sub("[^0-9]", "", x)


def match_filename(paths, star_id):
    return [f for f in paths if digits_only(os.path.basename(f).split('.')[0]).zfill(6) == star_id.zfill(6)]


def find_epochs_to_reject_old(data: DataParser, catalog_f2, n_transits, nparam, max_n_reject):
    ra_resid = Angle(data.residuals.values * np.sin(data.scan_angle.values), unit='mas')
    dec_resid = Angle(data.residuals.values * np.cos(data.scan_angle.values), unit='mas')

    data.calculate_inverse_covariance_matrices(cross_scan_along_scan_var_ratio=1E5)
    fitter = AstrometricFitter(inverse_covariance_matrices=data.inverse_covariance_matrix,
                               epoch_times=data.epoch, central_epoch_dec=1991.25, central_epoch_ra=1991.25,
                               fit_degree=1, use_parallax=False)
    _, _, chisquared = fitter.fit_line(ra_resid.mas, dec_resid.mas, return_all=True)

    dchisq_per_epoch = fitter.astrometric_solution_vector_components['ra'] * ra_resid.mas.reshape(-1, 1) + \
                       fitter.astrometric_solution_vector_components['dec'] * dec_resid.mas.reshape(-1, 1)
    reject_idx = []
    n_reject = 0
    # calculate f2 without rejecting any observations
    f2 = compute_f2(n_transits - nparam, chisquared)
    idx = list(np.arange(dchisq_per_epoch.shape[0]))
    # if f2 does not agree, try and find outliers based on making chisquared a stationary point.
    while n_reject < max_n_reject and not np.isclose(catalog_f2, f2, atol=0.05):
        # compute sum of dchisq components**2 for every possible combination of rejecting one observation.
        trials = np.ones((len(dchisq_per_epoch), len(dchisq_per_epoch)))
        np.fill_diagonal(trials, 0)
        trials = np.stack([trials] * dchisq_per_epoch.shape[1], axis=-1)
        dchisq_trials = dchisq_per_epoch * trials
        sum_squared_components = np.sum(np.sum(dchisq_trials, axis=1)**2, axis=1)
        reject = np.argmin(sum_squared_components).flatten()[0]
        # save the true index of the rejected observation
        reject_idx.append(idx[reject])
        # remove the rejected observation from the dchisq array and the true index array.
        idx.pop(reject)
        dchisq_per_epoch = np.delete(dchisq_per_epoch, reject, axis=0)
        n_reject += 1  # record that we rejected an observation.
        # find the best fit solution
        cov_matrix = np.linalg.pinv(np.sum(fitter.astrometric_chi_squared_matrices[idx], axis=0), hermitian=True)
        ra_solution_vecs = fitter.astrometric_solution_vector_components['ra'][idx]
        dec_solution_vecs = fitter.astrometric_solution_vector_components['dec'][idx]
        chi2_vector = np.dot(ra_resid.mas[idx], ra_solution_vecs) + np.dot(dec_resid.mas[idx], dec_solution_vecs)
        solution = np.matmul(cov_matrix, chi2_vector)
        # calculating chisq of the fit.
        chisquared = chisq_of_fit(solution, ra_resid.mas[idx], dec_resid.mas[idx],
                                  fitter.ra_epochs[idx], fitter.dec_epochs[idx],
                                  fitter.inverse_covariance_matrices[idx], use_parallax=False)
        f2 = compute_f2(n_transits - n_reject - nparam, chisquared)
    if not np.isclose(catalog_f2, f2, atol=0.05):
        print('catalog f2 value is {0} while the found value is {1}. Outlier rejection was not'
              'able to recover which observations were rejected in the catalog entry.'.format(catalog_f2, f2))
    sum_squared_chisq = np.sum(np.sum(dchisq_per_epoch, axis=0) ** 2)
    if sum_squared_chisq > 0.1:
        print('sum of the squares of the chisquared derivatives is {0}. This should be closer to zero. The solution'
              'is likely not a stationary point of the residuals.'.format(sum_squared_chisq))
    return reject_idx


def find_epochs_to_reject(data: DataParser, catalog_f2, n_transits, nparam, max_n_reject):
    ra_resid = Angle(data.residuals.values * np.sin(data.scan_angle.values), unit='mas')
    dec_resid = Angle(data.residuals.values * np.cos(data.scan_angle.values), unit='mas')

    data.calculate_inverse_covariance_matrices(cross_scan_along_scan_var_ratio=1E5)
    fitter = AstrometricFitter(inverse_covariance_matrices=data.inverse_covariance_matrix,
                               epoch_times=data.epoch, central_epoch_dec=1991.25, central_epoch_ra=1991.25,
                               fit_degree=1, use_parallax=False)
    _, _, chisquared = fitter.fit_line(ra_resid.mas, dec_resid.mas, return_all=True)
    #z_score = data.residuals.values/data.along_scan_errs.values

    # calculate f2 without rejecting any observations
    reject_idx = []
    f2 = compute_f2(n_transits - nparam, chisquared)
    valid_solution = np.isclose(catalog_f2, f2, atol=0.05)
    max_n_reject = 3
    if not valid_solution:
        full_idx = np.arange(len(data))
        n_reject = 0
        # sort by zscore and only brute force search over the worst 15 observations or something.
        while n_reject < max_n_reject and not valid_solution:
            n_reject += 1
            chisquareds = []
            combinations = itertools.combinations(full_idx, n_reject)
            subsets = set(combinations)
            print(len(subsets))
            for idx_to_reject in subsets:
                idx = [i for i in full_idx if i not in idx_to_reject]
                # find the best fit solution
                cov_matrix = np.linalg.pinv(np.sum(fitter.astrometric_chi_squared_matrices[idx], axis=0), hermitian=True)
                ra_solution_vecs = fitter.astrometric_solution_vector_components['ra'][idx]
                dec_solution_vecs = fitter.astrometric_solution_vector_components['dec'][idx]
                chi2_vector = np.dot(ra_resid.mas[idx], ra_solution_vecs) + np.dot(dec_resid.mas[idx], dec_solution_vecs)
                solution = np.matmul(cov_matrix, chi2_vector)
                # calculating chisq of the fit.
                chisquareds.append(chisq_of_fit(solution, ra_resid.mas[idx], dec_resid.mas[idx],
                                   fitter.ra_epochs[idx], fitter.dec_epochs[idx],
                                   fitter.inverse_covariance_matrices[idx], use_parallax=False))
            f2_trials = compute_f2(n_transits - n_reject - nparam, chisquareds)
            best_trial = np.argmin(np.abs(f2_trials - catalog_f2))
            f2 = f2_trials[best_trial]
            reject_idx = list(list(subsets)[best_trial])
            valid_solution = np.isclose(f2_trials[best_trial], catalog_f2, atol=0.05)
    if not np.isclose(catalog_f2, f2, atol=0.05):
        print('catalog f2 value is {0} while the found value is {1}. Outlier rejection was not '
              'able to recover which observations were rejected in the catalog entry.'.format(catalog_f2, f2))
    return reject_idx


def compute_f2(nu, chisquared):
    return (9*nu/2)**(1/2)*((chisquared/nu)**(1/3) + 2/(9*nu) - 1)
