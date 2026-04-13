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
from scipy import stats, special
import requests
import warnings
from ast import literal_eval
import os
import re
import glob
import itertools
from math import ceil, floor
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from importlib.resources import files

from astropy.time import Time
from astropy.table import QTable, Column, Table

from htof import settings as st
from htof.utils.data_utils import merge_consortia, safe_concatenate
from htof.utils.parse_utils import gaia_obmt_to_tcb_julian_year, parse_html
from htof.utils.ftp import download_and_save_hip21_data_to

import abc


class DataParser(object):
    """
    Base class for parsing Hip1, Hip2 and Gaia data. self.epoch, self.covariance_matrix and self.scan_angle are saved
    as pandas.DataFrame. use .values (e.g. self.epoch.values) to call the ndarray version.
    """
    def __init__(self, scan_angle=None, epoch=None, residuals=None, inverse_covariance_matrix=None,
                 along_scan_errs=None, parallax_factors=None, meta=None):
        if meta is None:
            meta = {}
        self.scan_angle = pd.Series(scan_angle, dtype=np.float64)
        self._epoch = pd.DataFrame(epoch, dtype=np.float64)
        self.residuals = pd.Series(residuals, dtype=np.float64)
        self.parallax_factors = pd.Series(parallax_factors, dtype=np.float64)
        self.along_scan_errs = pd.Series(along_scan_errs, dtype=np.float64)
        self.inverse_covariance_matrix = inverse_covariance_matrix
        self.meta = meta

    @staticmethod
    def get_intermediate_data_file_path(star_id: str, intermediate_data_directory: str):
        star_id = str(star_id)
        if digits_only(star_id) != star_id:
            raise RuntimeError('Only numeric hip_ids allowed for star_id. E.g. use "27321" instead of Hip 27321. '
                               'Do not include any characters other than digits in the star_id.')
        # take the file(s) whose paths contain only the hip id
        filepath = os.path.join(os.path.join(intermediate_data_directory, '**/'), '*' + star_id.lstrip('0') + '*')
        filepath_list = match_filename(glob.glob(filepath, recursive=True), star_id)
        #
        if len(filepath_list) == 1:
            found_filepath = filepath_list[0]
            message = ''
        elif len(filepath_list) == 0:
            found_filepath = ''
            message = 'No file with name containing {0} or {1} or {2} found ' \
                      'in {3}.'.format(star_id, star_id.lstrip('0'), star_id.zfill(6), intermediate_data_directory)
        else: # more than 1 file found that matches exactly.
            raise FileNotFoundError('Unable to find the correct file among the {0} files containing {1} found in '
                                    '{2}.'.format(len(filepath_list), star_id, intermediate_data_directory))
        return found_filepath, message

    def download_and_save(self, star_id, intermediate_data_directory):
        raise RuntimeError('Downloading the IAD from the web is not implemented for this Parser.')

    def read_intermediate_data_file(self, star_id: str, intermediate_data_directory: str, skiprows, header, sep):
        filepath, msg = self.get_intermediate_data_file_path(star_id, intermediate_data_directory)
        if filepath == '':  # if no file is found:
            print(msg)
            filepath = self.download_and_save(star_id, intermediate_data_directory)
        return pd.read_csv(filepath, sep=sep, skiprows=skiprows, header=header, engine='python')

    @abc.abstractmethod
    def parse(self, star_id: str, intermediate_data_directory: str, **kwargs):
        pass    # pragma: no cover

    @classmethod
    def parse_and_instantiate(cls, star_id: str, intermediate_data_directory: str, **kwargs):
        # ideally, this should replace the parse method above. It makes much more sense to build
        #  a DataParser object immediately when you want to parse.
        parser = cls()
        parser.parse(star_id, intermediate_data_directory, **kwargs)
        return parser

    def julian_day_epoch(self):
        return self._epoch.values.flatten()

    @property
    def epoch(self):
        return self._epoch.values.flatten()

    def calculate_inverse_covariance_matrices(self, cross_scan_along_scan_var_ratio=np.inf):
        self.inverse_covariance_matrix = calc_inverse_covariance_matrices(self.scan_angle,
                                                                          cross_scan_along_scan_var_ratio=cross_scan_along_scan_var_ratio,
                                                                          along_scan_errs=self.along_scan_errs,
                                                                          star_id=self.meta.get('star_id', None))

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

    def scale_along_scan_errs(self, scaling_factor):
        if len(self) == 0:
            raise ValueError('Cannot scale the along scan errors, nor create uniform errors and scale those '
                             'because this data parser does not have any data (i.e., len(self) is 0).')
        if len(self.along_scan_errs) == 0:
            self.along_scan_errs = pd.Series(np.ones(len(self)), dtype=np.float64)
        self.along_scan_errs *= scaling_factor

    def __add__(self, other):
        all_scan_angles = pd.concat([self.scan_angle, other.scan_angle])
        all_epoch = pd.concat([pd.DataFrame(self.julian_day_epoch()), pd.DataFrame(other.julian_day_epoch())])
        all_residuals = pd.concat([self.residuals, other.residuals])
        all_along_scan_errs = pd.concat([self.along_scan_errs, other.along_scan_errs])
        all_parallax_factors = pd.concat([self.parallax_factors, other.parallax_factors])
        all_inverse_covariance_matrix = safe_concatenate(self.inverse_covariance_matrix,
                                                         other.inverse_covariance_matrix)

        return DataParser(scan_angle=all_scan_angles, epoch=all_epoch, residuals=all_residuals,
                          inverse_covariance_matrix=all_inverse_covariance_matrix,
                          along_scan_errs=all_along_scan_errs, parallax_factors=all_parallax_factors)

    def __radd__(self, other):
        if other == 0:
            return self
        return self.__add__(other)

    def __len__(self):
        return len(self._epoch)


class GaiaData(DataParser):
    DEAD_TIME_TABLE_NAME = None

    def __init__(self, scan_angle=None, epoch=None, residuals=None, inverse_covariance_matrix=None,
                 min_epoch=-np.inf, max_epoch=np.inf, along_scan_errs=None, meta=None):
        super(GaiaData, self).__init__(scan_angle=scan_angle, along_scan_errs=along_scan_errs,
                                       epoch=epoch, residuals=residuals, meta=meta,
                                       inverse_covariance_matrix=inverse_covariance_matrix)
        self.min_epoch = min_epoch
        self.max_epoch = max_epoch

    def download_gost_data(self, star_id):
        if int(star_id) < 1 or int(star_id) > 120404:
            raise RuntimeError("Can not download data. The Hipparcos star ID is likely invalid.")
        target = f"HIP{star_id}"
        # fetch xml text
        response = self.query_gost_xml(target)
        if response is None:
            raise RuntimeError("Downloading the scanning law from GOST failed. Try again later, or download this"
                               " file manually using the GOST online interface.")
        # parse xml text to pandas DataFrame
        data = self.parse_xml(response)
        if data is None:
           raise RuntimeError("Can not parse data. The Hipparcos star ID is likely invalid.")
        # keep first astronomic field hit of each observation
        data = self.keep_field_hits(data)
        return data

    def save_gost_data(self, star_id: str, data: pd.DataFrame, intermediate_data_directory: str):
        path = os.path.join(intermediate_data_directory, f"HIP{star_id}.csv")
        os.makedirs(intermediate_data_directory, exist_ok=True)
        data.to_csv(path, index=False, index_label=False)
        return path

    def query_gost_xml(self, target):
        url = f"https://gaia.esac.esa.int/gost/GostServlet?name={target}&service=1"
        try:
            with requests.Session() as s:
                s.get(url)
                headers = {"Cookie": f"JSESSIONID={s.cookies.get_dict()['JSESSIONID']}"}
                response = s.get(url, headers=headers, timeout=180)
                return response.text
        except:
            warnings.warn("Querying the GOST service failed.")
            return None

    def parse_xml(self, response):
        try:
            root = ET.fromstring(response)
        except:
            warnings.warn("The GOST service returned an invalid xml file.")
            return None
        columns = ["Target", "ra[rad]", "dec[rad]", "ra[h:m:s]", "dec[d:m:s]", "ObservationTimeAtGaia[UTC]",
                   "CcdRow[1-7]", "zetaFieldAngle[rad]", "scanAngle[rad]", "Fov[FovP=preceding/FovF=following]",
                   "parallaxFactorAlongScan", "parallaxFactorAcrossScan", "ObservationTimeAtBarycentre[BarycentricJulianDateInTCB]"]
        rows = []
        name = root.find('./targets/target/name').text
        raR = root.find('./targets/target/coords/ra').text
        decR = root.find('./targets/target/coords/dec').text
        raH = root.find('./targets/target/coords/raHms').text
        decH = root.find('./targets/target/coords/decDms').text
        for event in root.findall('./targets/target/events/event'):
            details = event.find('details')
            observationTimeAtGaia = event.find('eventUtcDate').text
            ccdRow = details.find('ccdRow').text
            zetaFieldAngle = details.find('zetaFieldAngle').text
            scanAngle = details.find('scanAngle').text
            fov = details.find('fov').text
            parallaxFactorAl = details.find('parallaxFactorAl').text
            parallaxFactorAc = details.find('parallaxFactorAc').text
            observationTimeAtBarycentre = event.find('eventTcbBarycentricJulianDateAtBarycentre').text
            rows.append([name, raR, decR, raH, decH, observationTimeAtGaia, ccdRow,
                         zetaFieldAngle, scanAngle, fov, parallaxFactorAl, parallaxFactorAc, observationTimeAtBarycentre])
        data = pd.DataFrame(rows, columns=columns)
        data = data.astype({"Target": str,"ra[rad]": float, "dec[rad]": float,"ra[h:m:s]": str,"dec[d:m:s]": str,"ObservationTimeAtGaia[UTC]": str,"CcdRow[1-7]": int,"zetaFieldAngle[rad]": float,"scanAngle[rad]": float,"Fov[FovP=preceding/FovF=following]": str,"parallaxFactorAlongScan": float,"parallaxFactorAcrossScan": float,"ObservationTimeAtBarycentre[BarycentricJulianDateInTCB]": float })
        return data

    def keep_field_hits(self, data):
        """
        Gost files downloaded from the web through REST contain sequences of ten observations, for every observation of the
        star in the scanning law. The first entry is the skymapper CCD hit, and the extra entries are redundant
        (the hits for astrometric field CCD's 1 through 9). Only the second observation of each sequence should be saved.
        This function saves the second observation (this is the hit on the first astrometric field CCD (AF1)).
        """
        format = "%Y-%m-%dT%H:%M:%S.%f"
        t1 = datetime.strptime(data['ObservationTimeAtGaia[UTC]'][0], format)
        buffer = timedelta(hours=1)
        seq = []
        obs = []
        for index, row in data.iterrows():
            rowTime = row['ObservationTimeAtGaia[UTC]']
            t2 = datetime.strptime(rowTime, format)
            if(t2-t1 < buffer):
                seq.append(row)
            else:
                t1 = datetime.strptime(row['ObservationTimeAtGaia[UTC]'], format)
                if len(seq) > 2:
                    obs.append(seq[1])
                seq = []
                seq.append(row)
        columns = ["Target", "ra[rad]", "dec[rad]", "ra[h:m:s]", "dec[d:m:s]", "ObservationTimeAtGaia[UTC]", "CcdRow[1-7]", "zetaFieldAngle[rad]", "scanAngle[rad]", "Fov[FovP=preceding/FovF=following]", "parallaxFactorAlongScan", "parallaxFactorAcrossScan", "ObservationTimeAtBarycentre[BarycentricJulianDateInTCB]"]
        data = pd.DataFrame(obs, columns=columns)
        data = data.reset_index(drop=True)
        return data

    def download_and_save(self, star_id, intermediate_data_directory):
        print(f'Attempting to download it from the web and to save it to {intermediate_data_directory}.')
        data = self.download_gost_data(str(star_id))
        filepath = self.save_gost_data(str(star_id), data, intermediate_data_directory)
        return filepath

    def parse(self, star_id, intermediate_data_directory, **kwargs):
        self.meta['star_id'] = star_id
        data = self.read_intermediate_data_file(star_id, intermediate_data_directory,
                                                skiprows=0, header='infer', sep=r'\s*,\s*')
        data = self.trim_data(data['ObservationTimeAtBarycentre[BarycentricJulianDateInTCB]'],
                              data, self.min_epoch, self.max_epoch)
        data = self.reject_dead_times(data['ObservationTimeAtBarycentre[BarycentricJulianDateInTCB]'], data)
        self._epoch = data['ObservationTimeAtBarycentre[BarycentricJulianDateInTCB]']
        self.scan_angle = data['scanAngle[rad]']
        self.parallax_factors = data['parallaxFactorAlongScan']

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
                 along_scan_errs=None, meta=None):
        super(DecimalYearData, self).__init__(scan_angle=scan_angle, along_scan_errs=along_scan_errs,
                                              epoch=epoch, residuals=residuals, meta=meta,
                                              inverse_covariance_matrix=inverse_covariance_matrix)

    def parse(self, star_id, intermediate_data_directory, **kwargs):
        pass  # pragma: no cover

    def julian_day_epoch(self):
        return Time(self._epoch.values.flatten(), format='decimalyear').jd


def calc_inverse_covariance_matrices(scan_angles, cross_scan_along_scan_var_ratio=np.inf,
                                     along_scan_errs=None, star_id=None):
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
    if np.any(np.isclose(along_scan_errs, 0)):
        warnings.warn(f'The IAD of {star_id} contained an along scan error that '
                      'is zero. This is unphysical, the observation should '
                      'probably have been marked as rejected. '
                      'In order to compute the inverse covariance matrices for '
                      'this source we are setting this AL error to a large '
                      'number (1 arcsec) and continue. ', RuntimeWarning)
        along_scan_errs[np.isclose(along_scan_errs, 0)] = 1000
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

    def download_hip_data(self, star_id):
        if int(star_id) < 1 or int(star_id) > 120404:
            raise RuntimeError("Can not download data. The Hipparcos star ID is likely invalid.")
        response = self.query_hip_html(star_id)
        if response is None:
            raise RuntimeError("Downloading the data from the Hipparcos/Tycho Catalogue Data failed. Try again later, "
                               "or download this file manually using the HIP Catalogue online interface.")
        data = parse_html(response)
        if data is None:
            raise RuntimeError("Can not parse data. The Hipparcos star ID is likely invalid.")
        return data

    def save_hip_data(self, star_id: str, data: str, intermediate_data_directory: str):
        os.makedirs(intermediate_data_directory, exist_ok=True)
        path = os.path.join(intermediate_data_directory, f"{star_id}.txt")
        with open(path, 'w') as file:
            file.write(data)
        return path

    def query_hip_html(self, star_id):
        url = f"https://hipparcos-tools.cosmos.esa.int/cgi-bin/HIPcatalogueSearch.pl?hipiId={star_id}"
        try:
            with requests.Session() as s:
                response = s.get(url, timeout=180)
                return response.text
        except:
            warnings.warn("Querying the HIP service failed.")
            return None

    def download_and_save(self, star_id, intermediate_data_directory):
        print(f'Attempting to download it from the web and to save it to {intermediate_data_directory}.')
        data = self.download_hip_data(str(star_id))
        filepath = self.save_hip_data(str(star_id), data, intermediate_data_directory)
        return filepath

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
        if (data_choice != 'NDAC') and (data_choice != 'FAST') and (data_choice != 'MERGED')\
                and (data_choice != 'BOTH'):
            raise ValueError('data choice has to be either NDAC or FAST or MERGED or BOTH.')
        self.meta['star_id'] = star_id
        data = self.read_intermediate_data_file(star_id, intermediate_data_directory,
                                                skiprows=10, header='infer', sep=r'\s*\|\s*')
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
        self.parallax_factors = data['IA5']

    @staticmethod
    def _select_data(data, data_choice):
        # restrict intermediate data to either NDAC, FAST, or merge the NDAC and FAST results.
        if data_choice == 'MERGED':
            data = merge_consortia(data)
        elif data_choice != 'BOTH':
            data = data[data['IA2'].str.upper() == {'NDAC': 'N', 'FAST': 'F'}[data_choice]]
        return data

    @staticmethod
    def _fix_unnamed_column(data, correct_key='IA2', col_idx=1):
        data.rename(columns={data.columns[col_idx]: correct_key}, inplace=True)
        return data


class HipparcosRereductionDVDBook(DecimalYearData):
    def __init__(self, scan_angle=None, epoch=None, residuals=None, inverse_covariance_matrix=None,
                 along_scan_errs=None, meta=None):
        super(HipparcosRereductionDVDBook, self).__init__(scan_angle=scan_angle, along_scan_errs=along_scan_errs,
                                                          epoch=epoch, residuals=residuals, meta=meta,
                                                          inverse_covariance_matrix=inverse_covariance_matrix)
        self._additional_rejected_epochs = {}  # epochs that need to be rejected due to the write out bug.
        self._rejected_epochs = {}  # epochs that are known rejects, e.g.,
        # those that have negative AL errors in the java tool
        self._cpsi = None
        self._spsi = None
        self._iorb = None

    def read_header(self, star_id, intermediate_data_directory):
        header = self.read_intermediate_data_file(star_id, intermediate_data_directory,
                                                  skiprows=0, header=None, sep=r'\s+')
        return header

    def parse(self, star_id, intermediate_data_directory, error_inflate=True, header_rows=1,
              attempt_adhoc_rejection=True, **kwargs):
        """
        :param: star_id:
        :param: intermediate_data_directory:
        :param: error_inflate: True if the along-scan errors are to be corrected by the inflation factor
        according to Appendix B of D. Michalik et al. 2014. Only turn this off for tests, or if the parameters
        required to compute the error inflation are unavailable.
        :param: header_rows: int.
        :return:

        Compute scan angles and observations epochs from van Leeuwen 2007, table G.8
        see also Figure 2.1, section 2.5.1, and section 4.1.2
        NOTE: that the Hipparcos re-reduction book and the figures therein describe the
        scan angle against the north ecliptic pole.
        NOTE: In the actual intermediate astrometry data on the DVD the scan angle psi
        is given in the equatorial system. This is similar to the original
        Hipparcos and Gaia (Source: private communication between Daniel
        Michalik and Floor van Leeuwen, April 2019), which define the scan angle theta
        as East of the North equatorial pole. theta = pi / 2 - psi,
        see Brandt et al. (2021), Section 2.2.2."
        """
        self.meta['star_id'] = star_id
        header = self.read_header(star_id, intermediate_data_directory)
        data = self.read_intermediate_data_file(star_id, intermediate_data_directory,
                                                skiprows=header_rows, header=None, sep=r'\s+')
        self.scan_angle = np.arctan2(data[3], data[4])  # data[3] = sin(theta) = cos(psi), data[4] = cos(theta) = sin(psi)
        self._epoch = data[1] + 1991.25
        self.residuals = data[5]  # unit milli-arcseconds (mas)
        self.along_scan_errs = data[6]  # unit milli-arcseconds (mas)
        self.parallax_factors = data[2]
        self.meta['catalog_f2'] = header.iloc[0][6]
        self.meta['catalog_soltype'] = header.iloc[0][4]
        #
        self._cpsi = data[3]
        self._spsi = data[4]
        self._iorb = data[0]

        n_transits, nparam, percent_rejected = header.iloc[0][2], get_nparam(header.iloc[0][4]), header.iloc[0][7]
        if attempt_adhoc_rejection:
            warnings.warn(f"For source {self.meta['star_id']}. The DVD IAD does not indicate which observation epochs were "
                           "rejected for the final solution. htof will attempt to find which epochs to "
                           "reject in order to reproduce the catalog parameters. However, if this source "
                           "also has some corrupted residuals (see Brandt et al. 2021, Section 4), then "
                           "this will fail. We recommend you switch to using the IAD from the Java tool, "
                           "since that version of the IAD indicates rejected epochs with negative "
                           "uncertainties.", UserWarning)
            self.rejected_epochs = find_epochs_to_reject_DVD(self, n_transits, percent_rejected, nparam, self.meta['catalog_f2'])
        if error_inflate:
            # adjust the along scan errors so that the errors on the best fit parameters match the catalog.
            self.along_scan_errs *= self.error_inflation_factor(n_transits, nparam, self.meta['catalog_f2'])
        return header, data

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
        the number of transits used in the actual solution.
        """
        num_transits_used = ntr
        nu = num_transits_used - nparam  # equation B.1 of D. Michalik et al. 2014
        Q = nu * (np.sqrt(2/(9*nu))*f2 + 1 - 2/(9*nu))**3  # equation B.3
        u = np.sqrt(Q/nu)  # equation B.4. This is the chi squared statistic of the fit.
        return u

    def _reject_epochs(self, attr_to_set, value):
        residuals_to_reject, orbits_to_reject = value['residual/along_scan_error'], value['orbit/scan_angle/time']
        not_outlier = np.ones(len(self), dtype=bool)
        np.put(not_outlier, residuals_to_reject, False)
        self.residuals, self.along_scan_errs = self.residuals[not_outlier], self.along_scan_errs[not_outlier]
        not_outlier = np.ones(len(self), dtype=bool)
        np.put(not_outlier, orbits_to_reject, False)
        self._epoch, self.scan_angle = self._epoch[not_outlier], self.scan_angle[not_outlier]
        self.parallax_factors, self._iorb = self.parallax_factors[not_outlier], self._iorb[not_outlier]
        self._cpsi, self._spsi = self._cpsi[not_outlier], self._spsi[not_outlier]
        setattr(self, attr_to_set, value)

    @property
    def additional_rejected_epochs(self):
        return self._additional_rejected_epochs

    @additional_rejected_epochs.setter
    def additional_rejected_epochs(self, value):
        self._reject_epochs('_additional_rejected_epochs', value)

    @property
    def rejected_epochs(self):
        return self._rejected_epochs

    @rejected_epochs.setter
    def rejected_epochs(self, value):
        self._reject_epochs('_rejected_epochs', value)


class HipparcosRereductionJavaTool(HipparcosRereductionDVDBook):
    EPOCHREJECTLIST = Table.read(files('htof') / 'data' / 'epoch_reject_shortlist.csv', format='ascii')

    def __init__(self, scan_angle=None, epoch=None, residuals=None, inverse_covariance_matrix=None,
                 along_scan_errs=None, meta=None):
        super(HipparcosRereductionJavaTool, self).__init__(scan_angle=scan_angle, along_scan_errs=along_scan_errs,
                                                           epoch=epoch, residuals=residuals,
                                                           inverse_covariance_matrix=inverse_covariance_matrix,
                                                           meta=meta)

    def read_header(self, star_id, intermediate_data_directory):
        fpath, msg = self.get_intermediate_data_file_path(star_id, intermediate_data_directory)
        with open(fpath) as f:
            lines = f.readlines()
            hline_fst = [float(i) for i in lines[6].split('#')[1].split()]
            hline_scd = [float(i) for i in lines[8].split('#')[1].split()]
            hline_trd = [float(i) if not ('---' in i) else np.nan for i in lines[10].split('#')[1].split()]
        hline_fst = {key: val for key, val in zip(['HIP', 'MCE', 'NRES', 'NC',
                                                'isol_n', 'SCE', 'F2', 'F1'], hline_fst)}
        hline_scd = {key: val for key, val in zip(['Hp','B-V','VarAnn','NOB','NR'], hline_scd)}
        hline_trd = {key: val for key, val in zip(['RAdeg', 'DEdeg', 'Plx', 'pm_RA', 'pm_DE',
                                                'e_RA', 'e_DE', 'e_Plx', 'e_pmRA', 'e_pmDE', 'dpmRA',
                                                'dpmDE', 'e_dpmRA', 'e_dpmDE', 'ddpmRA', 'ddpmDE',
                                                'e_ddpmRA', 'e_ddpmDE', 'upsRA', 'upsDE', 'e_upsRA',
                                                'e_upsDE', 'var'], hline_trd)}
        # fixing some of the value types.
        hline_fst['MCE'] = int(hline_fst['MCE'])
        hline_fst['HIP'] = int(hline_fst['HIP'])
        hline_fst['isol_n'] = int(hline_fst['isol_n'])
        hline_fst['NC'] = int(hline_fst['NC'])
        hline_fst['SCE'] = int(hline_fst['SCE'])
        hline_fst['isol_n'] = int(hline_fst['isol_n'])
        hline_scd['VarAnn'] = int(hline_scd['VarAnn'])
        return {'first': hline_fst, 'second': hline_scd, 'third': hline_trd}

    def download_and_save(self, star_id, intermediate_data_directory):
        print(f'Attempting to download it from the web and to save it to {intermediate_data_directory}.')
        os.makedirs(intermediate_data_directory, exist_ok=True)
        filepath = os.path.join(intermediate_data_directory, 'H' + str(star_id).zfill(6) + '.d')
        download_and_save_hip21_data_to(star_id, filepath)
        return filepath

    def parse(self, star_id, intermediate_data_directory, error_inflate=True, attempt_adhoc_rejection=True,
              reject_known=True, **kwargs):
        self.meta['star_id'] = star_id
        raw_data = self.read_intermediate_data_file(star_id, intermediate_data_directory,
                                                    skiprows=13, header=None, sep=r'\s+')
        header = self.read_header(star_id, intermediate_data_directory)
        self.scan_angle = np.arctan2(raw_data[3], raw_data[4])  # data[3] = sin(theta) = cos(psi), data[4] = cos(theta) = sin(psi)
        self._epoch = raw_data[1] + 1991.25
        self.residuals = raw_data[5]  # unit milli-arcseconds (mas)
        self.along_scan_errs = raw_data[6]  # unit milli-arcseconds (mas)
        self.parallax_factors = raw_data[2]
        self.meta['catalog_f2'] = header['first']['F2']
        self.meta['catalog_soltype'] = header['first']['isol_n']
        #
        self._cpsi = raw_data[3]
        self._spsi = raw_data[4]
        self._iorb = raw_data[0]
        n_transits, n_expected_transits = header['first']['NRES'], header['second']['NOB']
        n_additional_reject = int(n_transits) - int(n_expected_transits)
        # self.meta['catalog_f2'] = header.iloc[0][6]  # this is already set in HipparcosRereductionDVDBook.parse()
        # self.meta['catalog_soltype'] = header.iloc[0][4]  # this is already set in HipparcosRereductionDVDBook.parse()
        max_n_auto_reject = 4
        if attempt_adhoc_rejection:
            if 3 >= n_additional_reject > 0:
                self.additional_rejected_epochs = find_epochs_to_reject_java(self, n_additional_reject)
            if max_n_auto_reject >= n_additional_reject > 3:
                orbit_number = raw_data[0].values
                self.additional_rejected_epochs = find_epochs_to_reject_java_large(self, n_additional_reject, orbit_number)
            if n_additional_reject > max_n_auto_reject:
                # These take too long to do automatically, pull the epochs to reject from the file that we computed
                correct_id = header['first']['HIP']
                t = self.EPOCHREJECTLIST[self.EPOCHREJECTLIST['hip_id'] == int(correct_id)]
                if len(t) == 1:
                    self.additional_rejected_epochs = {'residual/along_scan_error': literal_eval(t['residual/along_scan_error'][0]),
                                                       'orbit/scan_angle/time': literal_eval(t['orbit/scan_angle/time'][0])}
                else:
                    warnings.warn(f'Cannot fix {star_id}. It has more than {max_n_auto_reject} corrupted epochs than can be '
                                  f'corrected on-the-fly. The correct epochs to reject are not in our precomputed list '
                                  f'(epoch_reject_shortlist.csv). This happens for sources where it is computationally '
                                  f'infeasible to find an ad-hoc correction.', UserWarning)    # pragma: no cover
        if not attempt_adhoc_rejection and n_additional_reject > 0:
            warnings.warn(f"attempt_adhoc_rejection = False and {star_id} has {n_additional_reject} "
                          "discrepant observations. You have disabled the ad-hoc "
                          "correction for this Java tool source. The IAD do not correspond "
                          "to the best fit catalog solution. ", UserWarning)
        epochs_to_reject = np.where(self.along_scan_errs <= 0)[0] # note that we have to reject
        # the epochs with negative along scan errors (the formally known epochs that need to be rejected)
        # AFTER we have done the bug correction (rejected the epochs from the write out bug). This order
        # is important because the ad-hoc correction shuffles the orbits.
        if len(epochs_to_reject) > 0 and reject_known:
            # setting self.rejected_epochs also rejects the epochs (see the @rejected_epochs.setter)
            self.rejected_epochs = {'residual/along_scan_error': list(epochs_to_reject),
                                    'orbit/scan_angle/time': list(epochs_to_reject)}
        # compute f2 of the residuals (with ad-hoc correction where applicable)
        nparam = get_nparam(str(int(header['first']['isol_n'])))
        Q = np.sum((self.residuals/self.along_scan_errs)**2)
        n_transits_final = len(self)
        # note that n_transits_final = n_expected_transits - number of indicated rejects (By negative AL errors)
        self.meta['calculated_f2'] = special.erfcinv(stats.chi2.sf(Q, n_transits_final - nparam)*2)*np.sqrt(2)
        if error_inflate:
            # WARNING: we use the catalog (Van Leeuwen 2014 Java tool F2) f2 value here to calculate the error inflation
            # factor. this is because for some sources, the calculated f2 value is much larger than the
            # catalog value. E.g., HIP 87275 has a catalog f2 of 65.29, and a newly calculated f2 is using
            # chi2.sf is infinity.
            # Therefore the error inflation in the catalog is ~7, while the error inflation assuming
            # the new f2 is infinity. We adopt the catalog f2 so as to reproduce the catalog solution and errors.
            # The developers have not yet found this f2 discrepency to be an issue, but any source with it
            # should still be treated with caution.
            self.along_scan_errs *= self.error_inflation_factor(n_transits_final, nparam, self.meta['catalog_f2'])
        return header, raw_data


class GaiaDR2(GaiaData):
    DEAD_TIME_TABLE_NAME = files('htof') / 'data' / 'astrometric_gaps_gaiadr2_08252020.csv'

    def __init__(self, scan_angle=None, epoch=None, residuals=None, inverse_covariance_matrix=None, meta=None,
                 min_epoch=st.GaiaDR2_min_epoch, max_epoch=st.GaiaDR2_max_epoch, along_scan_errs=None):
        super(GaiaDR2, self).__init__(scan_angle=scan_angle, along_scan_errs=along_scan_errs,
                                      epoch=epoch, residuals=residuals,
                                      inverse_covariance_matrix=inverse_covariance_matrix,
                                      min_epoch=min_epoch, max_epoch=max_epoch, meta=meta)


class GaiaeDR3(GaiaData):
    DEAD_TIME_TABLE_NAME = files('htof') / 'data' / 'astrometric_gaps_gaiaedr3_12232020.csv'

    def __init__(self, scan_angle=None, epoch=None, residuals=None, inverse_covariance_matrix=None, meta=None,
                 min_epoch=st.GaiaeDR3_min_epoch, max_epoch=st.GaiaeDR3_max_epoch, along_scan_errs=None):
        super(GaiaeDR3, self).__init__(scan_angle=scan_angle, along_scan_errs=along_scan_errs,
                                      epoch=epoch, residuals=residuals,
                                      inverse_covariance_matrix=inverse_covariance_matrix,
                                      min_epoch=min_epoch, max_epoch=max_epoch, meta=meta)


def digits_only(x: str):
    return re.sub("[^0-9]", "", x)


def match_filename(paths, star_id):
    return [f for f in paths if digits_only(os.path.basename(f).split('.')[0]).zfill(6) == star_id.zfill(6)]


def find_epochs_to_reject_DVD(data: DataParser, n_transits, percent_rejected, nparam, catalog_f2):
    # just looks for combinations of orbits within the dvd IAD that yield a stationary point of chisquared.
    # Note that this does not work for sources with the data corruption.
    chi2_thresh = 1
    possible_rejects = np.arange(len(data))
    min_n_reject = max(floor((percent_rejected - 1) / 100 * n_transits), 0)
    max_n_reject = max(ceil((percent_rejected + 1) / 100 * n_transits), 1)
    max_n_reject = min(max_n_reject, 3)  # limit to three rejected sources so that combinatorics dont blow up.
    # calculate the chisquared partials
    sin_scan = np.sin(data.scan_angle.values)
    cos_scan = np.cos(data.scan_angle.values)
    dt = data.epoch - 1991.25
    rows_to_keep = np.ones(len(data), dtype=bool)
    orbit_factors = np.array([data.parallax_factors.values, sin_scan, cos_scan, dt * sin_scan, dt * cos_scan])
    residual_factors = (data.residuals.values / data.along_scan_errs.values ** 2)
    chi2_vector = (2 * residual_factors * orbit_factors).T
    sum_chisquared_partials_norejects = np.sqrt(np.sum(np.sum(chi2_vector, axis=0) ** 2))
    # we should be able to do the orbit reject calculation fairly easily in memory.
    # for 100 choose 3 we have like 250,000 combinations of orbits -- we should be able to
    # do those in 10,000 orbit chunks in memory and gain a factor of 10,000 speed up.
    candidate_row_rejects_pern = [[]]
    candidate_row_chisquared_partials_pern = [sum_chisquared_partials_norejects]
    n_reject = max(min_n_reject, 1)
    while n_reject < max_n_reject:
        candidate_row_rejects = []
        candidate_row_chisquared_partials = []
        combinations = list(set(itertools.combinations(possible_rejects, int(n_reject))))
        for rows_to_reject in combinations:
            rows_to_keep[list(rows_to_reject)] = False
            # sum the square of the chi2 partials to decide for whether or not it is a stationary point.
            sum_chisquared_partials = np.sqrt(np.sum(np.sum(chi2_vector[rows_to_keep], axis=0) ** 2))
            candidate_row_rejects.append(rows_to_reject)
            candidate_row_chisquared_partials.append(sum_chisquared_partials)
            # reset for the next loop:
            rows_to_keep[list(rows_to_reject)] = True
        n_reject += 1
        candidate_row_rejects_pern.append(np.array(candidate_row_rejects)[np.argmin(candidate_row_chisquared_partials)])
        candidate_row_chisquared_partials_pern.append(np.min(candidate_row_chisquared_partials))
    # see if any of the rejections are viable (i.e., check if this IAD is messed up in an unrepairable way)
    if np.min(candidate_row_chisquared_partials_pern) > chi2_thresh:
        warnings.warn(f"Failed to find which observations of this DVD source {data.meta['star_id']} "
                      f"that should have been marked as rejected. "
                      f"The chi squared partials were larger than {chi2_thresh}. "
                      f"DVD source {data.meta['star_id']} is likely a source with corrupted data. "
                      f"Aborting rejection routine and using IAD as was "
                      f"read from the DVD data. ", UserWarning)    # pragma: no cover
        return {'residual/along_scan_error': [], 'orbit/scan_angle/time': []}
    # exclude any rejections that do not yield stationary points.
    viable_rejections = np.where(np.array(candidate_row_chisquared_partials_pern) < chi2_thresh)[0]
    candidate_row_rejects_pern = [candidate_row_rejects_pern[v] for v in viable_rejections]
    candidate_row_chisquared_partials_pern = [candidate_row_chisquared_partials_pern[v] for v in viable_rejections]
    # calculate f2 values for all the viable rejections
    candidate_row_f2_vals_pern = []
    data_minus_model_squared = ((data.residuals.values / data.along_scan_errs.values) ** 2)
    for r in candidate_row_rejects_pern:
        rows_to_keep[list(r)] = False
        chisquared = np.sum(data_minus_model_squared[rows_to_keep])
        candidate_row_f2_vals_pern.append(compute_f2(n_transits - nparam, chisquared))
        rows_to_keep[list(r)] = True
    # restrict viable choices to the one that best matches f2
    reject_idx = candidate_row_rejects_pern[np.argmin(np.abs(np.array(candidate_row_f2_vals_pern) - catalog_f2))]
    return {'residual/along_scan_error': list(reject_idx), 'orbit/scan_angle/time': list(reject_idx)}


def find_epochs_to_reject_java(data: DataParser, n_additional_reject):
    # Note there are degeneracies in the best epochs to reject. E.g. for hip 39, as long as the last
    #  residual is rejected, basically any of the 1426 orbits (Because they are all similar)
    #  can be rejected and they result in a very similar chisquared.
    possible_rejects = np.arange(len(data))
    # calculate the chisquared partials
    sin_scan = np.sin(data.scan_angle.values)
    cos_scan = np.cos(data.scan_angle.values)
    dt = data.epoch - 1991.25
    resid_reject_idx = [len(data) - 1 - i for i in range(int(n_additional_reject))]  # always reject the repeated observations.
    # need to iterate over popping orbit combinations
    orbits_to_keep = np.ones(len(data), dtype=bool)
    residuals_to_keep = np.ones(len(data), dtype=bool)
    residuals_to_keep[resid_reject_idx] = False

    residual_factors = (data.residuals.values / data.along_scan_errs.values ** 2)[residuals_to_keep]
    mask_rejected_resid = (data.along_scan_errs.values > 0).astype(bool)[residuals_to_keep]
    _orbit_factors = np.array([data.parallax_factors.values, sin_scan, cos_scan, dt * sin_scan, dt * cos_scan]).T
    # we should be able to do the orbit reject calculation fairly easily in memory.
    # for 100 choose 3 we have like 250,000 combinations of orbits -- we sghould be able to
    # do those in 10,000 orbit chunks in memory and gain a factor of 10,000 speed up.
    candidate_orbit_rejects = []
    candidate_orbit_chisquared_partials = []
    for orbit_to_reject in itertools.combinations(possible_rejects, int(n_additional_reject)):
        orbits_to_keep[list(orbit_to_reject)] = False
        # now we want to try a variety of deleting orbits and sliding the other orbits
        # upward to fill the vacancy.
        # this pops the orbits out and shifts all the orbits after:
        orbit_factors = _orbit_factors[orbits_to_keep].T
        # this simultaneously deletes one of the residuals, assigns the remaining residuals to the
        # shifted orbits, and calculates the chi2 partials vector per orbit:
        chi2_vector = (2 * residual_factors * orbit_factors).T
        # sum the square of the chi2 partials to decide for whether or not it is a stationary point.
        sum_chisquared_partials = np.sqrt(np.sum(np.sum(chi2_vector[mask_rejected_resid], axis=0) ** 2))
        candidate_orbit_rejects.append(orbit_to_reject)
        candidate_orbit_chisquared_partials.append(sum_chisquared_partials)
        # reset for the next loop:
        orbits_to_keep[list(orbit_to_reject)] = True
    orbit_reject_idx = np.array(candidate_orbit_rejects)[np.argmin(candidate_orbit_chisquared_partials)]
    if np.min(candidate_orbit_chisquared_partials) > 0.5:
        warnings.warn(f"Completed the ad-hoc correction for java tool source {data.meta['star_id']}, "
                      f"but the chisquared partials are "
                      "still larger than 0.5. Treat the results of this "
                      "source with caution.", UserWarning)    # pragma: no cover

    return {'residual/along_scan_error': list(resid_reject_idx),
            'orbit/scan_angle/time': list(orbit_reject_idx)}


def find_epochs_to_reject_java_large(data: DataParser, n_additional_reject, orbit_number):
    # this is for any java tool object where n_additional_reject is greater than 3.
    # we assume the scan angles and times of rows in the same orbit are similar, therefore we only have
    # to try all combinations of distributing n_additional_reject rejected epochs among N orbits
    # calculate the chisquared partials
    orbit_prototypes, orbit_index, orbit_multiplicity = np.unique(orbit_number, return_index=True, return_counts=True)
    num_unique_orbits = len(orbit_prototypes)
    sin_scan = np.sin(data.scan_angle.values)
    cos_scan = np.cos(data.scan_angle.values)
    dt = data.epoch - 1991.25
    resid_reject_idx = [len(data) - 1 - i for i in range(int(n_additional_reject))]  # always reject the repeated observations.
    # need to iterate over popping orbit combinations
    orbits_to_keep = np.zeros(len(data), dtype=bool)
    residuals_to_keep = np.ones(len(data), dtype=bool)
    residuals_to_keep[resid_reject_idx] = False

    residual_factors = (data.residuals.values / data.along_scan_errs.values ** 2)[residuals_to_keep]
    mask_rejected_resid = (data.along_scan_errs.values > 0).astype(bool)[residuals_to_keep]
    _orbit_factors = np.array([sin_scan, cos_scan, dt * sin_scan, dt * cos_scan]).T
    # we should be able to do the orbit reject calculation fairly easily in memory.
    # for 100 choose 3 we have like 250,000 combinations of orbits -- we sghould be able to
    # do those in 10,000 orbit chunks in memory and gain a factor of 10,000 speed up.
    candidate_orbit_rejects = []
    candidate_orbit_chisquared_partials = []
    for rejects_from_each_orbit in partitions(n_additional_reject, num_unique_orbits):
        if np.any(rejects_from_each_orbit > orbit_multiplicity):
            # ignore any trials of rejects that put e.g. 10 rejects into an orbit with only 4 observations.
            continue
        end_index = orbit_index + orbit_multiplicity - np.array(rejects_from_each_orbit)
        for s, e in zip(orbit_index, end_index):
            orbits_to_keep[s:e] = True
        # now we want to try a variety of deleting orbits and sliding the other orbits
        # upward to fill the vacancy.
        # this pops the orbits out and shifts all the orbits after:
        orbit_factors = _orbit_factors[orbits_to_keep].T
        # this simultaneously deletes one of the residuals, assigns the remaining residuals to the
        # shifted orbits, and calculates the chi2 partials vector per orbit:
        chi2_vector = (2 * residual_factors * orbit_factors).T
        # sum the square of the chi2 partials to decide for whether or not it is a stationary point.
        sum_chisquared_partials = np.sqrt(np.sum(np.sum(chi2_vector[mask_rejected_resid], axis=0) ** 2))
        candidate_orbit_rejects.append(rejects_from_each_orbit)
        candidate_orbit_chisquared_partials.append(sum_chisquared_partials)
        # reset for the next loop:
        orbits_to_keep[:] = False
    rejects_from_each_orbit = np.array(candidate_orbit_rejects)[np.argmin(candidate_orbit_chisquared_partials)]
    # now transform rejects_from_each_orbit into actual orbit indices that we are going to reject.
    end_index = orbit_index + orbit_multiplicity - np.array(rejects_from_each_orbit)
    for s, e in zip(orbit_index, end_index):
        orbits_to_keep[s:e] = True
    orbit_reject_idx = np.where(~orbits_to_keep)[0]
    if np.min(candidate_orbit_chisquared_partials) > 0.5:
        warnings.warn(f"Completed the ad-hoc correction for java tool source {data.meta['star_id']}, "
                      f"but the chisquared partials are "
                      "still larger than 0.5. Treat the results of this "
                      "source with caution.", UserWarning)    # pragma: no cover

    return {'residual/along_scan_error': list(resid_reject_idx),
            'orbit/scan_angle/time': list(orbit_reject_idx)}


def partitions(n, k):
    """
    yield all possible weighs to distribute n rejected rows among k orbits.
    This is just the solution to the "stars and bars" problem.
    Theorem 2: https://en.wikipedia.org/wiki/Stars_and_bars_%28combinatorics%29

    From https://stackoverflow.com/questions/28965734/general-bars-and-stars
    """
    for c in itertools.combinations(range(n+k-1), k-1):
        yield [b-a-1 for a, b in zip((-1,)+c, c+(n+k-1,))]


def get_nparam(nparam_header_val):
    # strip the solution type (5, 7, or 9) from the solution type, which is a number 10xd+s consisting of
    # two parts: d and s. see Note 1 on Vizier for the Hipparcos re-reduction.
    return int(str(int(nparam_header_val))[-1])


def compute_f2(nu, chisquared):
    # equation B.2 of D. Michalik et al. 2014. Joint astrometric solution of Hipparcos and Gaia
    return (9*nu/2)**(1/2)*((chisquared/nu)**(1/3) + 2/(9*nu) - 1)
