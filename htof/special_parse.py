import numpy as np
import warnings
from importlib.resources import files
from pandas import DataFrame, Series
from shutil import copy
from copy import deepcopy

from astropy.time import Time
from astropy.coordinates import Angle
from astropy import units as u
from scipy import stats, special
from astropy.table import Table
from htof.fit import AstrometricFitter
from htof.parse import HipparcosRereductionJavaTool, HipparcosRereductionDVDBook, DataParser


class Hipparcos2Recalibrated(HipparcosRereductionJavaTool):
    """
    A parser class which re-calibrates the hipparcos 2 data according to Brandt et al. 2022

    This is a monstrous parser class because well it is doing a lot.

    Note that this "Parser" class actually refits the data. This is programmatically poorly set up,
    in the sense that we parse the data and fit it, mixing this class with an AstrometricFitter class... For now,
    this is OK because the objective of this class is mainly so other users can save and inspect the recalibrated
    hipparcos 2 IAD.

    """
    EPOCHREJECTLIST = Table.read(files('htof') / 'data' / 'epoch_reject_shortlist.csv', format='ascii')

    def __init__(self, scan_angle=None, epoch=None, residuals=None, inverse_covariance_matrix=None,
                 along_scan_errs=None, meta=None, residual_offset=0.141, cosmic_dispersion=2.25):
        super(Hipparcos2Recalibrated, self).__init__(scan_angle=scan_angle, along_scan_errs=along_scan_errs,
                                                     epoch=epoch, residuals=residuals,
                                                     inverse_covariance_matrix=inverse_covariance_matrix,
                                                     meta=meta)
        self.residual_offset = residual_offset
        self.cosmic_dispersion = cosmic_dispersion
        self.recalibrated_header = None
        self.recalibrated_data = None

    def parse(self, star_id, intermediate_data_directory, attempt_adhoc_rejection=True,
              reject_known=True, **kwargs):
        # important that the 2007 error inflation is turned off (error_inflate=False)
        header, raw_data = super(Hipparcos2Recalibrated, self).parse(star_id, intermediate_data_directory,
                                                                     error_inflate=False,
                                                                     attempt_adhoc_rejection=attempt_adhoc_rejection,
                                                                     reject_known=reject_known, **kwargs)
        apply_calibrations = True
        if not (self.meta['catalog_soltype'] == 5 or self.meta['catalog_soltype'] == 7 or self.meta['catalog_soltype'] == 9):
            warnings.warn(f'This source has a solution type of {self.meta["catalog_soltype"]}. '
                          f'htof will only recalibrate 5, 7, and 9 parameter solutions currently. '
                          f'No recalibration will be performed.')
            apply_calibrations = False
        if attempt_adhoc_rejection is False or reject_known is False:
            warnings.warn('We have not tested recalibration without rejecting any of the flagged or bugged '
                          'observations. No recalibration will be performed.')
            apply_calibrations = False

        if apply_calibrations:
            # apply the calibrations
            self.residuals += self.residual_offset  # note that this modifies the raw_data column also.
            self.along_scan_errs = np.sqrt(self.along_scan_errs**2 + self.cosmic_dispersion**2)
            self.calculate_inverse_covariance_matrices()
            # munge the parallax factors into the correct form. Note that we are using
            # the parallax factors from the catalog here to keep everything consistent.
            ra_motion, dec_motion = to_ra_dec_basis(self.parallax_factors.values, self.scan_angle.values)
            parallactic_perturbations = {'ra_plx': ra_motion, 'dec_plx': dec_motion}
            # refit the data, calculate the new residuals, and parameters.
            fit_degree = {5: 1, 7: 2, 9: 3}[int(self.meta['catalog_soltype'])]
            fitter = AstrometricFitter(inverse_covariance_matrices=self.inverse_covariance_matrix,
                                       epoch_times=Time(Time(self.julian_day_epoch(), format='jd'), format='jyear').value,
                                       central_epoch_dec=1991.25,
                                       central_epoch_ra=1991.25,
                                       fit_degree=fit_degree,
                                       use_parallax=True,
                                       parallactic_pertubations=parallactic_perturbations)
            # get residuals in ra and dec.
            ra = Angle(self.residuals.values * np.sin(self.scan_angle.values), unit='mas')
            dec = Angle(self.residuals.values * np.cos(self.scan_angle.values), unit='mas')
            # fit the residuals
            coeffs, errors, Q, new_residuals = fitter.fit_line(ra.mas, dec.mas, return_all=True)
            # compute the along-scan residuals
            new_residuals = to_along_scan_basis(new_residuals[:, 0], new_residuals[:, 1], self.scan_angle.values)
            self.residuals = Series(new_residuals, index=self.residuals.index)
            """
            update the header with the new statistics
            """
            ntransits, nparam = len(self), int(self.meta['catalog_soltype'])
            header['second']['NOB'] = len(self)
            header['second']['NR'] = 0  # because we automatically remove any "flagged as rejected" observations.
            header['first']['F1'] = 0
            header['first']['NRES'] = len(self)
            header['first']['F2'] = special.erfcinv(stats.chi2.sf(Q, ntransits - nparam)*2)*np.sqrt(2)
            if np.isfinite(header['first']['F2']):
                header['first']['F2'] = np.round(header['first']['F2'], 4)
            # update the best fit parameters with the new values
            # dpmRA  dpmDE  e_dpmRA  e_dpmDE  ddpmRA  ddpmDE  e_ddpmRA  e_ddpmDE
            for i, key, dp, in zip(np.arange(nparam),
                                   ['Plx', 'RAdeg', 'DEdeg', 'pm_RA', 'pm_DE', 'dpmRA', 'dpmDE', 'ddpmRA', 'ddpmDE'],
                                   [2, 8, 8, 2, 2, 2, 2, 2, 2]):
                if key != 'RAdeg' and key != 'DEdeg':
                    header['third'][key] = np.round(header['third'][key] + coeffs[i], dp)
                else:
                    # convert the perturbation from mas to degrees.
                    header['third'][key] = np.round(header['third'][key] + (coeffs[i] * u.mas).to(u.degree).value, dp)
            # update the errors with the new errors
            for i, key, dp, in zip(np.arange(nparam),
                                   ['e_Plx', 'e_RA', 'e_DE', 'e_pmRA', 'e_pmDE', 'e_dpmRA', 'e_dpmDE', 'e_ddpmRA', 'e_ddpmDE'],
                                   [2, 2, 2, 2, 2, 2, 2, 2, 2]):
                header['third'][key] = np.round(errors[i], dp)
            # save the modified header to the class, because these will be
            # used by self.write_as_javatool_format()
            self.recalibrated_header = deepcopy(header)
            """
            update the raw_data columns with the new data. Note that rejected/bugged epochs are already
            taken care of.
            """
            # data order in Java tool data: IORB   EPOCH    PARF    CPSI    SPSI     RES   SRES
            recalibrated_data = DataFrame({'1': self._iorb, '2': self._epoch - 1991.25, '3': self.parallax_factors,
                                           '4': self._cpsi, '5': self._spsi, '6': np.round(self.residuals.values, 3),
                                           '7': np.round(self.along_scan_errs.values, 3)})
            self.recalibrated_data = recalibrated_data
            header, raw_data = None, None  # the raw header and raw data have been modified, so clear them.
        return header, raw_data

    def write_as_javatool_format(self, path: str):
        """
        Variant of the .write() method that is specially for this Parser. It writes out the fixed data
        with the exact same format as the input JavaTool IAD (i.e., the .d files).
        We recommend using the normal .write() format. This method is for users who do not want to adopt the
        htof file format output by the normal .write() format.

        :param: path: path to write the file to. filename should end with .d . ".d" is the same file ending as the
        java tool IAD. But in principle, most common ascii file extensions should work (e.g., ".txt" etc).
        """
        if self.recalibrated_header is None or self.recalibrated_data is None:
            warnings.warn('This source was NOT recalibrated, see earlier warnings as to why. Will not save any '
                          f'output file at {path} because no recalibration was done.')
            return None
        # order of the header values:
        ks1 = ['HIP', 'MCE', 'NRES', 'NC', 'isol_n', 'SCE', 'F2', 'F1']
        ks2 = ['Hp', 'B-V', 'VarAnn', 'NOB', 'NR']
        ks3 = ['RAdeg', 'DEdeg', 'Plx', 'pm_RA', 'pm_DE', 'e_RA', 'e_DE', 'e_Plx', 'e_pmRA', 'e_pmDE',
               'dpmRA', 'dpmDE', 'e_dpmRA', 'e_dpmDE', 'ddpmRA', 'ddpmDE', 'e_ddpmRA', 'e_ddpmDE',
               'upsRA', 'upsDE', 'e_upsRA', 'e_upsDE', 'var']
        # byte entries of each header value. Stored as (start, stop) inclusive intervals.
        # the exact bytes are stored, according to the IAD readme, so that the output format is exactly the same.
        bs1 = [(3, 8), (10, 15), (17, 19), (22, 22), (25, 27), (32, 35), (37, 42), (44, 45)]
        bs2 = [(3, 9), (11, 16), (18, 18), (25, 27), (29, 31)]
        bs3 = [(3, 14), (16, 27), (29, 36), (38, 45), (47, 54),
               (56, 61), (63, 68), (70, 75), (77, 82), (84, 89), (91, 96),
               (98, 103), (105, 110), (114, 119), (123, 128), (131, 136), (139, 144),
               (149, 154), (159, 164), (167, 172), (175, 180), (184, 189), (193, 198)]
        header_template_fpath = files('htof') / 'data' / 'hip2_recalibrated_header.txt'
        copy(header_template_fpath, path)  # copy the template file to the output path.
        # populate the header lines.
        f = open(path, 'r')
        lines = f.readlines()
        for idx, hline, ks, bs, line_length in zip([6, 8, 10], ['first', 'second', 'third'],
                                                   [ks1, ks2, ks3], [bs1, bs2, bs3], [48, 34, 201]):
            matter = ["#"] + [" "] * (line_length - 1)  # strings are immutable in python. This is a gross solution to get past that.
            for key, byte_start_stop in zip(ks, bs):
                minn, maxx = min(byte_start_stop)-1, max(byte_start_stop)-1  # -1 because python indexes from 0.
                final_len = 1 + maxx-minn
                value = str(self.recalibrated_header[hline][key])
                matter[minn: maxx + 1] = list(value[:final_len].ljust(final_len, " "))
            matter = "".join(matter) # convert to a string instead of a list of characters.
            lines[idx] = matter + '\n'
            lines[idx] = lines[idx].replace('nan', '---')
        lines[-1] += '\n'
        # populate the IAD entries
        iad_bytes = [(3, 6), (8, 14), (16, 22), (24, 30), (32, 38), (42, 46), (49, 53)]
        significance = [None, 4, 4, 4, 4, 2, 2]  # None means values in that column will not be rounded.
        #
        line_length = 56
        for line in self.recalibrated_data.to_numpy():
            matter = [" "] * line_length  # strings are immutable in python. This is a gross solution to get past that.
            for col, value, byte_start_stop, sig in zip(np.arange(7), line, iad_bytes, significance):
                minn, maxx = min(byte_start_stop) - 1, max(byte_start_stop) - 1  # -1 because python indexes from 0.
                final_len = 1 + maxx - minn
                if col != 0:
                    # left justify every column except the orbit column. Round floats also.
                    value = value if sig is None else np.round(value, sig)
                    value = str(value) if value < 0 else ' ' + str(value)  # left pad with a space if it is positive.
                    matter[minn: maxx + 1] = list(value[:final_len].ljust(final_len, " "))
                else:
                    # right justify the orbit column
                    value = str(int(value))
                    matter[minn: maxx + 1] = list(value[:final_len].rjust(final_len, " "))
            matter = "".join(matter) + '\n'  # convert to a string instead of a list of characters.
            lines.append(matter)
        # write it out to file.
        f = open(path, 'w')
        f.writelines(lines)
        f.close()
        return None


class Hipparcos2ParserFactory:
    """
    A factory method for HipparcosRereductionDVDBook and HipparcosRereductionJavaTool. It detects
    which format the IAD is in, then chooses and returns the appropriate parser.
    """

    @staticmethod
    def get_appropriate_parser(filepath):
        if filepath == '':
            return HipparcosRereductionJavaTool
        datatype = get_datatype(filepath)
        if datatype == 'hip2dvd':
            return HipparcosRereductionDVDBook
        if datatype == 'hip2javatool':
            return HipparcosRereductionJavaTool

    @classmethod
    def parse_and_instantiate(cls, star_id: str, intermediate_data_directory: str, **kwargs):
        filepath, msg = DataParser.get_intermediate_data_file_path(star_id, intermediate_data_directory)
        CorrectDataParser = cls.get_appropriate_parser(filepath)
        return CorrectDataParser.parse_and_instantiate(star_id, intermediate_data_directory, **kwargs)


def to_ra_dec_basis(value, scan_angle):
    """
    Convert values (e.g. residuals) along the direction of the scan to the same value in RA and Dec. I.e. assume the value
    has a zero across_scan component and all of the value is in the along-scan direction, then convert that value to a vector in
    RA and Dec.
    these maths are just from https://en.wikipedia.org/wiki/Rotation_of_axes  .
    """
    dec_value, ra_value = value * np.cos(scan_angle), value * np.sin(scan_angle)
    return ra_value, dec_value


def to_along_scan_basis(ra_value, dec_value, scan_angle):
    """
    Convert values (e.g. residuals) in RA and Dec to the same value along the direction of the scan. I.e convert from RA, DEC to
    Along-scan , Across-scan basis, then keep only the along-scan component.
    these maths are just from https://en.wikipedia.org/wiki/Rotation_of_axes  .

    note that the inputs ra_value and dec_value may have nonzero across scan components. This function will zero
    out the cross-scan component. to_ra_dec_basis and to_along_scan_basis() are ONLY inverse transforms for data that
    has components solely in the across-scan direction.
    """
    along_scan_value = dec_value * np.cos(scan_angle) + ra_value * np.sin(scan_angle)
    return along_scan_value


def get_datatype(filepath):
    with open(filepath) as f:
        lines = f.readlines()
    if 'This file contains residual records' in lines[0]:
        datatype = 'hip2javatool'
    else:
        datatype = 'hip2dvd'
    return datatype

