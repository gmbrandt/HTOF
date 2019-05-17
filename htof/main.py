"""
Driver script for htof.
The Fitter class is what a user should use to both parse intermediate data and fit data
to the intermediate epochs.
"""

import numpy as np
import warnings

from htof.fit import AstrometricFitter
from htof.parse import HipparcosRereductionData, GaiaData, HipparcosOriginalData, fractional_year_epoch_to_jd


class Astrometry(object):
    parsers = {'GaiaDR2': GaiaData, 'Hip1': HipparcosOriginalData, 'Hip2': HipparcosRereductionData}

    def __init__(self, data_choice, star_id, intermediate_data_directory, fitter=None, data=None):
        if data is None:
            DataParser = self.parsers[data_choice]
            data = DataParser()
            data.parse(star_id=star_id,
                       intermediate_data_directory=intermediate_data_directory)
            data.calculate_inverse_covariance_matrices(cross_scan_along_scan_var_ratio=1E5)
        if fitter is None and data is not None:
            fitter = AstrometricFitter(inverse_covariance_matrices=data.inverse_covariance_matrix,
                                       epoch_times=data.julian_day_epoch())
        self.data = data
        self.fitter = fitter

    def fit(self, ra_vs_epoch, dec_vs_epoch, central_epoch_dec=0, central_epoch_ra=0, central_epoch_fmt='MJD'):
        solution_vector = self.fitter.fit_line(ra_vs_epoch=ra_vs_epoch,
                                               dec_vs_epoch=dec_vs_epoch)
        solution_vector = self._shift_to_central_epoch(solution_vector, central_epoch_dec,
                                                       central_epoch_ra, central_epoch_fmt)
        return solution_vector

    @staticmethod
    def _shift_to_central_epoch(solution_vector, central_epoch_dec, central_epoch_ra, central_epoch_fmt):
        if central_epoch_fmt == 'frac_year':
            if central_epoch_dec > 3000 or central_epoch_ra > 3000:
                warnings.warn('central epoch in RA or DEC was chosen to be > 3000. Are you sure this'
                              'is a fractional year date and not a MJD? If MJD, set central_epoch_fmt=MJD.',
                              UserWarning)
            central_epoch_dec = fractional_year_epoch_to_jd(central_epoch_dec, half_day_correction=True)
            central_epoch_ra = fractional_year_epoch_to_jd(central_epoch_ra, half_day_correction=True)

        ra0, dec0, mu_ra, mu_dec = solution_vector
        ra0 += mu_ra * central_epoch_ra
        dec0 += mu_dec * central_epoch_dec
        return np.array([ra0, dec0, mu_ra, mu_dec])
