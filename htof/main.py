"""
Driver script for htof.
The Fitter class is what a user should use to both parse intermediate data and fit data
to the intermediate epochs.
"""

import numpy as np
import warnings

from htof.fit import AstrometricFitter
from htof.parse import HipparcosRereductionData, GaiaData, HipparcosOriginalData


class Astrometry(object):
    parsers = {'GaiaDR2': GaiaData, 'Hip1': HipparcosOriginalData, 'Hip2': HipparcosRereductionData}

    def __init__(self, data_choice, star_id, intermediate_data_directory, fitter=None, data=None,
                 central_epoch_ra=0, central_epoch_dec=0, central_epoch_fmt='BJD'):
        if data is None:
            DataParser = self.parsers[data_choice]
            data = DataParser()
            data.parse(star_id=star_id,
                       intermediate_data_directory=intermediate_data_directory)
            data.calculate_inverse_covariance_matrices(cross_scan_along_scan_var_ratio=1E5)
        if fitter is None and data is not None:
            fitter = AstrometricFitter(inverse_covariance_matrices=data.inverse_covariance_matrix,
                                       epoch_times=data.julian_day_epoch(),
                                       central_epoch_dec=central_epoch_dec,
                                       central_epoch_ra=central_epoch_ra,
                                       central_epoch_fmt=central_epoch_fmt)
        self.data = data
        self.fitter = fitter
        # NOTE: AstrometricFitter stores central_epoch_dec and central_epoch_fmt as 'BJD'.
        # If central_epoch_fmt = 'frac_year', AstrometricFitter will convert them to BJD first.

    def fit(self, ra_vs_epoch, dec_vs_epoch, pm_units='mas_per_day'):
        solution_vector = self.fitter.fit_line(ra_vs_epoch=ra_vs_epoch,
                                               dec_vs_epoch=dec_vs_epoch)
        if pm_units == 'mas_per_day':
            return solution_vector
        if pm_units == 'mas_per_year':
            return solution_vector * np.array([1, 1, 365.25, 365.25])
