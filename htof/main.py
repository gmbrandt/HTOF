"""
Driver script for htof.
The Fitter class is what a user should use to both parse intermediate data and fit data
to the intermediate epochs.

Author: G. Mirek Brandt
"""

import numpy as np
from astropy.time import Time

from htof.fit import AstrometricFitter
from htof.parse import HipparcosRereductionData, GaiaData, HipparcosOriginalData


class Astrometry(object):
    parsers = {'GaiaDR2': GaiaData, 'Hip1': HipparcosOriginalData, 'Hip2': HipparcosRereductionData}

    def __init__(self, data_choice, star_id, intermediate_data_directory, fitter=None, data=None,
                 central_epoch_ra=0, central_epoch_dec=0, format='jd', norm=True):
        if data is None:
            DataParser = self.parsers[data_choice]
            data = DataParser()
            data.parse(star_id=star_id,
                       intermediate_data_directory=intermediate_data_directory)
            data.calculate_inverse_covariance_matrices(cross_scan_along_scan_var_ratio=1E5)
        if fitter is None and data is not None:
            fitter = AstrometricFitter(inverse_covariance_matrices=data.inverse_covariance_matrix,
                                       epoch_times=Time(Time(data.julian_day_epoch(), format='jd'), format=format).value,
                                       central_epoch_dec=Time(central_epoch_dec, format=format).value,
                                       central_epoch_ra=Time(central_epoch_ra, format=format).value,
                                       norm=norm)
        self.data = data
        self.fitter = fitter

    def fit(self, ra_vs_epoch, dec_vs_epoch, return_all=False):
        return self.fitter.fit_line(ra_vs_epoch=ra_vs_epoch, dec_vs_epoch=dec_vs_epoch, return_all=return_all)
