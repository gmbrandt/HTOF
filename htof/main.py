"""
Driver script for htof.
The Astrometry class is what a user should use to both parse and fit the intermediate astrometric data.

Author: G. Mirek Brandt
"""

from astropy.time import Time
from astropy.coordinates import Angle
import warnings

from htof.fit import AstrometricFitter
from htof.parse import GaiaeDR3, GaiaDR2, GaiaData
from htof.parse import HipparcosOriginalData, HipparcosRereductionJavaTool, HipparcosRereductionDVDBook
from htof.sky_path import parallactic_motion, earth_ephemeris, earth_sun_l2_ephemeris


class Astrometry(object):
    parsers = {'gaiaedr3': GaiaeDR3, 'gaiadr2': GaiaDR2, 'gaia': GaiaData, 'hip21': HipparcosRereductionJavaTool,
               'hip1': HipparcosOriginalData, 'hip2': HipparcosRereductionDVDBook}
    ephemeri = {'gaiadr2': earth_sun_l2_ephemeris, 'gaia': earth_sun_l2_ephemeris,
                'hip1': earth_ephemeris, 'hip2': earth_ephemeris, 'hip21': earth_ephemeris}

    def __init__(self, data_choice, star_id, intermediate_data_directory, fitter=None, data=None,
                 central_epoch_ra=0, central_epoch_dec=0, format='jd', fit_degree=1,
                 use_parallax=False, central_ra=None, central_dec=None, normed=False):

        if data is None:
            DataParser = self.parsers[data_choice.lower()]
            data = DataParser()
            data.parse(star_id=star_id,
                       intermediate_data_directory=intermediate_data_directory)
            data.calculate_inverse_covariance_matrices()

        parallactic_pertubations = None
        if use_parallax and isinstance(central_ra, Angle) and isinstance(central_dec, Angle):
            if central_epoch_dec != central_epoch_ra:
                warnings.warn('central_epoch_dec != central_epoch_ra. '
                              'Using central_epoch_ra as the central_epoch to compute the parallax motion.',
                              UserWarning)    # pragma: no cover
            ra_motion, dec_motion = parallactic_motion(Time(data.julian_day_epoch(), format='jd').jyear,
                                                       central_ra.mas, central_dec.mas, 'mas',
                                                       Time(central_epoch_ra, format=format).jyear,
                                                       ephemeris=self.ephemeri[data_choice.lower()])
            parallactic_pertubations = {'ra_plx': ra_motion, 'dec_plx': dec_motion}

        if fitter is None and data is not None:
            fitter = AstrometricFitter(inverse_covariance_matrices=data.inverse_covariance_matrix,
                                       epoch_times=Time(Time(data.julian_day_epoch(), format='jd'), format=format).value,
                                       central_epoch_dec=Time(central_epoch_dec, format=format).value,
                                       central_epoch_ra=Time(central_epoch_ra, format=format).value,
                                       fit_degree=fit_degree,
                                       use_parallax=use_parallax,
                                       parallactic_pertubations=parallactic_pertubations,
                                       normed=normed)
        self.data = data
        self.fitter = fitter

    def fit(self, ra_vs_epoch, dec_vs_epoch, return_all=False):
        return self.fitter.fit_line(ra_vs_epoch=ra_vs_epoch, dec_vs_epoch=dec_vs_epoch, return_all=return_all)

    def optimal_central_epochs(self):
        return {'ra': self.fitter.find_optimal_central_epoch('ra'), 'dec': self.fitter.find_optimal_central_epoch('dec')}
