The following is example code for how one would both load in and fit a line to astrometric intermediate data
from Hipparcos and Gaia.

# Sec 1: Loading intermediate data and fitting a line to a given set of ra and dec positions.
# Let ra_vs_epoch, dec_vs_epoch be 1d arrays of ra and dec positions.
# Assume we want to fit to data from GaiaDR2 on the star with hip id 027321. The choices of data are
# 'GaiaDR2', 'Hip1' and 'Hip2'. The following lines parse the intermediate data and fit a line.

from htof.main import Astrometry

fitter = Astrometry('GaiaDR2', star_id='027321', 'path/to/intermediate_data/')  # parse
ra0, dec0, mu_ra, mu_dec = fitter.fit(ra_vs_epoch, dec_vs_epoch)  # fit

# If you want to specify a central epoch in fractional year, instead call (after instantiating the fitter):

ra0, dec0, mu_ra, mu_dec = fitter.fit(ra_vs_epoch, dec_vs_epoch, central_epoch_dec=2015.25,
                                      central_epoch_ra=2015.5, central_epoch_fmt='frac_year')
# this converts internally the fractional years to MJD, so specifying the central epochs in MJD is
# marginally faster. In that case you would call:

ra0, dec0, mu_ra, mu_dec = fitter.fit(ra_vs_epoch, dec_vs_epoch, central_epoch_dec=mjd_date_2,
                                      central_epoch_ra=mjd_date_1)

# The following sections describe how to reproduce the fit from Astrometry.fit, which is essentially
# just a wrapper for data parsing and fitting all in one.

# Sec 2: Loading data
from htof.parse import HipparcosOriginalData # or GaiaData or HipparcosReReduction
data = HipparcosOriginalData()
data.parse(star_id='049699',
           intermediate_data_directory='Hip1/IntermediateData/)
data.calculate_inverse_covariance_matrices()

# data now has a variety of intermediate data products such as the scan angles, the epochs when each
# data point was collected, the inverse covariance matrices describing the errors of the scan,
# and the MJD epochs accessible through data.julian_day_epoch() .

# Sec 3: Fitting a line to the astrometry. Given a parsed data object (described in Sec 1) to fit
# a line to the a given set of ra_vs_epoch, dec_vs_epoch positions we simply call:
    fitter = AstrometricFitter(inverse_covariance_matrices=data.inverse_covariance_matrix,
                               epoch_times=data.julian_day_epoch())
    solution_vector = fitter.fit_line(ra_vs_epoch, dec_vs_epoch)
    ra0, dec0, mu_ra, mu_dec = solution_vector
# where ra(mjd) = ra0 + mu_ra * mjd, and same for dec.