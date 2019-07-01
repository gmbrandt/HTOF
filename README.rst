htof
===============

This repo contains htof, the package for parsing intermediate data from the Gaia and
Hipparcos Satellites, and reproducing linear fits to their astrometry.

.. image:: https://coveralls.io/repos/github/gmbrandt/HTOF/badge.svg?branch=master
    :target: https://coveralls.io/github/gmbrandt/HTOF?branch=master

.. image:: https://travis-ci.org/gmbrandt/HTOF.svg?branch=master
    :target: https://travis-ci.org/gmbrandt/HTOF


Installation
------------
htof can be installed in the usual way, by running

.. code-block:: bash

    pip install .

While in the root directory of this repo.

Usage
-----
The following examples show how one would both load in and fit a line to the astrometric intermediate data
from either Hipparcos data reduction or Gaia (Currently only data release 2, GaiaDR2).

Let ra_vs_epoch, dec_vs_epoch be 1d arrays of ra and dec positions.
Assume we want to fit to data from GaiaDR2 on the star with hip id 027321. The choices of data
are :code:`GaiaDR2`, :code:`Hip1` and :code:`Hip2`. The following lines parse the intermediate data and fit a line.

.. code-block:: python

    from htof.main import Astrometry
    fitter = Astrometry('GaiaDR2', star_id='027321', 'path/to/intermediate_data/')  # parse
    ra0, dec0, mu_ra, mu_dec = fitter.fit(ra_vs_epoch, dec_vs_epoch)  # fit

For Hipparcos 2, the path to the intermediate data would point to :code:`IntermediateData/resrec/`.
Note that the intermediate data files must be in the same format as the test intermediate data files found in this
repository under :code:`htof/test/data_for_tests/`.

If you want to specify a central epoch in barycentric Julian day, instead call:

.. code-block:: python

    from htof.main import Astrometry

    fitter = Astrometry('GaiaDR2', star_id='027321', 'path/to/intermediate_data/', central_epoch_ra=2456892, central_epoch_dec=2456892, central_epoch_fmt='BJD')
    ra0, dec0, mu_ra, mu_dec = fitter.fit(ra_vs_epoch, dec_vs_epoch)

The above would set the central epoch for the right ascension (ra) to 2456892 BJD, and declination (dec) to 2456892 BJD.
One could also set the central epochs to years using the :code:`frac_year` keyword and supplying a year:

.. code-block:: python

    from htof.main import Astrometry
    fitter = Astrometry('GaiaDR2', star_id='027321', 'path/to/intermediate_data/', central_epoch_ra=2000, central_epoch_dec=2000, central_epoch_fmt='frac_year')
    ra0, dec0, mu_ra, mu_dec = fitter.fit(ra_vs_epoch, dec_vs_epoch)

One can then access the BJD central epochs via

.. code-block:: python

    fitter.central_epoch_dec
    fitter.central_epoch_ra

Both Hipparcos and Gaia catalogs list parallaxes in milli-arcseconds (mas). We convert all three
catalog epochs to barycentric julian day by default, therefore a fit to astrometry has proper motions
with units of mas/day by default. If you want mas/year, then use the keyword :code:`pm_units` (proper motion units):

.. code-block:: python

    ra0, dec0, mu_ra, mu_dec = fitter.fit(ra_vs_epoch, dec_vs_epoch, pm_units='mas_per_year')

Which will return the same mu_ra and mu_dec as with :code:`pm_units='mas_per_day'` but multiplied by 365.25.

The following appendix describes in more detail how to perform the above operations without
using the Astrometry object, if you ever desired to do so.

Appendix
--------
This section describes how to reproduce the fit from Astrometry.fit from the Usage section. The
Astrometry object is essentially just a wrapper for data parsing and fitting all in one.

.. code-block:: python

    from htof.parse import HipparcosOriginalData # or GaiaData or HipparcosReReduction
    data = HipparcosOriginalData()
    data.parse(star_id='049699', intermediate_data_directory='Hip1/IntermediateData/)
    data.calculate_inverse_covariance_matrices()

data now has a variety of intermediate data products such as the scan angles, the epochs when each
data point was collected, the inverse covariance matrices describing the errors of the scan,
and the BJD epochs accessible through :code:`data.julian_day_epoch()`.

Now to fit a line to the astrometry. Given a parsed data object, we simply call:

.. code-block:: python

    fitter = AstrometricFitter(inverse_covariance_matrices=data.inverse_covariance_matrix, epoch_times=data.julian_day_epoch())
    solution_vector = fitter.fit_line(ra_vs_epoch, dec_vs_epoch)
    ra0, dec0, mu_ra, mu_dec = solution_vector

where :code:`ra(mjd) = ra0 + mu_ra * mjd`, and same for declination.

License
-------

MIT License. See the LICENSE file for more information.