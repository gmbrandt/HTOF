htof
===============

This repo contains htof, the package for parsing intermediate data from the Gaia and
Hipparcos Satellites, and reproducing linear fits to their astrometry.

.. image:: https://coveralls.io/repos/github/gmbrandt/HTOF/badge.svg?branch=master
    :target: https://coveralls.io/github/gmbrandt/HTOF?branch=master

.. image:: https://travis-ci.org/gmbrandt/HTOF.svg?branch=master
    :target: https://travis-ci.org/gmbrandt/HTOF

Parallax is handled by the :code:`sky_path` module which was written by Anthony Brown
as a part of his astrometric-sky-path package: https://github.com/agabrown/astrometric-sky-path/

Installation
------------
htof can be installed in the usual way, by running

.. code-block:: bash

    pip install .

While in the root directory of this repo.

Usage: Four parameter fits
--------------------------
The following examples show how one would both load in and fit a line to the astrometric intermediate data
from either Hipparcos data reduction or Gaia (Currently only data release 2, GaiaDR2). A four parameter fit means
you are fitting a line to the data, without parallax.

Let ra_vs_epoch, dec_vs_epoch be 1d arrays of ra and dec positions.
Assume we want to fit to data from GaiaDR2 on the star with hip id 027321. The choices of data
are :code:`GaiaDR2`, :code:`Hip1` and :code:`Hip2`. The following lines parse the intermediate data and fit a line.

.. code-block:: python

    from htof.main import Astrometry
    fitter = Astrometry('GaiaDR2', star_id='027321', 'path/to/intermediate_data/', format='jd')  # parse
    ra0, dec0, mu_ra, mu_dec = fitter.fit(ra_vs_epoch, dec_vs_epoch)  # fit

ra_vs_epoch and dec_vs_epoch are the positions in right ascension and declination of the object.
These arrays must have the same shape as fitter.data.julian_day_epoch(),
which are the epochs in the intermediate data. :code:`format='jd'` specifies
the time units of the output best fit parameters. The possible choices of format
are the same as the choices for format in astropy.time.Time(val, format=format).
E.g. :code:`'decimalyear'`, :code:`'jd'` . If :code:`format='decimalyear'`, then the output :code:`mu_ra`
would have units of mas/year. If :code:`jd` then the output is mas/day. Both Hipparcos and Gaia catalogs list parallaxes
in milli-arcseconds (mas), and so positional units are always in mas for HTOF.

For Hipparcos 2, the path to the intermediate data would point to :code:`IntermediateData/resrec/`.
Note that the intermediate data files must be in the same format as the test intermediate data files found in this
repository under :code:`htof/test/data_for_tests/`. The best fit parameters have units of mas and mas/day by default.
The best fit skypath for right ascension is then :code:`ra0 + mu_ra * epochs`

If you want to specify a central epoch, you can do so with:

.. code-block:: python

    from htof.main import Astrometry

    fitter = Astrometry('GaiaDR2', star_id='027321', 'path/to/intermediate_data/', central_epoch_ra=2456892, central_epoch_dec=2456892, format='jd')
    ra0, dec0, mu_ra, mu_dec = fitter.fit(ra_vs_epoch, dec_vs_epoch)

The format of the central epochs must be specified along with the central epochs. The best fit sky path in right ascension would then be
:code:`ra0 + mu_ra * (epochs - centra_epoch_ra)`.

One can access the BJD epochs with

.. code-block:: python

    fitter.central_epoch_dec
    fitter.central_epoch_ra

If you want the standard (1-sigma) errors on the parameters, set :code:`return_all=True` when fitting:

.. code-block:: python

    from htof.main import Astrometry

    fitter = Astrometry('GaiaDR2', star_id='027321', 'path/to/intermediate_data/', central_epoch_ra=2456892, central_epoch_dec=2456892, format='jd')
    coeffs, errors = fitter.fit(ra_vs_epoch, dec_vs_epoch, return_all=True)

errors is an array the same shape as coeffs, where each entry is the 1-sigma error for the
parameter at the same location in the coeffs array. For Hip1 and Hip2, HTOF loads in the real
catalog errors and so these parameter error estimates should match those given in the catalog. However,
for Gaia we do not have the error estimates from the GOST tool and so the best-fit parameter errors to
Gaia will not match those reported by the Gaia members.

Usage: 5,7 and 9 parameter fits
-------------------------------
TODO: Discuss how to get parallax, how to generate the plx pertubations
changed format that the fitter returns (should it ever return a different format?, or should
it always be [0, ra0, dec0, mura, mudec, 0,] for higher order fits?

Appendix
--------
This section describes how to reproduce the fit from Astrometry.fit from the Usage section. The
Astrometry object is essentially just a wrapper for data parsing and fitting all in one.

.. code-block:: python

    from htof.parse import HipparcosOriginalData # or GaiaData or HipparcosReReduction
    data = HipparcosOriginalData()
    data.parse(star_id='049699', intermediate_data_directory='Hip1/IntermediateData/')
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