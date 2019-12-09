htof
===============

This repo contains htof, the package for parsing intermediate data from the Gaia and
Hipparcos satellites, and reproducing five, seven, and nine (or higher) parameter fits to their astrometry.

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

While in the root directory of this repo. It can also be installed with

.. code-block:: bash

    pip install git+https://github.com/gmbrandt/htof

Usage: Fits without Parallax
----------------------------
The following examples show how one would both load in and fit a line to the astrometric intermediate data
from either Hipparcos data reduction or Gaia (Currently only data release 2, GaiaDR2).

Let ra_vs_epoch, dec_vs_epoch be 1d arrays of ra and dec positions.
Assume we want to fit to data from GaiaDR2 on the star with hip id 027321. The choices of data
are :code:`GaiaDR2`, :code:`Hip1` and :code:`Hip2`. The following lines parse the intermediate data and fit a line.

.. code-block:: python

    from htof.main import Astrometry
    astro = Astrometry('GaiaDR2', star_id='027321', 'path/to/intermediate_data/', format='jd')  # parse
    ra0, dec0, mu_ra, mu_dec = astro.fit(ra_vs_epoch, dec_vs_epoch)

ra_vs_epoch and dec_vs_epoch are the positions in right ascension and declination of the object.
These arrays must have the same shape as astro.data.julian_day_epoch(),
which are the epochs in the intermediate data. :code:`format='jd'` specifies
the time units of the output best fit parameters. The possible choices of format
are the same as the choices for format in astropy.time.Time(val, format=format).
E.g. :code:`'decimalyear'`, :code:`'jd'` . If :code:`format='decimalyear'`, then the output :code:`mu_ra`
would have units of mas/year. If :code:`jd` then the output is mas/day. Both Hipparcos and Gaia catalogs list parallaxes
in milli-arcseconds (mas), and so positional units are always in mas for HTOF.

For Hipparcos 2, the path to the intermediate data would point to :code:`IntermediateData/resrec/`.
Note that the intermediate data files must be in the same format as the test intermediate data files found in this
repository under :code:`htof/test/data_for_tests/`. The best fit parameters have units of mas and mas/day by default.
The best fit skypath for right ascension is then :code:`ra0 + mu_ra * epochs`.

By default, the fit is a four-parameter fit: it returns the parameters to the line of best
fit to the sky path ra_vs_epoch, dec_vs_epoch. If you want a 6 parameter or 8 parameter fit, specify
fit_degree = 2 or fit_degree = 3 respectively. E.g.

.. code-block:: python

    from htof.main import Astrometry
    astro = Astrometry('GaiaDR2', star_id='027321', 'path/to/intermediate_data/', format='jd')
    ra0, dec0, mu_ra, mu_dec, 1/2*acc_ra, 1/2*acc_dec = astro.fit(ra_vs_epoch, dec_vs_epoch, fit_degree=2)

where 1/2*acc_ra and 1/2*acc_dec are 1/2 times the acceleration in right ascension and declination, respectively.
This factor of 1/2 is because HTOF uses a power series as the basis for all fits. If fit_degree = 3,
then the last two parameters would be one-sixth the jerk in right ascension and declination, respectively.

HTOF allows fits of arbitrarily high degree. E.g. setting fit_degree=5 would give a 13 parameter
fit (if using parallax as well). HTOF normalizes all epochs and times
from -1 to 1, so the linear algebra performed by HTOF is all very numerically stable.
This normalization can be disabled via the boolean keyword ``normed``, but one should be weary of turning this off.

If you want to specify a central epoch, you can do so with:

.. code-block:: python

    from htof.main import Astrometry

    astro = Astrometry('GaiaDR2', star_id='027321', 'path/to/intermediate_data/', central_epoch_ra=2456892, central_epoch_dec=2456892, format='jd')
    ra0, dec0, mu_ra, mu_dec = astro.fit(ra_vs_epoch, dec_vs_epoch)

The format of the central epochs must be specified along with the central epochs. The best fit sky path in right ascension would then be
:code:`ra0 + mu_ra * (epochs - centra_epoch_ra)`. The central epoch matters for numerical stability *only* when
``normed=False`` is set upon instantiation of ``Astrometry``.

Specifying :code:`GaiaDR2` will clip any intermediate data to fall within the observation
dates which mark the period covered by data release 2. Use :code:`Gaia` if you want any
and all observations within the intermediate data.

One can access the BJD epochs with

.. code-block:: python

    astro.central_epoch_dec
    astro.central_epoch_ra

If you want the standard (1-sigma) errors on the parameters, set :code:`return_all=True` when fitting:

.. code-block:: python

    from htof.main import Astrometry

    astro = Astrometry('GaiaDR2', star_id='027321', 'path/to/intermediate_data/', central_epoch_ra=2456892, central_epoch_dec=2456892, format='jd')
    coeffs, errors, chisq = astro.fit(ra_vs_epoch, dec_vs_epoch, return_all=True)


chisq is the chi-squared of the fit (the sum of `(data - model)^2`). For Hip2, this chi-squared
should equal the chi-squared calculated from the intermediate data via

`errors` is an array the same shape as coeffs, where each entry is the 1-sigma error for the
parameter at the same location in the coeffs array. For Hip1 and Hip2, HTOF loads in the real
catalog errors and so these parameter error estimates should match those given in the catalog. However,
for Gaia we do not have the error estimates from the GOST tool and so the best-fit parameter errors to
Gaia will not match those reported by the Gaia members.


.. code-block:: python

    chisq = chisq = np.sum(astro.data.residuals ** 2 / astro.data.along_scan_errs ** 2)


Usage: Fits with Parallax
-------------------------
TODO: Discuss how to get parallax, how to generate the plx pertubations. Why you have to specify
a central ra and dec, and that those must be instances of astropy.coordinates.Angle.

Appendix
--------
The Astrometry object is essentially just a wrapper for data parsing and fitting all in one, and consequently
could be limiting. This section describes how to reproduce Astrometry.fit by accessing the data parser objects and
the fitter object separately. You would do this if, for instance, you did not want to use
the built-in parallax motions generated by HTOF. I show here how to reproduce a five-parameter fit.


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

To fit a line with parallax, we first have to generate the parallactic motion about the central ra and dec. We do this
with the following code.

License
-------

MIT License. See the LICENSE file for more information.