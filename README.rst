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
htof can be installed with PyPi and pip, by running

.. code-block:: bash

    pip install htof

or by running

.. code-block:: bash

    pip install .


while in the root directory of this repo. It can also be installed directly from github using

.. code-block:: bash

    pip install git+https://github.com/gmbrandt/htof

Usage: Fits without Parallax
----------------------------
The following examples show how one would both load in and fit a line to the astrometric intermediate data
from either Hipparcos data reduction or Gaia. Gaia requires you to first download a .csv of the
predicted scans and scan epochs from GOST (https://gaia.esac.esa.int/gost/).

Let ra_vs_epoch, dec_vs_epoch be 1d arrays of ra and dec positions.
Assume we want to fit to data from GaiaDR2 on the star with hip id 027321. The choices of data
are :code:`GaiaeDR3`, :code:`GaiaDR2`, :code:`Gaia`, :code:`Hip1` and :code:`Hip2`.
The following lines parse the intermediate data and fit a line.

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

When using Gaia, one should download the largest stretch of GOST times possible (covering at least the eDR3
timespan, e.g. covering at least the dates BJD 2456892 to BJD 2457902).
:code:`GaiaeDR3` will select all data corresponding to the eDR3 data interval and exclude
eDR3 deadtimes. :code:`GaiaDR2` will select all data corresponding to the DR2 data interval (excluding dead times).
Finally, :code:`Gaia` will select all the data present in the GOST predicted observation file that you have
downloaded.

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
    ra0, dec0, mu_ra, mu_dec, acc_ra, acc_dec = astro.fit(ra_vs_epoch, dec_vs_epoch, fit_degree=2)

If fit_degree = 3, then the additional last two parameters would be the jerk in right ascension and declination, respectively.
The sky path in RA (for instance) should be reconstructed by `ra0 + mu_ra*t + 1/2*acc_ra*t**2` where `t` are the epochs
from `astro.fitter.epoch_times` minus the central epoch for RA (if provided).

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


`errors` is an array the same shape as coeffs, where each entry is the 1-sigma error for the
parameter at the same location in the coeffs array. For Hip1 and Hip2, HTOF loads in the real
catalog errors and so these parameter error estimates should match those given in the catalog. For Hip2, the
along scan errors are automatically inflated or deflated in accordance with D. Michalik et al. 2014.
For Gaia we do not have the error estimates from the GOST tool and so the best-fit parameter errors to
Gaia may not match those reported by the catalog.


`chisq` is the chi-squared of the fit (the sum of `(data - model)^2/error^2`). The `chisq` from `astro.fit`
should equal (for Hip1 and Hip2) the chi-squared calculated from the intermediate data:

.. code-block:: python

    chisq = chisq = np.sum(astro.data.residuals ** 2 / astro.data.along_scan_errs ** 2)

Saving processed intermediate data
----------------------------------
To save the scan angles, residuals, along-scan errors, inverse covariance matrices, and julian day
epochs, one can call ``Astrometry.data.write(path)`` to write out the data, where path is a string which
points to the full filepath including the data extension. We recommend ``.csv``, however any file extension
supported by ``astropy.table.Table.write()`` is supported. As well, one can call ``Astrometry.data.write(path)``
with any of the kwargs or args of ``astropy.table.Table.write()``.

Usage: Fits with Parallax
-------------------------
To fit an object with parallax, we need to provide a `central_ra` and `central_dec` to the `Astrometry` class. These positions
will be used to calculate the parallax components of the fit. Using beta pic as an example, we would do:


.. code-block:: python

    from htof.main import Astrometry
    # central ra and dec from the Hip1 catalog
    cntr_ra, cntr_dec = Angle(86.82118054, 'degree'), Angle(-51.06671341, 'degree')
    # generate fitter and parse intermediate data
    astro = Astrometry('Hip1', '27321', 'path/to/intermediate_data/', central_epoch_ra=1991.25,
                       central_epoch_dec=1991.25, format='jyear', fit_degree=1, use_parallax=True,
                       central_ra=cntr_ra, central_dec=cntr_dec)
    coeffs, errors, chisq = astro.fit(ra_vs_epoch, dec_vs_epoch, return_all=True)
    parallax, ra0, dec0, mu_ra, mu_dec = coeffs


Appendix
--------

Parsing and fitting manually
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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


If you have two astrometric missions, say Gaia and HipparcosOriginalData, you can concatenate
their processed intermediate data by summing the two class instances as follows:

.. code-block:: python

    from htof.parse import HipparcosOriginalData # or GaiaData or HipparcosReReduction
    hip = HipparcosOriginalData()
    hip.parse(star_id='049699', intermediate_data_directory='Hip1/IntermediateData/')
    hip.calculate_inverse_covariance_matrices()
    gaia = GaiaDR2()
    gaia.parse(star_id='049699', intermediate_data_directory='GaiaDR2/IntermediateData/')
    gaia.calculate_inverse_covariance_matrices()

    data = hip + gaia

There is a frame rotation between Gaia and Hipparcos, so the results of combining the two
missions and performing a fit to them should not be interpreted without serious care. One would have to account for frame rotation
in the intermediate data first.

Now to find the best fit astrometric parameters. Given a parsed data object, we simply call:

.. code-block:: python

    fitter = AstrometricFitter(inverse_covariance_matrices=data.inverse_covariance_matrix, epoch_times=data.julian_day_epoch())
    solution_vector = fitter.fit_line(ra_vs_epoch, dec_vs_epoch)
    ra0, dec0, mu_ra, mu_dec = solution_vector

where :code:`ra(mjd) = ra0 + mu_ra * mjd`, and same for declination.

To fit a line with parallax, we first have to generate the parallactic motion about the central ra and dec. We do this
with the following code.

.. code-block:: python

    from htof.sky_path import earth_ephemeris
    ra_motion, dec_motion = parallactic_motion(Time(hip.julian_day_epoch(), format='jd').jyear,
                                           central_ra.mas, central_dec.mas, 'mas',
                                           1991.25,
                                           ephemeris=self.ephemeri[data_choice.lower()])
    parallactic_pertubations = {'ra_plx': ra_motion, 'dec_plx': dec_motion}


Now that we have the parallax components of the fit, we can provide these to the `AstrometricFitter` object to
produce a fit which includes parallax. We now do:

.. code-block:: python

    fitter = AstrometricFitter(inverse_covariance_matrices=hip.inverse_covariance_matrix,
                               epoch_times=Time(hip.julian_day_epoch(), format='jd').jyear,
                               use_parallax=True,
                               parallactic_pertubations=parallactic_pertubations,
                               central_epoch_ra=1991.25, central_epoch_dec=1991.25)
    solution_vector = fitter.fit_line(ra_vs_epoch, dec_vs_epoch)
    parallax, ra0, dec0, mu_ra, mu_dec = solution_vector


For more examples, refer to the `examples.ipynb` Jupyter notebook. There we will make a figure like Figure 5 from the HTOF paper.

Flagged Sources
~~~~~~~~~~~~~~~
There are a small number of sources in the original reduction of Hipparcos and the re-reductions (e.g. the DVD version
of the 2007 re-reduction) that HTOF cannot well refit. These sources should be used cautiously and are listed by HIP ID in
the files in the htof/data directory, e.g. htof/data/hip1_flagged.txt for the 1997 reduction and
htof/data/hip2_dvd_flagged.txt for the 2007 re-reduction which came on the DVD accompanying the book. Every source in
these lists have a difference in the catalog best fit proper motions and the HTOF refit proper motions in excess
of 0.02 mas/yr in either RA or DEC or both.


Astrometric Gaps
~~~~~~~~~~~~~~~~
Not all of the planned observations will be used in the astrometric solution.
Some predicted scans will represent missed observations (satellite dead times),
executed but unusable observations (e.g.~from cool-down after decontamination),
or observations rejected as astrometric outliers.  Rejected observations could
be corrupted due to, e.g.~micro-clanks, scattered light from a nearby bright
source, crowded fields, micro-meteoroid hits,
etc.~(See https://www.cosmos.esa.int/web/gaia/dr2-data-gaps).
Such problematic observations do not constrain the DR2 astrometric solution.
The largest stretches of dead times and rejected observations are
published as astrometric gaps; 239 are listed at the time of this
publication for DR2 (available here https://www.cosmos.esa.int/web/gaia/dr2-data-gaps).
We fetched the DR2 dead times on 2020/08/25. htof accounts for these astrometric gaps in DR2.

The eDR3 dead times were fetched from https://www.aanda.org/articles/aa/pdf/forth/aa39709-20.pdf on
2020/12/23. htof accounts for these astrometric gaps in eDR3.


License
-------

MIT License. See the LICENSE file for more information.