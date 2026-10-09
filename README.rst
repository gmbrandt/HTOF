htof
===============

This repo contains htof, the package for parsing intermediate data from the Gaia and
Hipparcos satellites, and reproducing five, seven, and nine (or higher) parameter fits to their astrometry.

This build runs on Python 3.10-3.13 and NumPy 2.x (verified against NumPy 2.5,
SciPy 1.18, Astropy 8.0 and pandas 3.0). See ``CHANGES.md`` for what changed;
no fitting result was altered.

This repository is a fork of https://github.com/gmbrandt/HTOF at commit a75990e9, maintained at https://github.com/Realaqe/HTOF.

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

For a development install with the test dependencies:

.. code-block:: bash

    pip install -e ".[test]"

Verifying
---------

From the root directory of the repo:

.. code-block:: bash

    pytest -sv

All 140 tests should pass in about ten seconds. If they do not, something is
wrong with the install.

Usage
-----

HTOF has a rich variety of usages. We encourage the reader to consult the examples.ipynb jupyter notebook
for a set of usage examples (e.g., fitting the standard astrometric model to data, combining astrometric missions).
However, we also go into a few basic and specific use cases in this readme. Also see
examples_recalibrating_hip2 and GenerateSyntheticGaiaAstrometry for more uses of htof.

If you use HTOF, please cite the zenodo reference (https://doi.org/10.5281/zenodo.4104383) and
the source paper (https://ui.adsabs.harvard.edu/abs/2021AJ....162..230B/abstract)

Usage: Fits without Parallax
----------------------------
The following examples show how one would both load in and fit a line to the intermediate astrometric data (IAD)
from either Hipparcos data reduction or Gaia.

HTOF will download any missing IAD files for you (for Hip1, Hip2 (Java tool data), and Gaia). You should
provide a valid directory though for htof to save the file into for future use. Currently, for the
automatic download to work, you must provide a hipparcos name for the source (e.g., 27321).
Include ONLY the numeric part of the name. For an example of downloading Gaia data, see Example 6 in examples.ipynb

**For Gaia**:

If the automatic download of the GOST scanning law does not work, or the source does not have a
hipparcos ID. then you will have to download the GOST file manually with the user interface. Download a .csv of the
predicted scans and scan epochs from GOST (https://gaia.esac.esa.int/gost/). In particular, using the 'submit for
events forecast' feature on the website. One should select the widest range of dates
possible because htof automatically restricts the predicted epochs of observations
to the desired data release range (e.g., EDR3) and removes any astrometric gaps.

Let ra_vs_epoch, dec_vs_epoch be 1d arrays of ra and dec positions. Assume we want to fit to data from GaiaDR2 on the
star with hip id 027321. The following lines parse the intermediate data and fit a line.

.. code-block:: python

    from htof.main import Astrometry
    import numpy as np
    astro = Astrometry('GaiaDR2', '027321', 'htof/test/data_for_tests/GaiaDR2/IntermediateData', format='jyear')  # parse
    # note that if you do not have a GOST csv file with 027321 in the name, inside of
    # 'htof/test/data_for_tests/GaiaDR2/IntermediateData' , then htof will download it for you automatically!
    ra_vs_epoch = dec_vs_epoch = np.zeros(len(astro.data), dtype=float) # dummy set of ra and dec to fit.
    ra0, dec0, mu_ra, mu_dec = astro.fit(ra_vs_epoch, dec_vs_epoch)


:code:`GaiaeDR3` will select all data corresponding to the eDR3 data interval and exclude
eDR3 deadtimes. :code:`GaiaDR2` will select all data corresponding to the DR2 data interval (excluding dead times).
Finally, :code:`Gaia` will select all the data present in the GOST predicted observation file.

Note, selecting :code:`Gaia` and allowing HTOF to download the GOST scanning law may result in a wierd coverage of the
scanning law. We recommend sticking with defined data release intervals, so :code:`GaiaeDR3`, etc.

In the :code:`astro.fit` part, ra_vs_epoch and dec_vs_epoch are the positions in right ascension and declination of the object.
These arrays must have the same shape as astro.data.julian_day_epoch(),
which are the epochs in the intermediate data. :code:`format='jyear'` specifies
the time units of the output best fit parameters. The possible choices of format
are the same as the choices for format in astropy.time.Time(val, format=format). If :code:`format='jyear'`, then the output :code:`mu_ra`
would have units of mas/year. If :code:`jd` then the output is mas/day. Both Hipparcos and Gaia catalogs list parallaxes
in milli-arcseconds (mas), and so positional units are always in mas for HTOF.

**For Hipparcos** :

:code:`Hip2` refers to the DVD IAD which is now obsolete. :code:`Hip21` refers to the
Java Tool Intermediate Astrometric Data (IAD) and best fit parameters. This is the preferred set of
data to use with the 2007 re-reduction (preferred over the DVD IAD). The Hipparcos Java Tool data parser is meant for
the 2014 Java tool data (Java tool first released at
https://www.cosmos.esa.int/web/hipparcos/java-tools/intermediate-data, in 2014). As of 2021, there has not been an
update to the Java tool.

Remember, Hip21 will be automatically downloaded
per source. But if you want to avoid downloading the data on the fly, then the full Java Tool Intermediate Astrometric Data can be downloaded from
https://www.cosmos.esa.int/web/hipparcos/hipparcos-2 and extracted (ignore the _MACOSX folder if there is one).
One would then point any HTOF parser to the ResRec_JavaTool folder that contains the H00 etc. subfolders of the individual IAD files. So:

.. code-block:: python

    from htof.main import Astrometry
    astro = Astrometry('Hip21', star_id='027321', '/home/user/Downloads/ResRec_JavaTool_2014/ResRec_JavaTool_2014', format='jyear')  # parse
    ra0, dec0, mu_ra, mu_dec = astro.fit(ra_vs_epoch, dec_vs_epoch)



We discuss enabling fits with parallax later. By default, the fit is a four-parameter fit: it returns the parameters to the line of best
fit to the sky path ra_vs_epoch, dec_vs_epoch. If you want a 6 parameter or 8 parameter fit, specify
fit_degree = 2 or fit_degree = 3 respectively. E.g.

.. code-block:: python

    from htof.main import Astrometry
    astro = Astrometry('GaiaDR2', '027321', 'htof/test/data_for_tests/GaiaDR2/IntermediateData', format='jyear',
                       fit_degree=2)
    ra0, dec0, mu_ra, mu_dec, acc_ra, acc_dec = astro.fit(ra_vs_epoch, dec_vs_epoch)

If fit_degree = 3, then the additional last two parameters would be the jerk in right ascension and declination, respectively.
The sky path in RA (for instance) should be reconstructed by `ra0 + mu_ra*t + 1/2*acc_ra*t**2` where `t` are the epochs
from `astro.fitter.epoch_times` minus the central epoch for RA (if provided).

HTOF allows fits of arbitrarily high degree. E.g. setting fit_degree=3 would give a 9 parameter
fit (if using parallax as well). One should specify a central epoch for the fit, typically choosing the central epoch
from the catalog (e.g. 2015.5 for GaiaDR2, 2016 for GaiaEDR3, 1991.25 for Hipparcos). You can specify the central epoch by:

.. code-block:: python

    from htof.main import Astrometry

    astro = Astrometry('GaiaDR2', '027321', 'htof/test/data_for_tests/GaiaDR2/IntermediateData',
                       central_epoch_ra=2015.5, central_epoch_dec=2015.5, format='jyear')
    ra0, dec0, mu_ra, mu_dec = astro.fit(ra_vs_epoch, dec_vs_epoch)

The format of the central epochs must be specified along with the central epochs. The best fit sky path in right ascension would then be
:code:`ra0 + mu_ra * (epochs - centra_epoch_ra)`. The central epoch matters for numerical stability and covariances.
E.g., dont choose a central epoch like the year 1200 for GaiaDR2. One should almost always choose the central epoch
from the catalog.

One can access the BJD epochs with

.. code-block:: python

    astro.data.julian_day_epoch()

If you want the standard (1-sigma) errors on the parameters, set :code:`return_all=True` when fitting:

.. code-block:: python

    from htof.main import Astrometry

    astro = Astrometry('GaiaDR2', '027321', 'htof/test/data_for_tests/GaiaDR2/IntermediateData',
                        central_epoch_ra=2015.5, central_epoch_dec=2015.5, format='jyear')
    solution_vector, errors, chisq, residuals = astro.fit(ra_vs_epoch, dec_vs_epoch, return_all=True)


`errors` is an array the same shape as solution_vector, where each entry is the 1-sigma error for the
parameter at the same location in the solution_vector array. `chisq` is the formal chisquared of the
fit to the data, and `residuals` are the data - model residuals (given as a Nx2 shaped array, where N is the number
of transits. The first column are the ra residuals and the second are the declination residuals).
For a simple refit to the catalog IAD, `residuals` (converted to the AL basis) will equal (up to round off)
the residuals given in the IAD.  One could convert the residuals to the along scan basis by doing:

.. code-block:: python

    from htof.special_parse import to_along_scan_basis
    residuals = to_along_scan_basis(ra_decresiduals[:, 0], ra_decresiduals[:, 1], astro.data.scan_angle.values)
    # now residuals will be a one dimensional array of length N (number of transits), giving the residuals along the
    # scan.

For Hip1 and Hip21, HTOF loads in the real
catalog errors and so the parameter error estimates (`errors`) should match those given in the catalog. For Hip2, the
along scan errors are automatically inflated or deflated in accordance with D. Michalik et al. 2014.
For Gaia we do not have the error estimates from the GOST tool. The AL errors are set to 1 mas by default and so the
best-fit parameter errors to Gaia will not match those reported by the catalog.


`chisq` is the chi-squared of the fit (the sum of `(data - model)^2/error^2`). The `chisq` from `astro.fit`
should equal (for Hip1 and Hip2) the chi-squared calculated from the intermediate data:

.. code-block:: python

    chisq = np.sum(astro.data.residuals ** 2 / astro.data.along_scan_errs ** 2)

Saving processed intermediate data
----------------------------------
To save the scan angles, residuals, along-scan errors, inverse covariance matrices, and julian day
epochs, one can call ``Astrometry.data.write(path)`` to write out the data, where path is a string which
points to the full filepath including the data extension. We recommend ``.csv``, however any file extension
supported by ``astropy.table.Table.write()`` is supported. As well, one can call ``Astrometry.data.write(path)``
with any of the kwargs or args of ``astropy.table.Table.write()``.

Usage: Fits with Parallax
-------------------------
To fit an object with parallax, there are two ways. Both are equivalent.

    1. Let htof compute the parallax factors anew.
    2. Pull the parallax factors from the IAD or the scanning law.

Option 1: we need to provide a `central_ra` and `central_dec` to the `Astrometry` class. These positions
will be used to calculate the parallax components of the fit (the parallax factors). Using beta pic as an example,
we would do:


.. code-block:: python

    from htof.main import Astrometry
    import numpy as np
    from astropy.coordinates import Angle
    # central ra and dec from the Hip1 catalog
    cntr_ra, cntr_dec = Angle(86.82118054, 'degree'), Angle(-51.06671341, 'degree')
    # generate fitter and parse intermediate data
    astro = Astrometry('Hip1', '27321', 'htof/test/data_for_tests/Hip1/IntermediateData', central_epoch_ra=1991.25,
                       central_epoch_dec=1991.25, format='jyear', fit_degree=1, use_parallax=True,
                       central_ra=cntr_ra, central_dec=cntr_dec)
    ra_vs_epoch = dec_vs_epoch = np.zeros(len(astro.data), dtype=float) # dummy set of ra and dec to fit.
    solution_vector, errors, chisq, residuals = astro.fit(ra_vs_epoch, dec_vs_epoch, return_all=True)
    parallax, ra0, dec0, mu_ra, mu_dec = solution_vector


Option 2: In most use cases, this option is perfectly fine. And it is simpler. The object's parallax factors is available with the IAD (or the scanning law in the case of Gaia)
So you do not need a `central_ra` and `central_dec` to the `Astrometry` class. In which case, you can do:

.. code-block:: python

    from htof.main import Astrometry
    import numpy as np
    # generate fitter and parse intermediate data
    astro = Astrometry('Hip1', '27321', 'htof/test/data_for_tests/Hip1/IntermediateData', central_epoch_ra=1991.25,
                       central_epoch_dec=1991.25, format='jyear', fit_degree=1, use_parallax=True,
                       use_catalog_parallax_factors=True)
    ra_vs_epoch = dec_vs_epoch = np.zeros(len(astro.data), dtype=float) # dummy set of ra and dec to fit.
    solution_vector, errors, chisq, residuals = astro.fit(ra_vs_epoch, dec_vs_epoch, return_all=True)
    parallax, ra0, dec0, mu_ra, mu_dec = solution_vector

Note that we have set ``use_catalog_parallax_factors=True``. This tells htof to *not* compute parallax factors
anew, and instead to pull them from the IAD. If data choice was 'Gaiaedr3' instead of 'hip1', then the parallax factors
would come from the GOST csv file.

Appendix
--------

Parsing and fitting manually
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The Astrometry object is essentially just a wrapper for data parsing and fitting all in one, and consequently
could be limiting. This section describes how to reproduce Astrometry.fit by accessing the data parser objects and
the fitter object separately. You would do this if, for instance, you did not want to use
the built-in parallax motions generated by HTOF. Or if, you wanted to do a GaiaEDR3 fit with your own AL errors.
I show here how to reproduce a five-parameter fit.


.. code-block:: python

    from htof.parse import HipparcosOriginalData # or GaiaData or HipparcosReReduction
    data = HipparcosOriginalData()
    data.parse(star_id='004391', intermediate_data_directory='htof/test/data_for_tests/Hip1/IntermediateData/')
    data.calculate_inverse_covariance_matrices()

data now has a variety of intermediate data products such as the scan angles, the epochs when each
data point was collected, the inverse covariance matrices describing the errors of the scan,
and the BJD epochs accessible through :code:`data.julian_day_epoch()`.

You could modify the along-scan errors (let's say if you were doing a Gaia DR4/DR5 forecast) with:

.. code-block:: python

    from htof.parse import GaiaData
    import pandas as pd
    import numpy as np
    data = GaiaData() # GaiaData will load every scan you have in the .csv GOST file
    data.parse(star_id='27321', intermediate_data_directory='htof/test/data_for_tests/GaiaeDR3/IntermediateData')
    data.along_scan_errs = pd.Series(np.ones(len(data), dtype=float) * 0.22) # set every along scan error to 220 micro arc seconds.
    data.calculate_inverse_covariance_matrices()

Then we could go on and do the fit (detailed shortly after this) and we would have an estimate for the
parameter errors for a fictional Gaia mission that contained all the available scans on GOST (e.g., 10 years) with a
0.22 mas along scan error for each scan.

If you have two astrometric missions, say GaiaDR2 and HipparcosOriginalData, you can concatenate
their processed intermediate data by summing the two class instances as follows:

.. code-block:: python

    from htof.parse import HipparcosOriginalData, GaiaDR2
    hip = HipparcosOriginalData()
    hip.parse(star_id='027321', intermediate_data_directory='htof/test/data_for_tests/Hip1/IntermediateData/')
    hip.calculate_inverse_covariance_matrices()
    gaia = GaiaDR2()
    gaia.parse(star_id='027321', intermediate_data_directory='htof/test/data_for_tests/GaiaDR2/IntermediateData/')
    gaia.calculate_inverse_covariance_matrices()

    data = hip + gaia

There is a frame rotation between Gaia and Hipparcos that htof does not include, so the results of combining the two
missions and performing a fit to them should not be interpreted without serious care. One would have to account for frame rotation
in the intermediate data first.

Now to find the best fit astrometric parameters. Given a parsed data object, we simply call:

.. code-block:: python

    from htof.fit import AstrometricFitter
    from astropy.time import Time
    fitter = AstrometricFitter(inverse_covariance_matrices=data.inverse_covariance_matrix,
                               epoch_times=Time(data.julian_day_epoch(), format='jd').jyear,
                               central_epoch_dec=2016, # 2016, assuming we are working with gaia edr3 here.
                               central_epoch_ra=2016,
                               fit_degree=1,)
    ra_vs_epoch = dec_vs_epoch = np.zeros(len(data), dtype=float)  # dummy values of zero.
    solution_vector, errors, chisq, residuals = fitter.fit_line(ra_vs_epoch, dec_vs_epoch, return_all=True)
    ra0, dec0, mu_ra, mu_dec = solution_vector

where :code:`ra(jyear) = ra0 + mu_ra * (jyear - 2016)`, and same for declination.

To fit a line with parallax, we first have to generate the parallactic motion about the central ra and dec
(i.e., the parallax factors). We do this with the following code.

.. code-block:: python

    from htof.sky_path import earth_ephemeris, parallactic_motion
    from astropy.coordinates import Angle
    # define central_ra, central_dec as astropy.coordinates.Angle objects.
    cntr_ra, cntr_dec = Angle(86.82118054, 'degree'), Angle(-51.06671341, 'degree')
    ra_motion, dec_motion = parallactic_motion(Time(data.julian_day_epoch(), format='jd').jyear,
                                           cntr_ra.mas, cntr_dec.mas, 'mas',
                                           1991.25,
                                           ephemeris=earth_ephemeris) # earth ephemeris for hipparcos.
    parallactic_pertubations = {'ra_plx': ra_motion, 'dec_plx': dec_motion}


Now that we have the parallax factors of the fit, we can provide these to the `AstrometricFitter` object to
produce a fit which includes parallax. We now do:

.. code-block:: python

    fitter = AstrometricFitter(inverse_covariance_matrices=data.inverse_covariance_matrix,
                               epoch_times=Time(data.julian_day_epoch(), format='jd').jyear,
                               use_parallax=True,
                               parallactic_pertubations=parallactic_pertubations,
                               central_epoch_ra=1991.25, central_epoch_dec=1991.25)
    solution_vector = fitter.fit_line(ra_vs_epoch, dec_vs_epoch)
    parallax, ra0, dec0, mu_ra, mu_dec = solution_vector


For more examples, refer to the `examples.ipynb` Jupyter notebook. There we will make a figure like Figure 3 from the HTOF paper.

Flagged Sources
~~~~~~~~~~~~~~~
There are a number of sources in the DVD re-reduction that HTOF cannot well refit. These sources should be used cautiously
and are listed by HIP ID in the files in the htof/data directory:
htof/data/hip2_dvd_flagged.txt for the 2007 re-reduction which came on the DVD accompanying the book.

HTOF can refit well most Hip1 sources and nearly every source from the Hipparcos re-reduction
*but only if using the IAD from the Java tool*, which was recently posted online here: https://www.cosmos.esa.int/web/hipparcos/hipparcos-2
One should update to use the java tool IAD for the hipparcos re-reduction. The few sources that
htof cannot handle well are listed in htof/data/hip2_Javatool_flagged.txt and htof/data/hip1_flagged.txt for
the java tool Hip re-reduction IAD and the original reduction IAD, respectively.


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