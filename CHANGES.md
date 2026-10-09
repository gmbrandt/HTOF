1.2.0 (2026-08-28)
------------------
Modernisation release. No change to any fitting result; this is about running on
current Python and current NumPy.

- Supports Python 3.10-3.13. `python_requires` is now `>= 3.10`.
- **Fixes an import failure on Python 3.12+.** `htof.parse` and
  `htof.special_parse` imported `pkg_resources`, which setuptools no longer
  installs by default, so `import htof.parse` raised `ModuleNotFoundError`.
  Package data files are now located with `importlib.resources`, via the new
  `htof.utils.resources.resource_filename`.
- Packaging moved from `setup.py` to `pyproject.toml` (setuptools backend). The
  `pytest-runner` `setup_requires` hook, deprecated for years, is gone; test and
  coverage configuration now live in `pyproject.toml` instead of `pytest.ini`
  and `.coveragerc`.
- `htof.__version__` is now defined, read from the installed distribution
  metadata.
- `htof.validation.htof_adhoc_correction_f2` no longer runs an analysis at
  import time (which made the module unimportable without the full Hip21 IAD on
  disk, and tried to hit the ESA FTP server). Its body is now
  `compute_f2_comparison()` plus a `main()` with `--iad-directory` / `--output`,
  under `if __name__ == '__main__'`.
- Every bare `except:` is now `except Exception:`, so a `KeyboardInterrupt`
  during a long parse is no longer swallowed and turned into a misleading
  "connection failed" error.
- Fixed a `SyntaxWarning` from an unescaped `\d` in a `htof.fit` docstring.
- Verified against NumPy 2.5, SciPy 1.18, Astropy 8.0 and pandas 3.0: all 140
  tests pass.
- Verified on 2026-09-11: 140 tests pass on Python 3.12.14 with NumPy 2.5.3.
  Installs into a fresh Python 3.13 environment (2026-10-05).

1.1.5 (2022-06-24)
------------------
- Added the ability to download the Hipparcos re-reduction data (Java Tool Version). 
- Slight refactor of the internals of downloading Gaia/Hip1 files from the web.
- Crucial bug fix in get_intermediate_data_filename that sometimes would give the wrong IAD file
for small Hip ids.
- ONLY numeric star_ids are allowed in Parsers. E.g., 2732 is allowed and not H88iJJ2.

1.1.4 (2022-06-03)
------------------
- Misc bug fixes for data downloading, and warning fixes.

1.1.3 (2022-05-25)
------------------
- HipparcosOriginalData Parser classes can now automatically download the Hip 1997 IAD.

1.1.2 (2022-05-23)
------------------
- Updated the recalibration parameters.

1.1.1 (2022-05-10)
------------------
- Gaia Parser classes now can fetch GOST scanning law from the GOST API. 
- Downloaded GOST scanning laws are saved to the directory provided.
- having the examples in an examples/ folder broke the filepaths.

1.1.0 (2022-04-19)
------------------
- Added a new hip2 parser, in special_parse.py, that will add the residual offset and cosmic dispersion of 
  Brandt et al. 2022 and output new IAD with new best-fit parameters. Works for 5, 7, and 9 parameter sources.
- Added a new hip2 parser that autodetects whether the data file is the java tool or DVD. Technically, it is 
  a parser factory.
- Added the option for htof.main.Astrometry to load in and use the parallax factors supplied with
the hipparcos IAD.
- htof.special_parse.Hipparcos2Recalibrated can now write out a data file in the same format 
  (although they do not match byte-by-byte exactly) as the hip2 java tool IAD.
- fitting with return_all=True now returns 4 arrays, the last is the residuals (in ra and dec) of the fit.
- New examples jupyter notebook for recalibrating the hip2 java tool IAD.
- Astrometry() now has an along_scan_error_scaling attribute that will scale the
  along scan errors by that factor. This is particularly useful for gaia forecasting. Setting
  along_scan_error_scaling = 0.1 for instance, will artifically make all the Gaia GOST 
  scans have an error of 0.1 mas (normally, these along scan errors are not provided by GOST)

1.0.3 (2022-01-26)
------------------
- updated the jupyter examples notebook. Included a central epoch comparison to the Gaia archive.

1.0.2 (2021-12-20)
------------------
- Added Python 3.9 to travis testing suite. As of writing, python 3.10 testing is
not available with travis. I manually tested that pytest-sv works with 
  python 3.10.1 on my local ubuntu 20.04 machine.

1.0.1 (2021-09-01)
------------------
- updated docs.
- Added gaiaEDR3 ephemeris, resolving a bug when using use_parallax=True and gaiaEDR3.

1.0.0 (2021-09-01)
------------------
- Updated the parsing method to read in the new, final version of the Hip2 Java tool data that
will be published on the ESA website.
- Changed the Hip2 (DVD) refit so that the parameter errors are saved, not the parameter differences.
- Changed back the error inflation computation to use the catalog f2 value.

0.4.2 (2021-06-24)
------------------
- Introduced a .meta attribute to data parsers. This holds things like the catalog f2 value.
- Updated examples.ipynb.

0.4.1 (2021-06-15)
------------------
- Improved warning messages for the adhoc correction.

0.4.0 (2021-04-30)
------------------
- Implemented a better data fix for Hip2 java data. This fixes most of the 6400 discrepant sources
that have the Hip2 file-write error. Note that this is not perfect, there are some minor
  degeneracies between which epochs to reject. See the note in parse.find_epochs_to_reject()
- Note that this write-out bug fix does not work on the dvd data (note, 
  the dvd still has the same bug, it is just not easily correctable).
- Updated the flagged source list (hip2_javatool_flagged.txt).
- Parallax factors from the IAD are now loaded on data.parse()
- Updated parse to use the new version of the java tool IAD. Old versions of the java
tool IAD will not work.

0.3.5 (2021-04-19)
------------------
- Fixed improper syntax calls to panda etc. so that warnings are silence in python 3.8 and beyond.
- We have a deprecated call that cannot be fixed, and appears to be
  located upstream in ipython https://github.com/ipython/ipykernel/issues/560
- Removed normed option for fitting.

0.3.4 (2021-02-24)
------------------
- Removed the numba speed up from FastFitter. Will add it back in if there is ever a need for a fast fitter.

0.3.3 (2020-12-23)
------------------
- Added Gaia eDR3 deadtimes and a GaiaeDR3 class for Gaia early data release3.
- Added a 'gaiaedr3' data choice to the main.Astrometry class.
- Added the method AstrometricFitter.find_optimal_central_epoch() which will find
the central epoch that minimizes the covariance between position and proper motion
for either right-ascension (if calling find_optimal_central_epoch('ra')) or declination 
(if calling find_optimal_central_epoch('dec')).
- Added a convienience function to Astrometry: Astrometry.optimal_central_epochs() that
returns a dictionary with the epochs in ra and dec that give minimum (close to 0, like 1e-10)
covariance between ra and mu_ra, and dec and mu_dec respectively.

0.3.2 (2020-10-22)
-------------------
- Added a 'BOTH' data_choice option to the HipparcosOriginalData DataParser class. Selecting
'BOTH' will keep both NDAC and FAST data, i.e. it will leave the iad unmerged.
- Fixed typo in the flagged hip2 source list filename.

0.3.1 (2020-10-22)
-------------------
- Fixed a typo in the path to the package data for the dead time table.

0.3.0 (2020-10-17)
-------------------
- HipparcosRereductionCDBook has been renamed HipparcosRereductionDVDBook because the data come from a DVD.
- Added various utilities and scripts for validating HTOF's fitting and parsing 
routines over the entire Hipparcos catalog.
- Added a list of flagged sources for which HTOF cannot reproduce their catalog solution.
- Added an AstrometricFastFitter that fits lines much faster than AstrometricFitter.
- Added an examples.ipynb jupyter notebook. 

0.2.11 (2020-08-25)
-------------------
- Implemented dead time rejection for Gaia. GaiaData does not reject dead times by default, GaiaDR2 rejects
dead times posted at https://www.cosmos.esa.int/web/gaia/dr2-data-gaps under Astrometric Gaps, 
fetched on August 25th 2020. That dead time table is in htof/htof/data/astrometric_gaps_gaiadr2_08252020.csv

0.2.10 (2020-03-05)
------------------
- The standard errors on fit parameters for Hipparcos 1 and the re-reduction are now correct.  
There was an erroneous factor of the square root of two in both cases. 
- The D. Michalik et al. 2014 error inflation factor (appendix B) is now applied to the Hipparcos 2
intermediate data along-scan errors, which brings the standard errors on the best-fit parameters
into agreement with the values on **the DVD** (note the DVD catalog values disagree slightly
with those on Vizier)

0.2.9 (2020-02-05)
------------------
- Removed the half day correction when converting from decimal year to julian date.

0.2.8 (2020-01-24)
------------------
- Instances of IntermediateDataParser can now be added to each other with the 
standard python addition operator. Each data attribute of the (new) class instance created by summing
will be the concatenation of the data attributes from the input classes.

0.2.7 (2020-01-24)
------------------
- Any class which inherits from IntermediateDataParser now has the .write() method which
converts the data stored in the attributes of IntermediateDataParser into an astropy.table.Table
and writes it out to the specified path. One can call IntermediateDataParser.write() with any of
the keyword arguments of astropy.table.Table.

0.2.6 (2020-01-24)
------------------
- The fit astrometric parameters mu_ra, mu_dec, acc_ra, acc_dec etc... now include n!
so that the astrometric motion (e.g. for RA) is ra_0 + mu_ra x t + 1/2 x acc_ra x t + ...

0.2.5 (2019-12-16)
------------------
- Merging of the intermediate data for the original hipparcos reduction is now done
with a mean weighted by the FAST/NDAC covariance matrix. Prior, only the residuals
and errors used these weights. Now all columns (IA3, IA4 etc...) use these weights.
- Merging is faster by about a factor of 300. It is now only 40% slower to parse
data_choice=`MERGED` as it is to parse either `NDAC` or `FAST` alone. Fitting time is independent
of data choice.

0.2.4 (2019-12-16)
------------------
- For the original Hipparcos reduction intermediate data, eight sources had zero entries in column IA3. 
The epoch is computed via IA6/IA3 or IA7/IA4. 
The former point led to undefined or infinite epoch values for those eight sources. 
This bug is now fixed by computing the epoch with IA7/IA4 where abs(IA4) > abs(IA3).

0.2.3 (2019-12-09)
------------------
- Users can now select normed=False in AstrometricFitter and Astrometry, if they wish to disable
the internal normalization which enhances numerical stability. Most users would want to leave
normed=True.

0.2.2 (2019-12-09)
------------------
- For Hipparcos 1, users can now select a data_choice of 'MERGED' which will
merge the two NDAC and FAST consortia and then fit that data. 'MERGED' is now the
default option in the Astrometry object.

0.2.1 (2019-12-06)
------------------
- Parallax motion is now stored inside the fitter as a dictionary with `ra_plx` and `dec_plx` keys 
  so that dec and ra motion cannot be mixed up.

0.2.0 (2019-10-25)
------------------
- Added support for fitting parallaxes, and arbitrarily
  high degree polynomial fits to astrometry.
- All fits have the domains normalized for numerical stability.
Changes to how the user interacts with HTOF:
- `central_epoch_fmt=` In htof.main.Astrometry is now `format=` and should 
  follow the same convention as astropy.time.Time. E.g. `format='jd'` or `format='decimalyear'`
  The returned proper motions from Astrometry.fit will have time units consistent
  with `format`. E.g. setting `format='fracyear'` would return proper motions with
  units of mas/yr (and accelerations with mas/year^2 etc..).
- The GaiaData parser now does not trim the input gaia data to the DR2 region. There is a new parser, GaiaDR2 which auto
  trims the data. Anywhere where users used GaiaData (assuming it to trim the scanning law to DR2) should be replaced with GaiaDR2.

0.1.1 (2019-08-15)
------------------
- Bug fixes

0.1.0 (prior to 2019-08-15)
---------------------------
- Initial release.
