0.2.10 (2019-03-05)
------------------
- The standard errors on fit parameters for Hipparcos 1 and the re-reduction are now correct.  
There was an erroneous factor of the square root of two in both cases. 
- The D. Michalik et al. 2014 error inflation factor (appendix B) is now applied to the Hipparcos 2
intermediate data along-scan errors, which brings the standard errors on the best-fit parameters
into agreement with the values on **the CD** (note the CD catalog values disagree slightly
with those on Vizier)

0.2.9 (2019-02-05)
------------------
- Removed the half day correction when converting from decimal year to julian date.

0.2.8 (2019-01-24)
------------------
- Instances of IntermediateDataParser can now be added to each other with the 
standard python addition operator. Each data attribute of the (new) class instance created by summing
will be the concatenation of the data attributes from the input classes.

0.2.7 (2019-01-24)
------------------
- Any class which inherits from IntermediateDataParser now has the .write() method which
converts the data stored in the attributes of IntermediateDataParser into an astropy.table.Table
and writes it out to the specified path. One can call IntermediateDataParser.write() with any of
the keyword arguments of astropy.table.Table.

0.2.6 (2019-01-24)
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
