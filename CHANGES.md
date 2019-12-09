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
  follow the same convention as astropy.time.Time. E.g. `format='jd'` or `format='fracyear'`
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
