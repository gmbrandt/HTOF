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

0.1.1 (2019-08-15)
------------------
- Bug fixes

0.1.0 (prior to 2019-08-15)
---------------------------
- Initial release.