"""
NOTE 1: Gaia DR2 is based on data collected from the start of the nominal observations on 2014
 July 25 (10:30 UTC) until 2016 May 23 (11:35 UTC), or 668 days. However, the astrometric solution
 for this release did not use the observations during the first month after commissioning, when
 a special scanning mode (the ecliptic pole scanning law, EPSL) was employed. The data for the
 astrometry therefore start on 2014 Aug 22 21:00 UTC (BJD 2456892.375) and cover 640 days
 or 1.75 yr (therefore until 2016 May 23, or BJD 2457532.375).


NOTE 2: GaiaEDR3 is based on data collected from the start of the nom-inal
observations  on  25  July  2014  (10:30  UTC)  until  28  May2017 (08:45 UTC),
or 1038 days (data segments DS0–DS3 inFig. 1). Similarly to the astrometric solution for
DR2 (Lindegrenet al. 2018), this solution did not use the observations in the first month of
the operational phase, when the special ecliptic polescanning law (EPSL) was employed.
The data for the astrom-etry therefore start on 22 August 2014 (21:00 UTC) and cover
1009 days or 2.76 yr, with some interruptions mentioned below.

NOTE 2 is from section 2.2 of https://www.aanda.org/articles/aa/pdf/forth/aa39709-20.pdf
"""

GaiaDR2_min_epoch = 2456892.375  # Barycentric Julian Day (BJD)
GaiaDR2_max_epoch = 2457532.375  # Barycentric Julian Day (BJD)

GaiaeDR3_min_epoch = 2456892.375  # Barycentric Julian Day (BJD), 2014.6403 in jyear
GaiaeDR3_max_epoch = 2457901.375  # Barycentric Julian Day (BJD), 2017.4028 in jyear
