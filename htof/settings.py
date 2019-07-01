"""
NOTE: Gaia DR2 is based on data collected from the start of the nominal observations on 2014
 July 25 (10:30 UTC) until 2016 May 23 (11:35 UTC), or 668 days. However, the astrometric solution
 for this release did not use the observations during the first month after commissioning, when
 a special scanning mode (the ecliptic pole scanning law, EPSL) was employed. The data for the
 astrometry therefore start on 2014 Aug 22 21:00 UTC (BJD 2456892.375) and cover 640 days
 or 1.75 yr (therefore until 2016 May 23, or BJD 2457532.375).
"""

GaiaDR2_min_epoch = 2456892.375  # Barycentric Julian Day (BJD)
GaiaDR2_max_epoch = 2457532.375  # Barycentric Julian Day (BJD)
