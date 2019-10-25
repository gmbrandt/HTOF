"""
This module provides functions for calculating the astrometric paths of stars on the sky as seen by an
observer in orbit around the solar system barycentre. That is, the topocentric coordinate directions as a
function of time are calculated.

Author: Anthony Brown Nov 2018 - Nov 2018

From https://github.com/agabrown/astrometric-sky-path/  from commit: 039768eae04dca0b9b6615cccfc021e6a381bf4d
Reproduced with permission of the author.
"""

import numpy as np

from astropy import constants
from astropy import units
from astropy.time import Time
from astropy.coordinates import Angle
from astropy.coordinates import get_body_barycentric


def earth_ephemeris(t):
    """
    Calculate the ephemeris for the earth in the BCRS using astropy tools.
    
    NOTE: There are several versions of the solar system ephemeris available in astropy and others can be
    provided through a URL (see the documentation of astropy.coordinates.solar_system_ephemeris).
    Depending on the accuracy needed, employing  an ephemeris different from the default may be better.

    Parameters
    ----------
    
    t : array
        Barycentric Julian Year times at which to calculate the ephemeris.
        
    Returns
    -------

    Array of shape (3,t.size) representing the xyz components of the ephemeris at times t.

    Note: the units of angle should be radians and the units of time Julian years ('jyear'),
    while the distance unit is the AU.
    """
    times = Time(t, format='jyear', scale='tcb')  # 'tcb': Barycentric Coordinate Time (TCB)
    ephemeris = get_body_barycentric('earth', times)  # unit A.U.
    return np.vstack((ephemeris.x.value, ephemeris.y.value, ephemeris.z.value))


def earth_sun_l2_ephemeris(t):
    """
    Calculate the ephemeris for earth-sun L2 point in the BCRS using astropy tools.

    :param t: float array.
    Times at which to calculate the ephemeris in Julian years TCB.
    :return: float array.
    Array of shape (3,t.size) representing the xyz components of the ephemeris at times t.

    Note: 1.511 / 1.496 is the ratio of L2 semi-major axis to the earth's semi-major axis.
    """
    return earth_ephemeris(t) * 1.511 / 1.496


def epoch_topocentric_coordinates(alpha, delta, parallax, mura, mudec, vrad, t, refepoch, ephem):
    """
    For each observation epoch calculate the topocentric (as seen from a location on the earth's surface)
    coordinate directions (alpha(t), delta(t)) given the astrometric parameters of a source, the observation times,
    and the ephemeris (in the BCRS) for the observer. Also calculate the local plane coordinates xi(t) and eta(t).

    The code is partly based on the SOFA library (http://www.iausofa.org/) pmpx.c code.

    Parameters
    ----------
    
    alpha : float
        Right ascension at reference epoch (radians)
    delta : float
        Declination at reference epoch (radians)
    parallax : float
        Parallax (mas), negative values allowed
    mura : float
        Proper motion in right ascension, including cos(delta) factor (mas/yr)
    mudec : float
        Proper motion in declination (mas/yr)
    vrad : float
        Radial velocity (km/s)
    t : float array
        Observation times (Julian year TCB)
    refepoch : float
        Reference epoch (Julian year TCB)
    ephem : function
        Function providing the observer's ephemeris in BCRS at times t (units of AU)
                
    Returns (array, array, array, array)
    -------
    
    Arrays alpha, delta, xi, eta. Units are radians for (alpha, delta) and rad for (xi, eta).
    """
    # unit conversions
    _radtomas = (180 * 3600 * 1000) / np.pi
    _mastorad = np.pi / (180 * 3600 * 1000)
    _kmps_to_aupyr = (units.year.to(units.s) * units.km.to(units.m)) / constants.au.value

    # Normal triad, defined at the reference epoch.
    p = np.array([-np.sin(alpha), np.cos(alpha), 0.0])
    q = np.array([-np.sin(delta) * np.cos(alpha), -np.sin(delta) * np.sin(alpha), np.cos(delta)])
    r = np.array([np.cos(delta) * np.cos(alpha), np.cos(delta) * np.sin(alpha), np.sin(delta)])
   
    # Calculate observer's ephemeris.
    bO_bcrs = ephem(t)

    # Calculate the Roemer delay, take units into account.
    tB = t + np.dot(r, bO_bcrs) * constants.au.value / constants.c.value / units.year.to(units.s)
    
    plxrad = parallax*_mastorad
    murarad = mura*_mastorad
    mudecrad = mudec*_mastorad
    mur = vrad*_kmps_to_aupyr*np.abs(plxrad)

    uO = np.repeat(r, t.size).reshape((r.size, t.size))
    uO = uO + np.tensordot((p*murarad + q*mudecrad + r*mur), (tB-refepoch), axes=0) - plxrad*bO_bcrs
    
    # Local plane coordinates which approximately equal delta_alpha*cos(delta) and delta_delta
    xi = np.dot(p, uO)/np.dot(r, uO)
    eta = np.dot(q, uO)/np.dot(r, uO)

    alpha_obs = np.arctan2(uO[1, :], uO[0, :])
    delta_obs = np.arctan2(uO[2, :], np.sqrt(uO[0, :]**2+uO[1, :]**2))
                 
    return alpha_obs, delta_obs, xi, eta


def parallactic_motion(epochs, cntr_ra, cntr_dec, unit, refepoch, ephemeris=earth_ephemeris, parallax=1):
    """
    :param epochs: array of times in Julian year. Use astropy.time.Time.jyear to convert outside of this.
    :param cntr_ra: right ascension coordinate about which to calculate the parallactic motion. Should be in
                    the appropriate form for having units of unit.
    :param cntr_dec: declination coordinate about which to calculate the parallactic motion. Should be in
                    the appropriate form for having units of unit.
    :param unit: from Astropy.unit. Must be such that astropy.coordinates.Angle(cntr_ra, unit=unit) is sensical.
    :param refepoch: reference epoch in julian year.
    :param parallax: float. The parallax angle in milli-arcseconds.
    :param ephemeris: function.
          Function which intakes an array of Julian years and returns an array of shape (3,t.size)
          with the xyz components of the ephemeris at times t (along rows 0, 1 and 2 respectively).
    :return: [array, array]
    parallax motion about the center coordinate. E.g. Parallax_ra - cntr_ra and Parallax_dec - cntr_dec
    Where Parallax_ra would be an array of RA coordinates for parallax motion alone
    Output will have units of unit.
    """
    delta_ra, delta_dec = epoch_topocentric_coordinates(Angle(cntr_ra, unit=unit).rad,
                                                    Angle(cntr_dec, unit=unit).rad, parallax,
                                                    mura=0, mudec=0, vrad=0, t=epochs,
                                                    refepoch=refepoch, ephem=ephemeris)[2:]
    delta_ra, delta_dec = Angle(delta_ra, unit='radian').to(unit).value, Angle(delta_dec, unit='radian').to(unit).value
    return delta_ra, delta_dec