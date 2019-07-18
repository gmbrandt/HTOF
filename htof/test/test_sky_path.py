import numpy as np
from astropy.time import Time
from astropy import units

from htof.sky_path import earth_ephemeris, earth_sun_l2_ephemeris, epoch_topocentric_coordinates


def test_earth_ephemeris():
    perihelion, aphelion = Time(['2019-01-02T15:30:00.00', '2019-07-04T01:40:00'], format='isot', scale='utc')
    assert np.allclose(np.sum(earth_ephemeris([perihelion.jyear, aphelion.jyear])**2, axis=0),
                       [0.9832730, 1.0166664], atol=0.004)


def test_earth_sun_l2_ephemeris():
    t = Time('2019-01-02T15:30:00.00', format='isot', scale='utc')
    assert np.allclose(np.sqrt(np.sum(earth_ephemeris(t)**2)) * 1.511 / 1.496,
                       np.sqrt(np.sum(earth_sun_l2_ephemeris(t)**2)))


class TestEpochTopocentricCoordinates:
    def test_parallactic_motion_linear_in_parallax(self):
        alphadeg = np.random.random() * 360 - 180  # random float between -180 and 180
        deltadeg = np.random.random() * 180 - 90  # random float between -90 and 90
        mura = 0  # mas/yr
        parallax = 1  # mas
        mudec = 0  # mas/yr
        vrad = 0  # km/s
        refepoch = Time(1991.25, format='decimalyear', scale='tcb').jyear
        times = np.linspace(refepoch - 1, refepoch + 1, num=100)

        # alpha and delta are the radian values of the stars central coordinate on the sky.
        alpha = (alphadeg * units.degree).to(units.rad).value
        delta = (deltadeg * units.degree).to(units.rad).value

        f_ephem = earth_ephemeris
        ra, dec = epoch_topocentric_coordinates(alpha, delta, parallax, mura, mudec, vrad, times, refepoch, f_ephem)[:2]
        ra2, dec2 = epoch_topocentric_coordinates(alpha, delta, 2 * parallax, mura, mudec, vrad, times, refepoch, f_ephem)[:2]
        assert np.allclose([ra2 - alpha, dec2 - delta], 2 * np.array([ra - alpha, dec - delta]), rtol=1E-5)
