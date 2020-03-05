"""
Utility functions for the validation scripts.
"""
from htof.parse import DataParser, HipparcosOriginalData, HipparcosRereductionData
from htof.fit import AstrometricFitter
from htof.sky_path import parallactic_motion, earth_ephemeris
from astropy import time
from astropy.coordinates import Angle
import os
import numpy as np


def refit_hip_fromdata(data: DataParser, fit_degree, pmRA, pmDec, accRA=0, accDec=0, jerkRA=0, jerkDec=0,
                       cntr_RA=Angle(0, unit='degree'), cntr_Dec=Angle(0, unit='degree'),
                       plx=0, use_parallax=False):
    data.calculate_inverse_covariance_matrices()
    mas_to_degree = 1. / 60 / 60 / 1000
    # generate parallax motion
    jyear_epoch = time.Time(data.julian_day_epoch(), format='jd', scale='tcb').jyear
    # note that ra_motion and dec_motion are in degrees here.
    # generate sky path
    year_epochs = jyear_epoch - time.Time(1991.25, format='decimalyear', scale='tcb').jyear
    ra_ref = Angle(pmRA * mas_to_degree * year_epochs, unit='degree')
    dec_ref = Angle(pmDec * mas_to_degree * year_epochs, unit='degree')
    # acceleration terms
    ra_ref += Angle(1 / 2 * accRA * mas_to_degree * (year_epochs ** 2 - 0.81), unit='degree')
    dec_ref += Angle(1 / 2 * accDec * mas_to_degree * (year_epochs ** 2 - 0.81), unit='degree')
    # jerk terms
    ra_ref += 0
    dec_ref += 0
    # add parallax if necessary
    ra_motion, dec_motion = parallactic_motion(jyear_epoch, cntr_RA.degree, cntr_Dec.degree, 'degree',
                                               time.Time(1991.25, format='decimalyear', scale='tcb').jyear,
                                               ephemeris=earth_ephemeris)  # Hipparcos was in a geostationary orbit.

    ra_resid = Angle(data.residuals.values * np.sin(data.scan_angle.values), unit='mas')
    dec_resid = Angle(data.residuals.values * np.cos(data.scan_angle.values), unit='mas')
    ra_ref += ra_resid
    dec_ref += dec_resid
    # instantiate fitter
    fitter = AstrometricFitter(data.inverse_covariance_matrix, year_epochs, normed=False,
                               use_parallax=use_parallax, fit_degree=fit_degree,
                               parallactic_pertubations={'ra_plx': Angle(ra_motion, 'degree').mas,
                                                         'dec_plx': Angle(dec_motion, 'degree').mas})
    fit_coeffs, errors, chisq = fitter.fit_line(ra_ref.mas, dec_ref.mas, return_all=True)
    if not use_parallax:
        fit_coeffs -= np.array([0, 0, pmRA, pmDec, accRA, accDec, jerkRA, jerkDec])[:len(fit_coeffs)]
    if use_parallax:
        fit_coeffs -= np.array([0, 0, 0, pmRA, pmDec, accRA, accDec, jerkRA, jerkDec])[:len(fit_coeffs)]
    return fit_coeffs, errors, chisq


def refit_hip_object(data_choice, iad_dir, hip_id, use_parallax=False):
    data = {'hip1': HipparcosOriginalData(), 'hip2': HipparcosRereductionData()}[data_choice]
    data.parse(star_id=hip_id, intermediate_data_directory=iad_dir)
    fname = os.path.join(iad_dir, hip_id + '.txt')
    if data_choice == 'hip1':
        plx, cntr_RA, cntr_Dec, pmRA, pmDec, soltype = get_cat_values_hip1(fname)
        soltype = soltype.strip()
        fit_degree = {'5': 1, '7': 2, '9': 3}.get(soltype, None)
    else:
        plx, cntr_RA, cntr_Dec, pmRA, pmDec, soltype = get_cat_values_hip2(fname)
    # For five/seven/nine parameter fits, do the fit. For other solution types, return None
    if fit_degree is not None:
        return tuple((*refit_hip_fromdata(data, fit_degree, pmRA, pmDec, accRA=0, accDec=0,
                                  jerkRA=0, jerkDec=0, cntr_RA=cntr_RA, cntr_Dec=cntr_Dec,
                                  plx=plx, use_parallax=use_parallax), soltype))
    else:
        return None, None, None, soltype


def get_cat_values_hip1(fname):
    with open(fname) as f:
        lines = f.readlines()
        try:
            pmRA = float(lines[5].split(':')[1].split('P')[0])
            pmDec = float(lines[6].split(':')[1].split('P')[0])
            cntr_RA = Angle(float(lines[2].split(':')[1].split('Right')[0]), unit='degree')
            cntr_Dec = Angle(float(lines[3].split(':')[1].split('Declination')[0]), unit='degree')
            plx = float(lines[4].split(':')[1].split('Trigonometric')[0])
            sol_type = str(lines[7].split(':')[1].split('Code')[0])
        except:
            raise UnboundLocalError('could not read pmRA or pmDec from intermediate data of {0}'.format(fname))
    return plx, cntr_RA, cntr_Dec, pmRA, pmDec, sol_type


def get_cat_values_hip2(iad_dir, hip_id):
    return 0,0,0,0,0,'' 