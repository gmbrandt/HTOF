"""
Utility functions for the validation scripts.
"""
from htof.parse import DataParser, HipparcosOriginalData, HipparcosRereductionDVDBook, HipparcosRereductionJavaTool
from htof.fit import AstrometricFitter
from htof.sky_path import parallactic_motion, earth_ephemeris
from astropy import time
from astropy.coordinates import Angle
from astropy.table import Table
from glob import glob
import os
import warnings
import numpy as np


def refit_hip_fromdata(data: DataParser, fit_degree, cntr_RA=Angle(0, unit='degree'), cntr_Dec=Angle(0, unit='degree'),
                       use_parallax=False):
    data.calculate_inverse_covariance_matrices()
    # generate parallax motion
    jyear_epoch = time.Time(data.julian_day_epoch(), format='jd', scale='tcb').jyear
    # note that ra_motion and dec_motion are in degrees here.
    # generate sky path
    year_epochs = jyear_epoch - time.Time(1991.25, format='decimalyear', scale='tcb').jyear
    ra_motion, dec_motion = parallactic_motion(jyear_epoch, cntr_RA.degree, cntr_Dec.degree, 'degree',
                                               time.Time(1991.25, format='decimalyear', scale='tcb').jyear,
                                               ephemeris=earth_ephemeris)  # Hipparcos was in a geostationary orbit.

    ra_resid = Angle(data.residuals.values * np.sin(data.scan_angle.values), unit='mas')
    dec_resid = Angle(data.residuals.values * np.cos(data.scan_angle.values), unit='mas')
    # instantiate fitter
    fitter = AstrometricFitter(data.inverse_covariance_matrix, year_epochs, normed=False,
                               use_parallax=use_parallax, fit_degree=fit_degree,
                               parallactic_pertubations={'ra_plx': Angle(ra_motion, 'degree').mas,
                                                         'dec_plx': Angle(dec_motion, 'degree').mas})

    fit_coeffs, errors, chisq = fitter.fit_line(ra_resid.mas, dec_resid.mas, return_all=True)
    if not use_parallax:
        fit_coeffs = np.hstack([[0], fit_coeffs])
        errors = np.hstack([[0], errors])
    # pad so coeffs and errors are 9 long.
    errors = np.pad(errors, (0, 9 - len(fit_coeffs)))
    fit_coeffs = np.pad(fit_coeffs, (0, 9 - len(fit_coeffs)))
    # calculate the chisquared partials
    sin_scan = np.sin(data.scan_angle.values)
    cos_scan = np.cos(data.scan_angle.values)
    dt = data.epoch - 1991.25
    chi2_vector = (2 * data.residuals.values / data.along_scan_errs.values ** 2 * np.array(
        [sin_scan, cos_scan, dt * sin_scan, dt * cos_scan])).T
    chi2_partials = np.sum(chi2_vector, axis=0) ** 2
    return fit_coeffs, errors, chisq, chi2_partials


def refit_hip1_object(iad_dir, hip_id, hip_dm_g=None, use_parallax=False):
    data = HipparcosOriginalData()
    data.parse(star_id=hip_id, intermediate_data_directory=iad_dir)
    fname = os.path.join(iad_dir, hip_id + '.txt')

    plx, cntr_RA, cntr_Dec, pmRA, pmDec, soltype = get_cat_values_hip1(fname)
    soltype = soltype.strip()
    accRA, accDec, jerkRA, jerkDec = 0, 0, 0, 0
    fit_degree = {'5': 1}.get(soltype, None)
    if hip_dm_g is not None and (soltype == '7' or soltype == '9'):
        # do the fit for seven/nine parameter fits if we have the 7th and 9th parameters.
        idx = np.searchsorted(hip_dm_g['hip_id'].data, int(hip_id))  # int(hip_id) strips leading zeroes.
        accRA, accDec, jerkRA, jerkDec = hip_dm_g[idx][['accRA', 'accDec', 'jerkRA', 'jerkDec']]
        fit_degree = {'5': 1, '7': 2, '9': 3}.get(soltype, None)

    if fit_degree is not None:
        # abscissa residuals are always with respect to 5p solution for hip1. Do not feed accRA, etc..
        # when reconstructing the skypath
        diffs, errors, chisq, chi2_partials = refit_hip_fromdata(data, fit_degree, cntr_RA=cntr_RA, cntr_Dec=cntr_Dec,
                                                                 use_parallax=use_parallax)
        if soltype == '7' or soltype == '9':
            # to account for the 0.81 yr and 1.69 yr^2 basis offsets present in 7 and 9 parameter solutions:
            diffs -= compute_basis_offsets(accRA, accDec, jerkRA, jerkDec)
            # diffs[-4:] are not differences but actually the accRA, accDec, jerkRA, jerkDec parameters, so we need to subtract those off
            diffs[-4:] -= np.array([accRA, accDec, jerkRA, jerkDec])

        return tuple((diffs, errors, chisq, chi2_partials, soltype))
    else:
        return [None] * 9, [None] * 9, None, [None] * 4, soltype


def refit_hip2_object(iad_dir, hip_id, catalog: Table, seven_p_annex: Table = None, nine_p_annex: Table = None, use_parallax=False):
    data = HipparcosRereductionDVDBook()
    data.parse(star_id=hip_id, intermediate_data_directory=iad_dir)

    cat_params, cat_errors, soltype = get_cat_values_hip2(hip_id, catalog)
    cntr_RA, cntr_Dec = cat_params[1:3]
    soltype = soltype.strip()
    fit_degree = {'5': 1, '7': 2, '9': 3}.get(soltype[-1], None)
    # do the fit for seven/nine parameter fits if we have the 7th and 9th parameters.
    accRA_err, accDec_err, jerkRA_err, jerkDec_err = 0, 0, 0, 0
    if seven_p_annex is not None and fit_degree >= 2:
        idx = np.searchsorted(seven_p_annex['hip_id'].data, int(hip_id))  # int(hip_id) strips leading zeroes.
        accRA_err, accDec_err = seven_p_annex[idx][['acc_ra_err', 'acc_dec_err']]
    if nine_p_annex is not None and fit_degree == 3:
        idx = np.searchsorted(nine_p_annex['hip_id'].data, int(hip_id))  # int(hip_id) strips leading zeroes.
        jerkRA_err, jerkDec_err = nine_p_annex[idx][['jerk_ra_err', 'jerk_dec_err']]
    # do the fit
    cat_errors = np.hstack([cat_errors, [accRA_err, accDec_err, jerkRA_err, jerkDec_err]])
    if fit_degree is not None:
        diffs, errors, chisq, chi2_partials = refit_hip_fromdata(data, fit_degree, cntr_RA=cntr_RA, cntr_Dec=cntr_Dec,
                                                                 use_parallax=use_parallax)
        return tuple((diffs, errors - cat_errors, chisq, chi2_partials, soltype))
    else:
        return [None] * 9, [None] * 9, None, [None] * 4, soltype


def refit_hip21_object(iad_dir, hip_id, use_parallax=False):
    data = HipparcosRereductionJavaTool()
    data.parse(star_id=hip_id, intermediate_data_directory=iad_dir)
    fname = glob(os.path.join(iad_dir, '**/', "H" + hip_id.zfill(6) + ".csv"))[0]

    plx, cntr_RA, cntr_Dec, pmRA, pmDec, soltype = get_cat_values_hip21(fname)
    soltype = soltype.strip()
    fit_degree = {'5': 1, '7': 2, '9': 3}.get(soltype[-1], None)
    # For now, just do the 5 parameter sources of Hip2.
    if fit_degree == 1:
        diffs, errors, chisq, chi2_partials = refit_hip_fromdata(data, fit_degree, cntr_RA=cntr_RA, cntr_Dec=cntr_Dec,
                                                                 use_parallax=use_parallax)
        return tuple((diffs, errors, chisq, chi2_partials, soltype))
    else:
        return [None] * 9, [None] * 9, None, [None] * 4, soltype


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


def get_cat_values_hip21(fname):
    with open(fname) as f:
        lines = f.readlines()
        try:
            pmRA = float(lines[2].split()[3])
            pmDec = float(lines[2].split()[4])
            cntr_RA = Angle(float(lines[2].split()[0]), unit='degree')
            cntr_Dec = Angle(float(lines[2].split()[1]), unit='degree')
            plx = float(lines[2].split()[2])
            sol_type = str(lines[0].split()[4])
        except:
            raise UnboundLocalError('could not read pmRA or pmDec from intermediate data of {0}'.format(fname))
    return plx, cntr_RA, cntr_Dec, pmRA, pmDec, str(int(sol_type))


def get_cat_values_hip2(hip_id, catalog: Table):
    idx = np.searchsorted(catalog['hip_id'].data, int(hip_id))  # int(hip_id) strips leading zeroes.
    plx, cntr_RA, cntr_Dec, pmRA, pmDec, soltype = catalog[idx]['plx', 'ra', 'dec', 'pmRA', 'pmDec', 'soltype']
    plx_e, cntr_RA_e, cntr_Dec_e, pmRA_e, pmDec_e = catalog[idx]['plx_err', 'ra_err', 'dec_err', 'pmRA_err', 'pmDec_err']
    cntr_RA, cntr_Dec = Angle(cntr_RA, unit=catalog['ra'].unit), Angle(cntr_Dec, unit=catalog['dec'].unit)
    #
    params = [plx, cntr_RA, cntr_Dec, pmRA, pmDec]
    errors = [plx_e, cntr_RA_e, cntr_Dec_e, pmRA_e, pmDec_e]
    return params, errors, str(int(soltype))


def load_hip2_catalog(catalog_path):
    catalog = Table.read(catalog_path, format='ascii')
    catalog = catalog['col1', 'col7', 'col5', 'col6', 'col8', 'col9', 'col12', 'col10', 'col11', 'col13', 'col14', 'col2']
    catalog['col5'] = Angle(Angle(catalog['col5'], unit='rad'), unit='degree')
    catalog['col6'] = Angle(Angle(catalog['col6'], unit='rad'), unit='degree')
    # names from Table G.3 of Hipparcos: The new reduction of the raw data (2007)
    for name, rename in zip(['col1', 'col7', 'col5', 'col6', 'col8', 'col9', 'col12', 'col10', 'col11', 'col13', 'col14', 'col2'],
                            ['hip_id', 'plx', 'ra', 'dec', 'pmRA', 'pmDec', 'plx_err', 'ra_err', 'dec_err', 'pmRA_err', 'pmDec_err', 'soltype']):
        catalog.rename_column(name, rename)
    # sort for quick retrieval of data for any source.
    catalog.sort('hip_id')
    return catalog


def load_hip2_seven_p_annex(path):
    catalog = Table.read(path, format='ascii')
    catalog = catalog['col1', 'col3', 'col4', 'col5', 'col6']
    # names from Table G.5 of Hipparcos: The new reduction of the raw data (2007)
    for name, rename in zip(['col1', 'col3', 'col4', 'col5', 'col6'],
                            ['hip_id', 'acc_ra', 'acc_dec', 'acc_ra_err', 'acc_dec_err']):
        catalog.rename_column(name, rename)
    # sort for quick retrieval of data for any source.
    catalog.sort('hip_id')
    return catalog


def load_hip2_nine_p_annex(path):
    catalog = Table.read(path, format='ascii')
    catalog = catalog['col1', 'col3', 'col4', 'col5', 'col6', 'col7', 'col8', 'col9', 'col10']
    # names from Table G.6 of Hipparcos: The new reduction of the raw data (2007)
    for name, rename in zip(['col1', 'col3', 'col4', 'col5', 'col6', 'col7', 'col8', 'col9', 'col10'],
                            ['hip_id', 'acc_ra', 'acc_dec', 'jerk_ra', 'jerk_dec', 'acc_ra_err', 'acc_dec_err', 'jerk_ra_err', 'jerk_dec_err']):
        catalog.rename_column(name, rename)
    # sort for quick retrieval of data for any source.
    catalog.sort('hip_id')
    return catalog


def load_hip1_dm_annex(catalog_path):
    # load the hip1 doubles and multiples annex for the accelerating systems
    if not os.path.exists(catalog_path):
        warnings.warn('Doubles and multiples annex {0} file not found. Will only fit 5 parameter solutions.'
                      'For 7, 9 parameter validation, please download hip_dm_g.dat from '
                      'https://cdsarc.unistra.fr/ftp/I/239/hip_dm_g.dat.gz (it is not on Vizier as of Sep 16 2020). '
                      'Unzip it, and place it inside the hip1'
                      'intermediate data directory.'.format(catalog_path), UserWarning)
        return None
    catalog = Table.read(catalog_path, format='ascii')
    # column designations are from the Hipparcos and Tycho catalogs, Vol1, Table 2.3.3.
    catalog = catalog['col1', 'col2', 'col3', 'col4', 'col5', 'col7', 'col8', 'col9', 'col10']
    for name, rename in zip(['col1', 'col2', 'col3', 'col4', 'col5', 'col7', 'col8', 'col9', 'col10'],
                            ['hip_id', 'accRA', 'accDec', 'accRA_err', 'accDec_err', 'jerkRA', 'jerkDec', 'jerkRA_err', 'jerkDec_err']):
        catalog.rename_column(name, rename)
    # sort for quick retrieval of data for any source.
    catalog.sort('hip_id')
    return catalog.filled(0)  # this sets missing acc/jerks to 0 and returns a Table instead of a Masked Table


def load_hip1_catalog(catalog_path):
    catalog = Table.read(catalog_path, format='ascii')
    # catalog should be fetched from https://cdsarc.unistra.fr/ftp/I/239/hip_main.dat.gz
    # column designations are from https://cdsarc.unistra.fr/ftp/I/239/ReadMe
    catalog = catalog['col2', 'col12', 'col9', 'col10', 'col13', 'col14', 'col17', 'col15', 'col16', 'col18', 'col19', 'col60']
    for name, rename in zip(['col2', 'col12', 'col9', 'col10', 'col13', 'col14', 'col17', 'col15', 'col16', 'col18', 'col19', 'col60'],
                            ['hip_id', 'plx', 'ra', 'dec', 'pmRA', 'pmDec', 'plx_err', 'ra_err', 'dec_err', 'pmRA_err', 'pmDec_err', 'MultFlag']):
        catalog.rename_column(name, rename)
    # sort for quick retrieval of data for any source.
    catalog.sort('hip_id')
    return catalog.filled(0)  # this sets missing entries to 0 and returns a Table instead of a Masked Table


def compute_basis_offsets(accRA, accDec, jerkRA, jerkDec):
    # account for the 0.81 yr and 1.69 yr^2 basis offsets:
    # See THE HIPPARCOS CATALOGUE DOUBLE AND MULTIPLE SYSTEMS ANNEX by Lennart Lindegren (1997)
    a, b = 0.81, 1.69  # yr^2
    offsets = np.array([0, -a / 2 * accRA, -a / 2 * accDec, -b / 6 * jerkRA, -b / 6 * jerkDec, 0, 0, 0, 0])
    # compute differences between reference parameters and our best fit parameters
    return offsets
