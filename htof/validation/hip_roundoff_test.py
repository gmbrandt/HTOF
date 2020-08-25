from htof.parse import HipparcosOriginalData, HipparcosRereductionDVDBook
from htof.fit import AstrometricFitter
from htof.sky_path import parallactic_motion, earth_ephemeris
from astropy import time
from astropy.coordinates import Angle
import numpy as np
from argparse import ArgumentParser
import os


def parse(datacsv, data, data_choice='MERGED', perturb=True):
    if (data_choice is not 'NDAC') and (data_choice is not 'FAST') and (data_choice is not 'MERGED'):
        raise ValueError('data choice has to be either NDAC or FAST or MERGED.')
    datacsv = data._fix_unnamed_column(datacsv)
    if perturb:
        fourdig = ['IA3', 'IA4', 'IA5', 'IA6', 'IA7']
        twodig = ['IA8', 'IA9']
        threedig = ['IA10']
        for cols, digits in zip([fourdig, threedig, twodig], [4, 3, 2]):
            datacsv[cols] = datacsv[cols].to_numpy() + np.random.uniform(-.5 / (10**digits), .5 / (10**digits),
                                                                         datacsv[cols].to_numpy().shape)
    datacsv = data._select_data(datacsv, data_choice)
    # compute scan angles and observations epochs according to van Leeuwen & Evans 1997, eq. 11 & 12.
    data.scan_angle = np.arctan2(datacsv['IA3'], datacsv['IA4'])  # unit radians, arctan2(sin, cos)
    # Use the larger denominator when computing the epoch offset.
    # This increases numerical precision and avoids NaNs if one of the two fields (IA3, IA4) is exactly zero.
    data._epoch = 1991.25 + (datacsv['IA6'] / datacsv['IA3']).where(abs(datacsv['IA3']) > abs(datacsv['IA4']),
                                                              (datacsv['IA7'] / datacsv['IA4']))
    data.residuals = datacsv['IA8']  # unit milli-arcseconds (mas)
    data.along_scan_errs = datacsv['IA9']  # unit milli-arcseconds
    return data


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    parser = ArgumentParser(description='Enter a hip id (string)')
    parser.add_argument("-i", "--hip-id", required=True, default='100564', #1276 is bad
                        help="hip id of the target")
    args = parser.parse_args()

    hip_id = args.hip_id

    data_choice = 'MERGED'
    iad_dir = '/home/gmbrandt/Documents/hipparcosOriginalIntermediateData/'
    # set accRA, accDec to 0 to change to a five parameter model which generates the data.
    accRA = 0#-23.07
    accDec = 0#26.31

    # fit_degree (degree of the model which is fit to the data, i.e. what HTOF uses to fit to the motion)
    # fit_degree = 1 is 5 parameter, fit_degree = 2 is 7 parameter.
    fit_degree = 1
    use_parallax = False
    perturb = True # Whether or not to randomly perturb the digit beyond the last significant digit of the intermediate data.
    ###### end of user params

    with open(os.path.join(iad_dir, hip_id + '.txt')) as f:
        lines = f.readlines()
        try:
            cntr_ra = Angle(float(lines[2].split(':')[1].split('Right')[0]), unit='degree')
            cntr_dec = Angle(float(lines[3].split(':')[1].split('Declination')[0]), unit='degree')
            plx = float(lines[4].split(':')[1].split('Trig')[0])
            pmRA = float(lines[5].split(':')[1].split('Proper')[0])
            pmDec = float(lines[6].split(':')[1].split('Proper')[0])
        except:
            raise UnboundLocalError('could not read pmRA or pmDec from intermediate data of {0}'.format(fname))

    data = HipparcosOriginalData()
    # manually loading file
    datacsv = data.read_intermediate_data_file(hip_id, iad_dir, skiprows=10, header='infer', sep='\s*\|\s*')
    pmRAs, pmDecs = [], []
    for i in range(100):
        # parsing with a pertubation
        data = parse(datacsv.copy(), data, data_choice='MERGED', perturb=perturb)
        #data.parse(star_id=hip_id, intermediate_data_directory=iad_dir,
        #           data_choice=data_choice)
        data.calculate_inverse_covariance_matrices()
        mas_to_degree = 1./60/60/1000

        jyear_epoch = time.Time(data.julian_day_epoch(), format='jd', scale='tcb').jyear
        # note that ra_motion and dec_motion are in degrees here.
        # generate sky path
        year_epochs = jyear_epoch - time.Time(1991.25, format='decimalyear', scale='tcb').jyear
        ra_ref = Angle(pmRA * mas_to_degree * year_epochs, unit='degree')
        dec_ref = Angle(pmDec * mas_to_degree * year_epochs, unit='degree')
        # generate parallax motion
        ra_motion, dec_motion = parallactic_motion(jyear_epoch, cntr_ra.degree, cntr_dec.degree, 'degree',
                                                   time.Time(1991.25, format='decimalyear', scale='tcb').jyear,
                                                   ephemeris=earth_ephemeris)  # Hipparcos was in a geostationary orbit.
        if use_parallax:
            ra_ref += Angle(ra_motion * plx, unit='degree')
            dec_ref += Angle(dec_motion * plx, unit='degree')
        #acceleration terms
        ra_ref += Angle(1/2*accRA * mas_to_degree * (year_epochs**2 - 0.81), unit='degree')
        dec_ref += Angle(1/2*accDec * mas_to_degree * (year_epochs**2 - 0.81), unit='degree')
        #
        ra_model = 1.*ra_ref
        dec_model = 1.*dec_ref

        ra_resid = Angle(data.residuals.values * np.sin(data.scan_angle.values), unit='mas')
        dec_resid = Angle(data.residuals.values * np.cos(data.scan_angle.values), unit='mas')
        ra_ref += ra_resid
        dec_ref += dec_resid
        # instantiate fitter
        fitter = AstrometricFitter(data.inverse_covariance_matrix, year_epochs,
                                   use_parallax=use_parallax, fit_degree=fit_degree,
                                   parallactic_pertubations={'ra_plx': Angle(ra_motion, 'degree').mas,
                                                             'dec_plx': Angle(dec_motion, 'degree').mas})
        # fit
        fit_coeffs = fitter.fit_line(ra_ref.mas, dec_ref.mas)
        pmRAs.append(fit_coeffs[2])
        pmDecs.append(fit_coeffs[3])
        #print(pmRA, pmDec, 'original pmRA and pmDec')
        #print(fit_coeffs[-2:], 'best fit proper motions')

        #if use_parallax:
        #    print(fit_coeffs[0], ' = best fit parallax angle (mas)')
        #    print(plx, ' = input parallax angle (mas)')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6)) # width by length
    ax1.hist(np.array(pmRAs) - pmRA, label=r'$\Delta \mu_{RA}$')
    ax2.hist(np.array(pmDecs) - pmDec, label=r'$\Delta \mu_{DEC}$')
    ax1.legend(loc='best')
    ax2.legend(loc='best')
    ax1.set_xlabel(r'$\Delta \mu_{RA}$')
    ax2.set_xlabel(r'$\Delta \mu_{DEC}$')
    ax1.set_ylabel('Count')
    ax2.set_ylabel('Count')
    plt.title('Discrepancies w/ respect to catalog PMs for Hip {0}'.format(hip_id))
    #plt.savefig('/home/gmbrandt/Downloads/discrep_{0}.pdf'.format(hip_id))
    plt.show()
