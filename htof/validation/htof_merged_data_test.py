from htof.main import Astrometry
from astropy.time import Time
import numpy as np
from astropy.coordinates import Angle
import os
from astropy.table import Table

from multiprocessing import Pool

class Engine(object):
    def __init__(self, dirname):
        self.dirname = dirname
    def __call__(self, fname):
        hip_id = fname.split('.txt')[0]        
        with open(os.path.join(self.dirname, fname)) as f:
            lines = f.readlines()
            try:
                pmRA = float(lines[5].split(':')[1].split('P')[0])
                pmDec = float(lines[6].split(':')[1].split('P')[0])
                soltype = lines[7].split(':')[1].split('Code')[0]
            except:
                return {'hip_id': hip_id, 'pmRA_fit': None,'pmDec_fit': None,'pmRA_cat': None, 'pmDec_cat': None,
                        'soltype': 'error during read'}

        if '5' in soltype:
            try:
                pmRA_fit, pmDec_fit = test_fit(pmRA, pmDec, hip_id, self.dirname)
                return {'hip_id': hip_id, 'pmRA_fit': pmRA_fit,'pmDec_fit': pmDec_fit,'pmRA_cat': pmRA, 'pmDec_cat': pmDec, 'soltype': soltype}
            except:
                return {'hip_id': hip_id, 'pmRA_fit': None,'pmDec_fit': None,'pmRA_cat': None, 'pmDec_cat': None,
'soltype': 'error during fit'}
        else:
            return {'hip_id': hip_id, 'pmRA_fit': None,'pmDec_fit': None,'pmRA_cat': None, 'pmDec_cat': None,
                    'soltype': soltype}


def test_fit(pmRA, pmDec, hip_id, iad_dir):
    # generate fitter and parse intermediate data
    astro = Astrometry('Hip1', hip_id, iad_dir, central_epoch_ra=1991.25, central_epoch_dec=1991.25,
                       format='jyear', fit_degree=1, use_parallax=False)
    chisq = np.sum(astro.data.residuals ** 2 / astro.data.along_scan_errs ** 2)
    # generate ra and dec for each observation.
    year_epochs = Time(astro.data.julian_day_epoch(), format='jd', scale='tcb').jyear - \
                  Time(1991.25, format='decimalyear').jyear
    ra = Angle(pmRA * year_epochs, unit='mas')
    dec = Angle(pmDec * year_epochs, unit='mas')
    # add residuals
    ra += Angle(astro.data.residuals.values * np.sin(astro.data.scan_angle.values), unit='mas')
    dec += Angle(astro.data.residuals.values * np.cos(astro.data.scan_angle.values), unit='mas')
    #
    coeffs, errors, chisq_found = astro.fit(ra.mas, dec.mas, return_all=True)
    return coeffs[2], coeffs[3]


if __name__ == "__main__":
    dirname = '/home/gmbrandt/Documents/hipparcosOriginalIntermediateData'
    files = [i for i in os.listdir(dirname) if i.endswith("txt")]
    print('fitting ', len(files))
    try:
        pool = Pool(3) # set number of processors
        engine = Engine(dirname)
        data_outputs = pool.map(engine, files)
        out = Table(data_outputs)
        out.write('hip1_fits.csv', overwrite=True)
    finally: # To make sure processes are closed in the end, even if errors happen
        pool.close()
        pool.join()
