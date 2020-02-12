from htof.main import Astrometry
from htof.validation.utils import refit_hip_object
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
        result = refit_hip_object('hip1', self.dirname, hip_id, use_parallax=True)
        print(result)
        soltype = result[3]
        if result[0] is None:
            return {'hip_id': hip_id, 'diff_ra': None, 'diff_dec': None, 'plx' : None, 'diff_pm_ra': None, 'diff_pm_dec': None,
                    'soltype': soltype}
        else:
            plx, ra, dec, pm_ra, pm_dec = result[0][0:5]
            return {'hip_id': hip_id, 'diff_ra': ra, 'diff_dec': dec, 'plx' : plx, 'diff_pm_ra': pm_ra, 'diff_pm_dec': pm_dec,
                    'soltype': soltype}

if __name__ == "__main__":
    filename = 'hip1_fits_p' + (str)(os.getpid()) + '.csv'
    print(filename)
    dirname = '/home/dmichalik/HIPPARCOS_REREDUCTION/hipparcosOriginalIntermediateData'
    files = [i for i in os.listdir(dirname) if i.endswith("75152.txt")]
    print('fitting ', len(files))
    try:
        pool = Pool(3) # set number of processors
        engine = Engine(dirname)
        data_outputs = pool.map(engine, files)
        out = Table(data_outputs)
        out.write(filename, overwrite=True)
    finally: # To make sure processes are closed in the end, even if errors happen
        pool.close()
        pool.join()
