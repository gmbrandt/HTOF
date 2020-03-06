from htof.validation.utils import refit_hip1_object, refit_hip2_object, load_hip2_catalog
import os
from astropy.table import Table
from argparse import ArgumentParser
from glob import glob

from multiprocessing import Pool


class Engine(object):
    @staticmethod
    def format_result(result, hip_id, soltype):
        ra, dec, plx, pm_ra, pm_dec, acc_ra, acc_dec, jerk_ra, jerk_dec = [None]*9
        if soltype is '5':
            plx, ra, dec, pm_ra, pm_dec = result[0][0:5]
        elif soltype is '7':
            plx, ra, dec, pm_ra, pm_dec, acc_ra, acc_dec = result[0][0:7]
        elif soltype is '9':
            plx, ra, dec, pm_ra, pm_dec, acc_ra, acc_dec, jerk_ra, jerk_dec = result[0][0:9]
        return {'hip_id': hip_id, 'diff_ra': ra, 'diff_dec': dec, 'diff_plx' : plx, 'diff_pm_ra': pm_ra, 'diff_pm_dec': pm_dec,
                'soltype': soltype, 'diff_acc_ra': acc_ra, 'diff_acc_dec': acc_dec, 'diff_jerk_ra': jerk_ra, 'diff_jerk_dec': jerk_dec}


class Hip1Engine(Engine):
    def __init__(self, dirname, use_parallax, *args):
        self.dirname = dirname
        self.use_parallax = use_parallax

    def __call__(self, fname):
        hip_id = os.path.basename(fname).split('.txt')[0]
        result = refit_hip1_object(self.dirname, hip_id, use_parallax=self.use_parallax)
        soltype = result[3]
        return self.format_result(result, hip_id, soltype)


class Hip2Engine(Engine):
    def __init__(self, dirname, use_parallax, catalog):
        self.dirname = dirname
        self.catalog = catalog
        self.use_parallax = use_parallax

    def __call__(self, fname):
        hip_id = os.path.basename(fname).split('.d')[0].split('HIP')[1]
        result = refit_hip2_object(self.dirname, hip_id, self.catalog, use_parallax=self.use_parallax)
        soltype = result[3]
        return self.format_result(result, hip_id, soltype[-1])


if __name__ == "__main__":
    parser = ArgumentParser(description='Script for refitting the entire hipparcos catalog, 1997 or 2007.'
                                        'This will output a csv type file. Each row gives'
                                        'the difference in the best-fit parameters and the catalog parameters')
    parser.add_argument("-dir", "--iad-directory", required=True, default=None,
                        help="full path to the intermediate data directory")
    parser.add_argument("-hr", "--hip-reduction", required=True, default=None, type=int,
                        help="integer. 1 for 1997 reduction, 2 for 2007 CD reduction.")
    parser.add_argument("-o", "--output-file", required=False, default=None,
                        help="The output filename, with .csv extension. E.g. hip1_refit.csv."
                             "Will default to hip_processid.csv.")
    parser.add_argument("-c", "--cores", required=False, default=1, type=int,
                        help="Number of cores to use. Default is 1.")
    parser.add_argument("--ignore-parallax", required=False, action='store_true', default=False,
                        help="Number of cores to use. Default is 1.")
    parser.add_argument("-cpath", "--catalog-path", required=False, default=None,
                        help="path to the Hip re-reduction main catalog, e.g. Main_cat.d. Only required"
                             "if using hip 2.")

    args = parser.parse_args()

    # check arguments and assign values.
    if args.hip_reduction == 2 and args.catalog_path is None:
        raise ValueError('Hip 2 selected but no --catalog-path provided.')

    if args.output_file is None:
        output_file = 'hip' + args.hipreduction + '_refit' + (str)(os.getpid()) + '.csv'
    else:
        output_file = args.output_file

    # find the intermediate data files
    if args.hip_reduction == 1:
        files = glob(os.path.join(args.iad_directory, '*.txt'))[:3]
        engine = Hip1Engine
        catalog = None
    else:
        files = glob(os.path.join(args.iad_directory, '**/H*.d'))[:3]
        engine = Hip2Engine
        catalog = load_hip2_catalog(args.catalog_path)

    print('will fit {0} total hip {1} objects'.format(len(files), str(args.hip_reduction)))
    print('will save output table at', output_file)
    # do the fit.
    try:
        pool = Pool(args.cores)
        engine = engine(args.iad_directory, not args.ignore_parallax, catalog)
        data_outputs = pool.map(engine, files)
        out = Table(data_outputs)
        out.write(output_file, overwrite=True)
    finally: # To make sure processes are closed in the end, even if errors happen
        pool.close()
        pool.join()

