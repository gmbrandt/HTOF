"""
Script for looking at d_chi^2/d(a_i) where a_i are the standard astrometric parameters.
"""
import numpy as np
from astropy.coordinates import Angle
from argparse import ArgumentParser
from multiprocessing import Pool
import os

from htof.main import Astrometry
from astropy.table import Table


def chisq_partials(hip_id: str, iad_dir: str, reduction: str, fit_degree=1):
    try:
        astro = Astrometry(reduction, hip_id, iad_dir, central_epoch_ra=1991.25, normed=False,
                           central_epoch_dec=1991.25, format='jyear', fit_degree=fit_degree, use_parallax=False)
        ra_resid = Angle(astro.data.residuals.values * np.sin(astro.data.scan_angle.values), unit='mas')
        dec_resid = Angle(astro.data.residuals.values * np.cos(astro.data.scan_angle.values), unit='mas')
    except FileNotFoundError:
        return np.ones(5, dtype=float) * 10000
    return astro.fitter._chi2_vector(ra_resid.mas, dec_resid.mas)


def find_probable_outliers(hip_id, iad_dir, reduction, fit_degree=1, max_n_reject=1, thresh=0.1):
    try:
        astro = Astrometry(reduction, hip_id, iad_dir, central_epoch_ra=1991.25, normed=False,
                           central_epoch_dec=1991.25, format='jyear', fit_degree=fit_degree, use_parallax=False)
        ra_resid = Angle(astro.data.residuals.values * np.sin(astro.data.scan_angle.values), unit='mas')
        dec_resid = Angle(astro.data.residuals.values * np.cos(astro.data.scan_angle.values), unit='mas')
        dchisq_per_epoch = astro.fitter.astrometric_solution_vector_components['ra'] * ra_resid.mas.reshape(-1, 1) + \
                           astro.fitter.astrometric_solution_vector_components['dec'] * dec_resid.mas.reshape(-1, 1)
        # compute sum of squared components for a random sampling of rejected observations. See if ignoring one or two
        # give you chisq zero. We might need to take two pairs at a time, but perhaps we can reject one at a time, which
        # will be very computationally cheap
        epochs = find_epochs_to_reject(dchisq_per_epoch, max_n_reject, thresh)
        return epochs
    except FileNotFoundError:
        return []


def find_epochs_to_reject(dchisq_per_epoch, max_n_reject, thresh=0.1):
    init_dchisq = np.sum(np.sum(dchisq_per_epoch, axis=0)**2)
    reject_idx = []
    n_reject = 0
    dchisq_too_large = (init_dchisq > thresh)
    idx = list(np.arange(dchisq_per_epoch.shape[0]))
    while n_reject < max_n_reject and dchisq_too_large:
        # compute sum of dchisq components**2 for every possible combination of rejecting one observation.
        trials = np.ones((len(dchisq_per_epoch), len(dchisq_per_epoch)))
        np.fill_diagonal(trials, 0)
        trials = np.stack([trials] * dchisq_per_epoch.shape[1], axis=-1)
        dchisq_trials = dchisq_per_epoch * trials
        sum_squared_components = np.sum(np.sum(dchisq_trials, axis=1)**2, axis=1)
        reject = np.argmin(sum_squared_components).flatten()[0]
        # save the true index of the rejected observation
        reject_idx.append(idx[reject])
        # remove the rejected observation from the dchisq array and the true index array.
        idx.pop(reject)
        dchisq_per_epoch = np.delete(dchisq_per_epoch, reject, axis=0)
        # see if we should keep trying to reject
        n_reject += 1
        dchisq_too_large = (np.min(sum_squared_components) > thresh)
    return reject_idx


class ChisqNullTestEngine(object):
    def __init__(self, dirname, reduction):
        self.dirname = dirname
        self.reduction = reduction

    def __call__(self, hip_id):
        result = chisq_partials(hip_id, self.dirname, self.reduction)
        return {'hip_id': hip_id, 'dxdra0': result[0], 'dxddec0': result[1], 'dxdmura': result[2], 'dxdmudec': result[3]}


if __name__ == "__main__":
    parser = ArgumentParser(description='Script for checking the partial derivatives of chisquared. ignores parallax.')
    parser.add_argument("-dir", "--iad-directory", required=True, default=None,
                        help="full path to the intermediate data directory")
    parser.add_argument("-hr", "--hip-reduction", required=True, default=None, type=int,
                        help="integer. 1 for 1997 reduction, 2 for 2007 CD reduction, 21 for 2007 IADT tool.")
    parser.add_argument("-o", "--output-file", required=False, default=None,
                        help="The output filename, with .csv extension. E.g. hip1_refit.csv."
                             "Will default to hip_processid.csv.")
    parser.add_argument("-c", "--cores", required=False, default=1, type=int,
                        help="Number of cores to use. Default is 1.")
    parser.add_argument("-in", "--in-list", required=True, type=str,
                        help="In list of hip_ids to check. Must be a standard text file OR a .csv file "
                             "with a header called hip_id which gives the hip_id of the sources you want to fit")
    parser.add_argument("--debug", action='store_true', default=False, required=False,
                        help='If true, this will run the refit test on only 500 sources. Useful to check for '
                             'filepath problems before running the full test on all ~100000 sources.')

    args = parser.parse_args()
    if '.csv' not in args.in_list:
        hip_ids = np.genfromtxt(args.in_list).flatten().astype(str)
    else:
        hip_ids = np.array(Table.read(args.in_list)['hip_id'].data, dtype=str).flatten()

    if args.output_file is None:
        output_file = 'hip' + args.hip_reduction + '_chisq_null_test' + (str)(os.getpid()) + '.csv'
    else:
        output_file = args.output_file

    engine = ChisqNullTestEngine(args.iad_directory, 'hip' + str(args.hip_reduction))
    if args.debug:
        hip_ids = hip_ids[:500]
    print('will check {0} total hip {1} objects'.format(len(hip_ids), str(args.hip_reduction)))
    print('will save output table at', output_file)
    try:
        pool = Pool(args.cores)
        data_outputs = pool.map(engine, hip_ids)
        out = Table(data_outputs)
        out.sort('hip_id')
        out.write(output_file, overwrite=True)
    finally: # To make sure processes are closed in the end, even if errors happen
        pool.close()
        pool.join()
