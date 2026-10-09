"""
Compare the goodness-of-fit statistic F2 computed by htof, with and without the
ad-hoc correction of Brandt et al. 2021 (Section 4), against the catalog value
for the 6617 discrepant Hipparcos sources.

This is a validation *script*, not part of the htof library API.  Run it with::

    python -m htof.validation.htof_adhoc_correction_f2 \
        --iad-directory /path/to/Hip21 --output f2_comparison_results.csv

Nothing is executed at import time, so the module is safe to import.
"""
import argparse
import os

import numpy as np
from astropy.io import ascii as ascii_astropy
from astropy.table import Table

from htof.parse import HipparcosRereductionJavaTool
from htof.utils.resources import resource_filename

DEFAULT_IAD_DIRECTORY = os.path.join(os.getcwd(), 'htof/test/data_for_tests/Hip21')


def compute_f2_comparison(iad_directory=DEFAULT_IAD_DIRECTORY):
    """
    :param iad_directory: directory containing the Java-tool residual records
                          (the IAD) of the sources to check.
    :return: astropy.table.Table with one row per discrepant source.
    """
    discrepant_path = resource_filename('htof', 'data/hip21_java_nobs_discrepant.txt')
    discrepant = ascii_astropy.read(discrepant_path, names=["HIP", "diffNobs"])
    discrepant = discrepant[discrepant['diffNobs'] > 0]

    data = HipparcosRereductionJavaTool()
    results = np.zeros((len(discrepant), 5))

    for idx, hip_id in enumerate(discrepant["HIP"]):
        results[idx][0] = hip_id
        # parse data without ad-hoc correction, compute and store f2
        data.parse(star_id=hip_id, intermediate_data_directory=iad_directory,
                   attempt_adhoc_rejection=False)
        results[idx][1] = data.meta['catalog_f2']
        results[idx][2] = data.meta['calculated_f2']
        # parse data with ad-hoc correction, compute and store f2
        data.parse(star_id=hip_id, intermediate_data_directory=iad_directory,
                   attempt_adhoc_rejection=True)
        results[idx][3] = data.meta['calculated_f2']
        results[idx][4] = data.meta['calculated_f2'] - data.meta['catalog_f2']

    return Table(results, names=["HIP", "catalog_f2", "htof_f2_without", "htof_f2", "difference"],
                 dtype=['i8', 'f8', 'f8', 'f8', 'f8'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--iad-directory', default=DEFAULT_IAD_DIRECTORY,
                        help='Directory containing the Hipparcos 2 Java tool IAD files.')
    parser.add_argument('--output', default='f2_comparison_results.csv',
                        help='Path of the output csv table.')
    args = parser.parse_args(argv)

    table = compute_f2_comparison(args.iad_directory)
    table.write(args.output, overwrite=True)
    return table


if __name__ == '__main__':   # pragma: no cover
    main()
