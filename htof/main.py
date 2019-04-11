#!/usr/bin/env python
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from htof.fit import AstrometricFitter
from htof.parse import HipparcosRereductionData, calculate_covariance_matrices

"""
Utility functions for plotting.
"""


def plot_fitting_to_astrometric_data(astrometric_data):
    # solving
    fitter = AstrometricFitter(inverse_covariance_matrices=astrometric_data['covariance_matrix'],
                               epoch_times=astrometric_data['epoch_delta_t'])
    solution_vector = fitter.fit_line(ra_vs_epoch=astrometric_data['ra'],
                                      dec_vs_epoch=astrometric_data['dec'])
    # plotting
    plt.figure()
    plt.errorbar(astrometric_data['epoch_delta_t'], astrometric_data['ra'],
                 xerr=0, yerr=np.sqrt(astrometric_data['covariance_matrix'][:, 0, 0]),
                 fmt='ro', label='RA')
    plt.errorbar(astrometric_data['epoch_delta_t'], astrometric_data['dec'],
                 xerr=0, yerr=np.sqrt(astrometric_data['covariance_matrix'][:, 1, 1]),
                 fmt='bo', label='DEC')
    continuous_t = np.linspace(np.min(astrometric_data['epoch_delta_t']),
                               np.max(astrometric_data['epoch_delta_t']), num=200)
    ra0, dec0, mu_ra, mu_dec = solution_vector
    plt.plot(continuous_t, ra0 + mu_ra * continuous_t, 'r', label='RA fit')
    plt.plot(continuous_t, dec0 + mu_dec * continuous_t, 'b', label='DEC fit')
    plt.xlabel('$\Delta$ epoch')
    plt.ylabel('RA or DEC')
    plt.legend(loc='best')
    plt.title('RA and DEC linear fit using Covariance Matrices')


def plot_error_ellipse(ax, mu, cov_matrix, color="b"):
    """
    Based on
    http://stackoverflow.com/questions/17952171/not-sure-how-to-fit-data-with-a-gaussian-python.
    """
    # Compute eigenvalues and associated eigenvectors
    vals, vecs = np.linalg.eigh(cov_matrix)

    # Compute "tilt" of ellipse using first eigenvector
    x, y = vecs[:, 0]
    theta = np.degrees(np.arctan2(y, x))

    # Eigenvalues give length of ellipse along each eigenvector
    w, h = 2 * np.sqrt(vals)
    ellipse = Ellipse(mu, w, h, theta, color=color)  # color="k")
    ellipse.set_clip_box(ax.bbox)
    ellipse.set_alpha(0.2)
    ax.add_artist(ellipse)
    return ax


def generate_parabolic_astrometric_data(correlation_coefficient=0.0, sigma_ra=0.1, sigma_dec=0.1, num_measurements=20, crescendo=False):
    astrometric_data = {}
    num_measurements = num_measurements
    mu_ra, mu_dec = -1, 2
    acc_ra, acc_dec = -0.1, 0.2
    ra0, dec0 = -30, 40
    epoch_start = 0
    epoch_end = 200
    astrometric_data['epoch_delta_t'] = np.linspace(epoch_start, epoch_end, num=num_measurements)
    astrometric_data['dec'] = dec0 + astrometric_data['epoch_delta_t']*mu_dec + \
                              1 / 2 * acc_dec * astrometric_data['epoch_delta_t'] ** 2
    astrometric_data['ra'] = ra0 + astrometric_data['epoch_delta_t']*mu_ra + \
                             1 / 2 * acc_ra * astrometric_data['epoch_delta_t'] ** 2
    cc = correlation_coefficient
    astrometric_data['covariance_matrix'] = np.zeros((num_measurements, 2, 2))
    astrometric_data['covariance_matrix'][:] = np.array([[sigma_ra**2, sigma_ra*sigma_dec*cc],
                                                       [sigma_ra*sigma_dec*cc, sigma_dec**2]])
    if crescendo:
        astrometric_data['covariance_matrix'][:, 0, 0] *= np.linspace(1/10, 4, num=num_measurements)
        astrometric_data['covariance_matrix'][:, 1, 1] *= np.linspace(4, 1/10, num=num_measurements)
    for i in range(len(astrometric_data)):
        astrometric_data['inverse_covariance_matrix'][i] = np.linalg.pinv(astrometric_data['covariance_matrix'][i])
    astrometric_data['linear_solution'] = np.array([ra0, dec0, mu_ra, mu_dec])
    return astrometric_data


if __name__ == "__main__":

    data = HipparcosRereductionData()
    data.parse(intermediate_data_directory='/home/mbrandt21/Downloads/Hip2/IntermediateData/resrec',
               star_hip_id='27321')
    scan_angles = data.scan_angle.truncate(after=20)
    multiplier = 20
    covariances = calculate_covariance_matrices(scan_angles, cross_scan_along_scan_var_ratio=multiplier)
    f, ax = plt.subplots()
    for i in range(len(scan_angles)):
        center = data.julian_day_epoch()[i]
        ax = plot_error_ellipse(ax, mu=(center, 0), cov_matrix=covariances[i])
        ax.set_xlim((np.min(data.julian_day_epoch()), np.max(data.julian_day_epoch())))
        ax.set_ylim((-multiplier, multiplier))
        angle = scan_angles.values.flatten()[i]
        ax.plot([center, center -np.sin(angle)], [0, np.cos(angle)], 'k')
        ax.set_title('along scan angle {0} degrees east from the northern equatorial pole'.format(angle*180/np.pi))
    plt.axis('equal')

    data = HipparcosRereductionData()
    data.parse(intermediate_data_directory='/home/mbrandt21/Downloads/Hip2/IntermediateData/resrec',
               star_hip_id='49699')
    scan_angles = data.scan_angle
    astrometric_data = generate_parabolic_astrometric_data(correlation_coefficient=0, sigma_ra=5E2,
                                                           sigma_dec=5E2, num_measurements=len(scan_angles))
    astrometric_data['covariance_matrix'] = calculate_covariance_matrices(data.scan_angle, cross_scan_along_scan_var_ratio=10)
    astrometric_data['epoch_delta_t'] = data.julian_day_epoch()
    plot_fitting_to_astrometric_data(astrometric_data)

    plt.show()
