#!/usr/bin/env python
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from htof.fit import AstrometricFitter
from htof.parse import HipparcosRereductionData, calculate_covariance_matrices

"""
Driver script which currently just makes plots of interest.
"""


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


if __name__ == "__main__":
    plot_diagnostic_data = False
    plot_fake_orbit_fit = True
    if plot_diagnostic_data:
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
        plt.show()
    if plot_fake_orbit_fit:
        data = HipparcosRereductionData()
        data.parse(intermediate_data_directory='/home/mbrandt21/Downloads/Hip2/IntermediateData/resrec',
                   star_hip_id='27321')
        data.calculate_inverse_covariance_matrices()
        
        assert True
