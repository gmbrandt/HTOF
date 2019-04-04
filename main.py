#!/usr/bin/env python
import numpy as np
import pandas as pd
import os
import glob
import datetime
from astropy.time import Time

import abc

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


class IntermediateDataParser(object):
    """
    Base class for parsing Hip1 and Hip2 data. self.epoch, self.covariance_matrix and self.scan_angle are saved
    as panda dataframes. use .values (e.g. self.epoch.values) to call the ndarray version.
    """
    def __init__(self, scan_angle=None, epoch=None, residuals=None):
        self.scan_angle = scan_angle
        self.epoch = epoch
        self.residuals = residuals

    @staticmethod
    def read_intermediate_data_file(star_hip_id, intermediate_data_directory, skiprows, header, sep):
        filepath = os.path.join(os.path.join(intermediate_data_directory, '**/'), '*' + star_hip_id + '*')
        filepath_list = glob.glob(filepath, recursive=True)
        if len(filepath_list) > 1:
            raise ValueError('More than one input file with hip id {0} found'.format(star_hip_id))
        data = pd.read_csv(filepath_list[0], sep=sep, skiprows=skiprows, header=header, engine='python')
        return data

    @abc.abstractmethod
    def parse(self, star_id, intermediate_data_parent_directory, **kwargs):
        pass

    @staticmethod
    def convert_hip_style_epochs_to_julian_day(epochs, half_day_correction=True):
        jd_epochs = []
        for epoch in epochs.values:
            epoch_year = int(epoch)
            fraction = epoch - int(epoch)
            utc_time = datetime.datetime(year=epoch_year, month=1, day=1) + datetime.timedelta(days=365.25) * fraction
            if half_day_correction:
                utc_time += datetime.timedelta(days=0.5)
            jd_epochs.append(Time(utc_time).jd)
        return pd.DataFrame(data=np.array(jd_epochs), index=epochs.index)


class HipparcosOriginalData(IntermediateDataParser):
    def __init__(self, scan_angle=None, epoch=None, residuals=None):
        super(HipparcosOriginalData, self).__init__(scan_angle=scan_angle,
                                                    epoch=epoch, residuals=residuals)

    def parse(self, star_hip_id, intermediate_data_directory, data_choice='NDAC', convert_to_jd=True):
        """
        :param star_hip_id: a string which is just the number for the HIP ID.
        :param intermediate_data_directory: the path (string) to the place where the intermediate data is stored, e.g.
                Hip2/IntermediateData/resrec
                note you have to specify the file resrec or absrec. We use the residual records, so specify resrec.
        :param data_choice: 'FAST' or 'NDAC'. This slightly affects the scan angles. This mostly affects
        the residuals which are not used.
        """
        if (data_choice is not 'NDAC') and (data_choice is not 'FAST'):
            raise ValueError('data choice has to be either NDAC or FAST')
        data = self.read_intermediate_data_file(star_hip_id, intermediate_data_directory,
                                                skiprows=0, header='infer', sep='\s*\|\s*')
        # select either the data from the NDAC or the FAST consortium.
        data = data[data['IA2'] == data_choice[0]]
        # compute scan angles and observations epochs according to van Leeuwen & Evans 1997, eq. 11 & 12.
        self.scan_angle = np.arctan2(data['IA3'], data['IA4'])  # unit radians
        self.epoch = data['IA6'] / data['IA3'] + 1991.25
        if convert_to_jd:
            self.epoch = self.convert_hip_style_epochs_to_julian_day(self.epoch)
        self.residuals = data['IA8']  # unit milli-arcseconds (mas)


class HipparcosRereductionData(IntermediateDataParser):
    def __init__(self, scan_angle=None, epoch=None, residuals=None):
        super(HipparcosRereductionData, self).__init__(scan_angle=scan_angle,
                                               epoch=epoch, residuals=residuals)

    def parse(self, star_hip_id, intermediate_data_directory, convert_to_jd=True):
        data = self.read_intermediate_data_file(star_hip_id, intermediate_data_directory,
                                                skiprows=1, header=None, sep='\s+')
        # compute scan angles and observations epochs from van Leeuwen 2007, table G.8
        # see also Figure 2.1, section 2.5.1, and section 4.1.2
        self.scan_angle = np.arctan2(data[3], data[4])  # data[3] = cos(psi), data[4] = sin(psi)
        self.epoch = data[1] + 1991.25
        if convert_to_jd:
            self.epoch = self.convert_hip_style_epochs_to_julian_day(self.epoch)
        self.residuals = data[5]  # unit milli-arcseconds (mas)


class GaiaData(IntermediateDataParser):
    def __init__(self, scan_angle=None, epoch=None, residuals=None):
        super(GaiaData, self).__init__(scan_angle=scan_angle,
                                       epoch=epoch, residuals=residuals)

    def parse(self, star_hip_id, intermediate_data_directory, **kwargs):
        data = self.read_intermediate_data_file(star_hip_id, intermediate_data_directory,
                                                skiprows=0, header='infer', sep='\s*,\s*')
        self.epoch = data['ObservationTimeAtBarycentre[BarycentricJulianDateInTCB]']
        self.scan_angle = data['scanAngle[rad]']


class AstrometricFitter(object):
    """
    :param covariance_matrices: ndarray of length epoch times with 2x2 covariance matrices for each epoch
    :param epoch_times: the times for each epoch.
    """
    def __init__(self, covariance_matrices, epoch_times,
                 astrometric_chi_squared_matrices=None, astrometric_solution_vector_components=None):
        self.covariance_matrices = covariance_matrices
        self.epoch_times = epoch_times
        if astrometric_solution_vector_components is None:
            self.astrometric_solution_vector_components = self._init_astrometric_solution_vectors()
        if astrometric_chi_squared_matrices is None:
            self.astrometric_chi_squared_matrices = self._init_astrometric_chi_squared_matrices()

    def fit_line(self, ra_vs_epoch, dec_vs_epoch):
        """
        :param ra_vs_epoch: 1d array of right ascension, ordered the same as the covariance matrices and epochs.
        :param dec_vs_epoch: 1d array of declination, ordered the same as the covariance matrices and epochs.
        :return:
        """
        return np.linalg.solve(self._chi2_matrix(), self._chi2_vector(ra_vs_epoch=ra_vs_epoch,
                                                                      dec_vs_epoch=dec_vs_epoch))

    def _chi2_matrix(self):
        return np.sum(self.astrometric_chi_squared_matrices, axis=0)

    def _chi2_vector(self, ra_vs_epoch, dec_vs_epoch):
        ra_solution_vecs = self.astrometric_solution_vector_components['ra']
        dec_solution_vecs = self.astrometric_solution_vector_components['dec']
        # sum together the individual solution vectors for each epoch
        return np.dot(ra_vs_epoch, ra_solution_vecs) + np.dot(dec_vs_epoch, dec_solution_vecs)

    def _init_astrometric_solution_vectors(self):
        num_epochs = len(self.epoch_times)
        astrometric_solution_vector_components = {'ra': np.zeros((num_epochs, 4)),
                                                  'dec': np.zeros((num_epochs, 4))}
        for epoch in range(num_epochs):
            pseudo_inverse_cov_matrix = np.linalg.pinv(self.covariance_matrices[epoch])
            d, b, c, a = unpack_elements_of_matrix(pseudo_inverse_cov_matrix)
            b, c = -b, -c
            epoch_time = self.epoch_times[epoch]
            ra_vec, dec_vec = np.zeros(4).astype(np.float64), np.zeros(4).astype(np.float64)
            ra_vec[0] = -(-2 * d * epoch_time)
            ra_vec[1] = -((b + c) * epoch_time)
            ra_vec[2] = -(-2 * d)
            ra_vec[3] = -(b + c)

            dec_vec[0] = -((b + c) * epoch_time)
            dec_vec[1] = -(- 2 * a * epoch_time)
            dec_vec[2] = -(b + c)
            dec_vec[3] = -(- 2 * a)

            astrometric_solution_vector_components['ra'][epoch] = ra_vec
            astrometric_solution_vector_components['dec'][epoch] = dec_vec
        return astrometric_solution_vector_components

    def _init_astrometric_chi_squared_matrices(self):
        num_epochs = len(self.epoch_times)
        astrometric_chi_squared_matrices = np.zeros((num_epochs, 4, 4))
        for epoch in range(num_epochs):
            pseudo_inverse_cov_matrix = np.linalg.pinv(self.covariance_matrices[epoch])
            d, b, c, a = unpack_elements_of_matrix(pseudo_inverse_cov_matrix)
            b, c = -b, -c
            epoch_time = self.epoch_times[epoch]

            A = np.zeros((4, 4))

            A[:, 0] = np.array([2 * d * epoch_time,
                                (-b - c) * epoch_time,
                                2 * d,
                                (-b - c)])
            A[:, 1] = np.array([(-b - c) * epoch_time,
                                2 * a * epoch_time,
                                (-b - c),
                                2 * a])
            A[:, 2] = np.array([2 * d * epoch_time ** 2,
                                (-b - c) * epoch_time ** 2,
                                2 * d * epoch_time,
                                (-b - c) * epoch_time])
            A[:, 3] = np.array([(-b - c) * epoch_time ** 2,
                                2 * a * epoch_time ** 2,
                                (-b - c) * epoch_time,
                                2 * a * epoch_time])

            astrometric_chi_squared_matrices[epoch] = A
        return astrometric_chi_squared_matrices


def calculate_covariance_matrices(scan_angles, var_along_scan=1.0, var_cross_scan=1E5):
    # change to be ratio of cross to along scan. Then use np pseudo inverse. keep default ratio 1E5
    """
    :param scan_angles: pandas DataFrame with scan angles, e.g. as-is from the data parsers. scan_angles.values is a
                        numpy array with the scan angles
    :param var_along_scan: variance along the scan direction
    :param var_cross_scan: variance in the direction perpendicular to the scan direction
    :return An ndarray with shape (len(scan_angles), 2, 2), e.g. an array of covariance matrices in the same order
    as the scan angles
    """
    covariance_matrices = []
    cov_matrix_in_scan_basis = np.array([[var_cross_scan, 0], [0, var_along_scan]])
    # we define the along scan to be 'y' in the scan basis.
    for theta in scan_angles.values.flatten():
        # for Hipparcos, theta is measured against north, specifically east of the north equatorial pole
        c, s = np.cos(theta), np.sin(theta)
        Rccw = np.array([[c, -s], [s, c]])
        cov_matrix_in_ra_dec_basis = np.matmul(np.matmul(Rccw, cov_matrix_in_scan_basis), Rccw.T)
        covariance_matrices.append(cov_matrix_in_ra_dec_basis)
    return np.array(covariance_matrices)


def unpack_elements_of_matrix(matrix):
    return matrix.flatten()


def generate_parabolic_astrometric_data(correlation_coefficient=0.0, sigma_ra=0.1, sigma_dec=0.1, crescendo=False):
    astrometric_data = {}
    num_measurements = 20
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
    astrometric_data['solution_vector'] = np.array([ra0, dec0, mu_ra, mu_dec])
    return astrometric_data


def plot_fitting_to_curved_astrometric_data(crescendo=False):
    astrometric_data = generate_parabolic_astrometric_data(correlation_coefficient=0, sigma_ra=5E2, sigma_dec=5E2, crescendo=crescendo)
    # solving
    fitter = AstrometricFitter(covariance_matrices=astrometric_data['covariance_matrix'],
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


def plot_error_ellipse(mu, cov_matrix, color="b"):
    """
    Based on
    http://stackoverflow.com/questions/17952171/not-sure-how-to-fit-data-with-a-gaussian-python.
    """
    f, ax = plt.subplots(figsize=(3, 3))
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
    """
    scan_angles = pd.DataFrame(data=np.linspace(0, 2*np.pi, 6))
    covariances = calculate_covariance_matrices(scan_angles, var_along_scan=0.1, var_cross_scan=0.8)
    for i in range(len(scan_angles)):
        ax = plot_error_ellipse(mu=(0, 0), cov_matrix=covariances[i])
        ax.set_xlim((-1, 1))
        ax.set_ylim((-1, 1))
        angle = scan_angles.values.flatten()[i]
        ax.plot([0, -np.sin(angle)], [0, np.cos(angle)], 'k')
        ax.set_title('along scan angle {0} degrees east from the northern equatorial pole'.format(angle*180/np.pi))
    plt.show()
    """

    plot_fitting_to_curved_astrometric_data(crescendo=False)
    plot_fitting_to_curved_astrometric_data(crescendo=True)
    plt.show()
