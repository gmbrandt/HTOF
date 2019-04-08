Example python code for parsing the intermediate data from Gaia or Hip, and then fitting a line to an orbit with
arbitrary RA and DEC vs epoch given the Hip or Gaia covariance matrices (from the intermediate data) at those epochs.


from htof.main import DataParser # really this will be one of three parsers, e.g. GaiaData
from htof.main import AstrometricFitter


# things we do once at the start of the code:
data = DataParser()
data.parse(star_hip_id='49699',
           intermediate_data_directory='path/to/intermediate_data_parent_folder')
data.calculate_inverse_covariance_matrices(cross_scan_along_scan_var_ratio=1E5)
fitter = AstrometricFitter(inverse_covariance_matrices=data.inverse_covariance_matrix,
                           epoch_times=data.julian_day_epoch())

#  the following lines we do in the mcmc loops:
solution_vector = fitter.fit_line(ra_vs_epoch=ra_from_orbit_fitting_mcmc_trial,
                                  dec_vs_epoch=dec_from_orbit_fitting_mcmc_trial)

ra0, dec0, mu_ra, mu_dec = solution_vector
#  solution vector is the parametric solution to the line of best fit for the data. Where Ra(epoch) = ra0 + epoch * mu_ra, and Dec(epoch) = dec0 + epoch * mu_dec