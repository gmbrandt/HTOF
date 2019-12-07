import numpy as np
import pandas as pd
from copy import copy


def merge_consortia(data):
    """
    :param data: pandas.DataFrame.
    :return: merged_data: pandas.DataFrame.
    The input data with merged residuals, errors, sin(scan_angle), cos(scan_angle), epochs etc. The
    residuals and errors (along-scan errors) are merged according the Hipparcos and Tycho Catalogues Vol 3
    Section "Astrometric Catalogue Merging" (page 377).

    This function takes the merged sin of the scan angle and epochs etc to be the unweighted mean of
    those from both consortia.

    More: The intermediate astrometric data provides the correlation coefficients between the values
    for the residuals and errors obtained by the NDAC and FAST consortia. One can then rebuild the covariance
    between the measurements of those two consortia (i.e. treating their results as correlated measurements of
    some unknown value -- which is the merged value we are after). Given the covariance matrix for one orbit, C:
            C = np.asarray([[errF[i]**2, errN[i]*errF[i]*corr[i]],
                        [errN[i]*errF[i]*corr[i], errN[i]**2]])
    Where errN[i] is the NDAC along-scan error at the ith epoch, and errF is the FAST error, and corr[i] is the
    correlation between the two measurements (column IA10). The merged residual is then the value which minimizes
    the chisquared = (residual - merged_residual) C^(-1) (residual - merged_residual)
    with matrix dot products implied and residual = the vector of residuals = [NDAC residual, FAST residual]
    The merged error is 1/sum(C^(-1))**0.5
    Where C^(-1) is the matrix inverse of the covariance matrix C.

    """
    merged_data = pd.DataFrame(np.zeros((len(np.unique(data['A1'])), len(data.columns)), dtype=float),
                               columns=data.columns)
    for i, orbit in enumerate(np.unique(data['A1'])):
        merged_data.iloc[i] = merge_single_orbit(data[data['A1'] == orbit])
    return pd.DataFrame(merged_data)


def merge_single_orbit(data):
    """
    :param data: pandas.DataFrame. Intermediate Data for a single Hipparcos orbit.
    :return merged_orbit: pandas.DataFrame.
            see docstring of htof.utils.data_utils.merge_consortia for merged_data.
    """
    if len(data) < 2:
        # if only one reduction consortium exists for this orbit, return the reduction.
        return data.iloc[0]
    # for all values except the consortia (which is a string) and the residuals and errors, adopt the mean
    merged_orbit = data.loc[:, data.columns != 'IA2'].mean(axis=0)
    # calculate the covariance matrix with which to calculate the weighted average of the residuals.
    errN, errF, corr = data[data['IA2'] == 'N']['IA9'].iloc[0], data[data['IA2'] == 'F']['IA9'].iloc[0], data.iloc[0]['IA10']
    resN, resF = data[data['IA2'] == 'N']['IA8'].iloc[0], data[data['IA2'] == 'F']['IA8'].iloc[0]
    icov = np.linalg.pinv(np.array([[errF ** 2, errN * errF * corr],
                                    [errN * errF * corr, errN ** 2]]))
    res_arr = np.array([resF, resN])

    # get the residuals which minimize the chisquared, and the associated error.
    merged_orbit['IA8'] = np.sum(np.dot(res_arr, icov)) / np.sum(icov)
    merged_orbit['IA9'] = 1 / np.sum(icov) ** 0.5
    # set consortium to M for MERGED
    merged_orbit['IA2'] = 'M'
    return merged_orbit
