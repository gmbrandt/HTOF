import numpy as np
import pandas as pd


def merge_consortia(data):
    """
    :param data: pandas.DataFrame. The intermediate astrometric data for a Hipparcos 1 source.
    :return: merged_data: pandas.DataFrame.
    The input data with merged residuals, errors, sin(scan_angle), cos(scan_angle), epochs etc. The
    residuals and errors (along-scan errors) are merged according the Hipparcos and Tycho Catalogues Vol 3
    Section "Astrometric Catalogue Merging" (page 377).

    This function takes the merged sin of the scan angle and epochs etc to be the weighted mean of
    those from both consortia (weighted by the covariance between observations, see `More' below).

    Before merging, observations that were rejected by the Hipparcos team (flagged with lowercase f or n in
    the consortia (IA2) column) are removed and not used for merging. See the description of Field IA2 in
    the Hipparcos and Tycho Catalogues Vol 1, page 260.

    More: The intermediate astrometric data provides the correlation coefficients between the values
    for the residuals and errors obtained by the NDAC and FAST consortia. One can then rebuild the covariance
    between the measurements of those two consortia (i.e. treating their results as correlated measurements of
    some unknown value -- which is the merged value we are after). Given the covariance matrix for one orbit, C
    from equation 17.11 of the Hipparcos and Tycho Catalogues Vol 3 (page 377):
            C = np.asarray([[errF[i]**2, errN[i]*errF[i]*corr[i]],
                        [errN[i]*errF[i]*corr[i], errN[i]**2]])
    Where errN[i] is the NDAC along-scan error at the ith epoch, and errF is the FAST error, and corr[i] is the
    correlation between the two measurements (column IA10). The merged residual is then the value which minimizes
    the chisquared = (residual - merged_residual) C^(-1) (residual - merged_residual)
    with matrix dot products implied and residual = the vector of residuals = [NDAC residual, FAST residual]
    The merged error is 1/sum(C^(-1))**0.5
    Where C^(-1) is the matrix inverse of the covariance matrix C.
    """
    # exclude observations that were rejected for the merged solution (those with n, f instead of N, F)
    data.drop(np.argwhere(np.logical_or((data['IA2'] == 'n').to_numpy(),
                                        (data['IA2'] == 'f').to_numpy())).flatten(), inplace=True)
    # We transform to Numpy arrays because accessing and editing panda arrays is slower by factors of tens.
    cols_to_merge = ['A1', 'IA3', 'IA4', 'IA5', 'IA6', 'IA7', 'IA8', 'IA9', 'IA10']
    data_asarray = data[cols_to_merge].to_numpy()
    # merge data
    merged_data = _merge_orbits(data_asarray)
    merged_data = pd.DataFrame(merged_data, columns=cols_to_merge)
    merged_data['IA2'] = 'M'
    return merged_data


def _merge_orbits(data):
    """
    :param data: ndarray. Intermediate Data for a single Hipparcos 1 orbit.
                data must be a 2D array such that
                data[:, -1] = column IA10  (correlation coeff). Note data[:, -1] is the last column of data
                data[:, -2] = column IA9  (formal error along scan)
                data[:, -3] = column IA8  (residual along scan)
                data[:, 0] = column A1  (the orbit number)
    :return merged_orbit: ndarray. Merged intermediate astrometric data for a single orbit.
    Will have shape (m, N), where m is the number of unique orbits in data, i.e. m = len(np.unique(data[:, 0]))
    """
    # find single consortia orbits (either NDAC or FAST but not both)
    orbs, indices, counts = np.unique(data[:, 0], return_index=True, return_counts=True)
    # omit merging orbits with only a single consortium
    single_orbits = data[indices[counts == 1]]
    data = np.delete(data, indices[counts == 1], axis=0)
    # generate weights for merging
    err1, err2, corr = data[::2, -2], data[1::2, -2], data[::2, -1]
    icov = np.linalg.pinv(np.array([[err1 ** 2, err2 * err1 * corr],
                                    [err2 * err1 * corr, err2 ** 2]]).T.reshape((-1, 2, 2)))
    # constructing the weights via the best linear estimator method (which seeks to minimize the variance).
    weights = np.array([np.sum(np.dot([1, 0], icov), axis=1), np.sum(np.dot([0, 1], icov), axis=1)])
    weights /= np.sum(np.sum(icov, axis=-1), axis=-1)  # normalize weights by sum of elements of each icov matrix.
    # merge data
    merged_data = weights[0].reshape(-1, 1) * data[::2] + weights[1].reshape(-1, 1) * data[1::2]
    # evaluate variances on the BLUE estimator x'. Var(x') = w0^2*Var(x0) + w1^2*Var(x1) + 2 * w1 * w0 * Cov(x1, x2)
    # where w1, w2 are the weights and x' is the merged result computed above.
    merged_data[:, -2] = weights[0] ** 2 * err1 ** 2 + weights[1] ** 2 * err2 ** 2 + \
                         2 * corr * err1 * err2 * weights[0] * weights[1]
    # convert the variances to standard deviations
    merged_data[:, -2] = np.sqrt(merged_data[:, -2])
    # stack the merged_data back with the orbits with a single consortia (that could therefore not be merged)
    merged_data = np.vstack((merged_data, single_orbits))
    return merged_data[np.argsort(merged_data[:, 0])]


def safe_concatenate(a, b):
    if a is None and b is None:
        return None
    if a is None and b is not None:
        return b
    if a is not None and b is None:
        return a
    return np.concatenate([a, b])
