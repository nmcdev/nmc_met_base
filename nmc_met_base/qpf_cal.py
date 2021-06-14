# _*_ coding: utf-8 _*_

# Copyright (c) 2021 NMC Developers.
# Distributed under the terms of the GPL V3 License.

import numpy as np
import numba as nb


def cum_freq(data, bins=None, norm=True):
    """Calculate the cumulative frequency distribution.

    refer to:
        Zhu, Y. and Y. Luo, 2015: Precipitation Calibration Based on the
        Frequency-Matching Method. Wea. Forecasting, 30, 1109鈥?124,
        https://doi.org/10.1175/WAF-D-13-00049.1

    Arguments:
        data {numpy nd-array} -- numpy nd-array. The missing value is allowed.

    Keyword Arguments:
        bins {numpy array} -- the bin-edegs used to calculate CFD.
        norm {bool} -- normalize the distribution (default: {True})
    """

    # set the bin edges
    if bins is None:
        bins = np.concatenate(([0.1, 1], np.arange(2, 10, 1),
                               np.arange(10, 152, 2)))

    # mask the missing values and change negative to zero
    data = data[np.isfinite(data)]

    # calculate the cumulative frequency distribution
    cfd_array = np.full(bins.size, np.nan)
    for ib, b in enumerate(bins):
        cfd_array[ib] = np.count_nonzero(data <= b) * 1.0

    # normalize the distribution
    if norm:
        cfd_array /= data.size

    # return the bin edges and CFD
    return cfd_array, bins


def cfd_match(data, cfd_obs, cfd_fcst, bins):
    """
    Perform the frequency-matching methods.

    refer to:
        Zhu, Y. and Y. Luo, 2015: Precipitation Calibration Based on the
        Frequency-Matching Method. Wea. Forecasting, 30, 1109鈥?124,
        https://doi.org/10.1175/WAF-D-13-00049.1

    Args:
        data (np.array): forecast data to be calibrated.
        cdf_obs (np.array): 1D array, the cumulative frequency distribution of observations.
        cdf_fcst (np.array): 1D array, the cumulative frequency distribution of forecasts.
        bins (np.array): 1D array, bin edges for CFD.

    Returns:
        np.array: calibrated forecasts.
    """

    # construct interpolation
    data = np.ravel(data)
    data_cfd = np.interp(data, bins, cfd_fcst)
    data_cal = np.interp(data_cfd, cfd_obs, bins)

    # deal with out-range points, just keep the values.
    data_cal[data > np.max(bins)] = data[data > np.max(bins)]
    data_cal[data < np.min(bins)] = data[data < np.min(bins)]
    return data_cal


@nb.njit()
def quantile_mapping_stencil(pred, cdf_pred, cdf_true, land_mask, rad=1):
    """
    Quantile mapping with stencil grid points.

    Scheuerer, M. and Hamill, T.M., 2015. Statistical postprocessing of 
    ensemble precipitation forecasts by fitting censored, shifted gamma distributions. 
    Monthly Weather Review, 143(11), pp.4578-4596.
    
    Hamill, T.M., Engle, E., Myrick, D., Peroutka, M., Finan, C. and Scheuerer, M., 2017. 
    The US National Blend of Models for statistical postprocessing of probability of precipitation 
    and deterministic precipitation amount. Monthly Weather Review, 145(9), pp.3441-3463.

    refer to:
    https://github.com/yingkaisha/fcstpp/blob/main/fcstpp/gridpp.py

    Args:
        pred (np.array): ensemble forecasts. `shape=(ensemb_members, gridx, gridy)`.
        cdf_pred (np.array): quantile values of the forecast. `shape=(quantile_bins, gridx, gridy)`
        cdf_true (np.array): the same as `cdf_pred` for the analyzed condition.
        land_mask (np.array): boolean arrays with True for focused grid points (i.e., True for land grid point).
                              `shape=(gridx, gridy)`.
        rad (int, optional): grid point radius of the stencil. `rad=1` means 3-by-3 stencils.

    Return:
        out: quantile mapped and enlarged forecast. `shape=(ensemble_members, folds, gridx, gridy)`
             e.g., 3-by-3 stencil yields nine-fold more mapped outputs.
    """

    EN, Nx, Ny = pred.shape
    N_fold = (2*rad+1)**2
    out = np.empty((EN, N_fold, Nx, Ny,))
    out[...] = np.nan
    
    for i in range(Nx):
        for j in range(Ny):
            # loop over grid points
            if land_mask[i, j]:
                min_x = np.max([i-rad, 0])
                max_x = np.min([i+rad, Nx-1])
                min_y = np.max([j-rad, 0])
                max_y = np.min([j+rad, Ny-1])

                count = 0
                for ix in range(min_x, max_x+1):
                    for iy in range(min_y, max_y+1):
                        if land_mask[ix, iy]:
                            for en in range(EN):
                                out[en, count, i, j] = np.interp(
                                    pred[en, i, j], cdf_pred[:, ix, iy], cdf_true[:, ix, iy])
                            count += 1
    return out