# _*_ coding: utf-8 _*_

# Copyright (c) 2021 NMC Developers.
# Distributed under the terms of the GPL V3 License.

import numpy as np
import numba as nb


def prob_matched_ens_mean(values):
    """
    Perform probability-matched ensemble mean (PM).

    The QPFs rarely predict the rain pattern in exactly the same place. 
    When combining multiple rain fields to produce a deterministic rain
    forecast, the ensemble mean is likely to predict the best location 
    of the rain center, but the averaging process "smears" the rain rates
    so that the maximum rainfall is reduced and area of light rain is 
    artificially enhanced (see plot below). However, the rain rate frequency
    distribution in the original ensemble (collating all the rain rates from
    all the individual members) is usually closer to the observed rain rate 
    frequency distribution. Probability matching transforms the rain rate 
    distribution in the ensemble mean rain field to look like that from the 
    complete ensemble

    refer to:
    Ebert, E. E. (2001). "Ability of a poor man's ensemble to predict the probability 
      and distribution of precipitation." monthly weather review 129(10): 2461-2480.

    Args:
        values (np.array): ensemble forecasts, shape=(ensemble_members, lat, lon)

    Returns:
        np.array: probability-matched ensemble mean array, shape=(lat, lon)
    """

    # get dimensions
    nmem, nlat, nlon = values.shape

    # calculate ensemble mean
    ens_mean = values.mean(axis=0).flatten()

    # construct result variables
    ens_pm_mean = np.zeros(nlat * nlon)

    # calculate probability-matched ensemble mean
    values = values.flatten()
    ens_pm_mean[np.argsort(ens_mean)] = values[(np.argsort(values))[round(nmem/2.0-1)::nmem]]

    # restore shape
    values.shape = (nmem, nlat, nlon)
    ens_pm_mean.shape = (nlat, nlon)

    # return
    return ens_pm_mean


@nb.njit()
def schaake_shuffle(fcst, traj):
    """
    Perform the schaake shuffle method with history observation.

    Clark, M., Gangopadhyay, S., Hay, L., Rajagopalan, B. and Wilby, R., 2004. 
    The Schaake shuffle: A method for reconstructing space–time variability in forecasted 
    precipitation and temperature fields. Journal of Hydrometeorology, 5(1), pp.243-262.

    refer to:
      https://github.com/yingkaisha/fcstpp/blob/main/fcstpp/gridpp.py


    Args:
        fcst (np.array): ensemble forecasts, shape=(ensemble_members, lead_time, grid_points)
        traj (np.array): trajectories, shape=(history_time, lead_time, grid_points)
                         number of trajectories and ensemble memebers must match: `history_time == ensemb_members`.
                         number of forecast lead time must match: `fcst.shape == traj.shape`.
                         这里traj是从与预报时效lead_time同期的历史观测数据中(如[lead_time-7, lead_time+7]), 随机地选择
                         与ensemble_members数量相当的历史观测数据, 该历史观测携带了空间分布信息.

    Return:
        output: shuffled ensemble forecast, shape=(ensemb_members, lead_time, grid_points)
    """

    num_traj, N_lead, N_grids = traj.shape
    
    output = np.empty((num_traj, N_lead, N_grids))
    
    for l in range(N_lead):
        for n in range(N_grids):
            
            temp_traj = traj[:, l, n]
            temp_fcst = fcst[:, l, n]
            
            reverse_b_func = np.searchsorted(np.sort(temp_traj), temp_traj)
            
            output[:, l, n] = np.sort(temp_fcst)[reverse_b_func]
    return output


@nb.njit()
def schaake_shuffle_var(fcst, traj):
    """
    Perform the schaake shuffle method with history observation.

    Clark, M., Gangopadhyay, S., Hay, L., Rajagopalan, B. and Wilby, R., 2004. 
    The Schaake shuffle: A method for reconstructing space–time variability in forecasted 
    precipitation and temperature fields. Journal of Hydrometeorology, 5(1), pp.243-262.

    refer to:
      https://github.com/yingkaisha/fcstpp/blob/main/fcstpp/gridpp.py


    Args:
        fcst (np.array): ensemble forecasts, shape=(ensemb_members, grid_points, variables)
                         a three-dimensional matrix of ensemble forecasts
        traj (np.array): trajectories, shape=(history_time, grid_points, variables)
                         number of trajectories and ensemble memebers must match: `history_time == ensemb_members`.
                         number of forecast lead time must match: `fcst.shape == traj.shape`.
                         To correspond to the matrix fcst, we construct an identicallysized 
                           three-dimensional matrix traj derived from historical station
                           observations of the respective variables, The dates used to populate
                           the matrix traj are selected so as to lie within seven days before and
                           after the forecast date (dates can be pulled from all years in the 
                           historical record, except for the year of the forecast). Populating 
                           the traj matrix in this way means that data from the same date is
                           used for all grid points (j) and variables (k).

    Return:
        output: shuffled ensemble forecast, shape=(ensemb_members, grid_points, variables)
    """

    num_traj, N_grids, N_vars = traj.shape
    
    output = np.empty((num_traj, N_grids, N_vars))
    
    for l in range(N_grids):
        for n in range(N_vars):
            
            temp_traj = traj[:, l, n]
            temp_fcst = fcst[:, l, n]
            
            reverse_b_func = np.searchsorted(np.sort(temp_traj), temp_traj)
            
            output[:, l, n] = np.sort(temp_fcst)[reverse_b_func]
    return output


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


@nb.njit()
def bootstrap_fill(data, expand_dim, land_mask, fillval=np.nan):
    """
    Fill values with bootstrapped aggregation.
    该函数将集合预报成员维度采用bootstrap方法进行扩充.

    Args:
        data (np.array): a four dimensional array. `shape=(time, ensemble_members, gridx, gridy)`.
        expand_dim (integer): dimensions of `ensemble_members` that need to be filled. 
                              If `expand_dim` == `ensemble_members` then the bootstraping is not applied.
        land_mask (boolean): boolean arrays with True for focused grid points (i.e., True for land grid point).
                             `shape=(gridx, gridy)`.
        fillval (np.type, optional): fill values of the out-of-mask grid points. Defaults to np.nan.

    Return:
        out: bootstrapped data. `shape=(time, expand_dim, gridx, gridy)`
    """

    N_days, _, Nx, Ny = data.shape
    out = np.empty((N_days, expand_dim, Nx, Ny))
    out[...] = np.nan
    
    for day in range(N_days):
        for ix in range(Nx):
            for iy in range(Ny):
                if land_mask[ix, iy]:
                    data_sub = data[day, :, ix, iy]
                    flag_nonnan = np.logical_not(np.isnan(data_sub))
                    temp_ = data_sub[flag_nonnan]
                    L = len(temp_)
                    
                    if L == 0:
                        out[day, :, ix, iy] = fillval
                    elif L == expand_dim:
                        out[day, :, ix, iy] = temp_
                    else:
                        ind_bagging = np.random.choice(L, size=expand_dim, replace=True)
                        out[day, :, ix, iy] = temp_[ind_bagging]
    return out
