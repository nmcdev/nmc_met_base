# _*_ coding: utf-8 _*_

# Copyright (c) 2021 NMC Developers.
# Distributed under the terms of the GPL V3 License.

import numpy as np


def threat_score(in_obs, in_fcst, threshold,
                 matching=False, return_list=False):
    """
    calculate threat score and bias score.

    :param in_obs: observation or analysis data.
    :param in_fcst: forecast data, the same size as in_obs.
    :param threshold: threshold value.
    :param matching: if True, the obs and fcst will be sorted,
                     and the position bias is ignored.
    :param return_list: if True, return list, not dictionary.
    :return: threat and bias score.
    """

    # check missing value
    obs = in_obs.flatten()
    fcst = in_fcst.flatten()
    miss_index = np.logical_or(np.isnan(obs), np.isnan(fcst))
    if np.all(miss_index):
        return None
    obs = obs[np.logical_not(miss_index)]
    fcst = fcst[np.logical_not(miss_index)]
    if matching:
        obs = np.sort(obs)
        fcst = np.sort(fcst)

    # calculate NA, NB, NC
    o_flag = np.full(obs.size, 0)
    f_flag = np.full(fcst.size, 0)
    index = obs > threshold
    if np.count_nonzero(index) > 0:
        o_flag[index] = 1
    index = fcst > threshold
    if np.count_nonzero(index) > 0:
        f_flag[index] = 1
    NA = np.count_nonzero(
        np.logical_and(o_flag == 1, f_flag == 1))  # hits
    NB = np.count_nonzero(
        np.logical_and(o_flag == 0, f_flag == 1))  # false alarms
    NC = np.count_nonzero(
        np.logical_and(o_flag == 1, f_flag == 0))  # misses

    # threat score
    if NA+NB+NC == 0:
        ts = np.nan
    else:
        ts = NA*1.0/(NA+NB+NC)

    # equitable threat score
    hits_random = (NA+NC)*(NA+NB)*1.0/obs.size
    if NA+NB+NC-hits_random == 0:
        ets = np.nan
    else:
        ets = (NA-hits_random)*1.0/(NA+NB+NC-hits_random)

    # bias score
    if NA+NC == 0:
        bs = np.nan
    else:
        bs = (NA+NB)*1.0/(NA+NC)

    # the probability of detection
    if NA+NC == 0:
        pod = 0
    else:
        pod = NA*1.0/(NA + NC)

    # flase alarm ratio
    if NA+NB == 0:
        far = 1
    else:
        far = NB*1.0/(NA + NB)

    # return score
    if return_list:
        return [NA, NB, NC, hits_random, ts, ets, bs, pod, far]
    else:
        return {'NA': NA, 'NB': NB, 'NC': NC, 'HITS_random': hits_random,
                'TS': ts, 'ETS': ets, 'BS': bs, 'POD': pod, 'FAR': far}

