# _*_ coding: utf-8 _*_

# Copyright (c) 2021 NMC Developers.
# Distributed under the terms of the GPL V3 License.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


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

def quality_visual(ax=None):
    """Visualizing Multiple Measures of Forecast Quality
    refer to:
      Roebber, P.J., 2009: Visualizing Multiple Measures of Forecast Quality.
      Wea. Forecasting, 24, 601-608, https://doi.org/10.1175/2008WAF2222159.1

    Keyword Arguments:
        ax {matplotlib.axes} -- matplotlib axes instance (default: {None})
    """

    # set up figure
    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)

    # modifying rc settings
    plt.rc('font', size=20)
    plt.rc('axes', linewidth=3)
    plt.rc('xtick.major', size=14, width=3)
    plt.rc('ytick.major', size=14, width=3)
    plt.rc('xtick.minor', size=8, width=1)
    plt.rc('ytick.minor', size=8, width=1)

    # define SR
    SR = np.arange(0.0, 1.01, 0.01)

    # draw BS lines
    BSs = [0.3, 0.5, 0.8, 1.0, 1.3, 1.5, 2.0, 3.0, 5.0, 10.0]
    for bs in BSs:
        ax.plot(SR, bs*SR, color="black",
                linewidth=1, linestyle='--', label="BIAS")
        if bs < 1.0:
            ax.text(1.02, bs, str(bs), fontsize=16)
        elif bs > 1.0:
            ax.text(1.0-(bs-1)/bs, 1.02, str(bs), ha='center', fontsize=16)
        else:
            ax.text(1.02, 1.02, '1.0', fontsize=16)

    # draw CSI line
    CSIs = np.arange(0.1, 1.0, 0.1)
    x_pos = [0.5, 0.576, 0.652, 0.728, 0.804, 0.88, 0.88, 0.93, 0.97]
    for i, csi in enumerate(CSIs):
        pod = SR/(SR/csi + SR - 1.0)
        pod[pod < 0] = np.nan
        ax.plot(SR, pod, color="black", linewidth=1, label="CSI")
        ax.text(x_pos[i], x_pos[i]/(x_pos[i]/csi + x_pos[i] - 1.0),
                "{:.1f}".format(csi), backgroundcolor="white", fontsize=12)

    # set axis style
    majorLocator = MultipleLocator(0.1)
    minorLocator = MultipleLocator(0.02)
    ax.xaxis.set_major_locator(majorLocator)
    ax.xaxis.set_minor_locator(minorLocator)
    ax.yaxis.set_major_locator(majorLocator)
    ax.yaxis.set_minor_locator(minorLocator)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_xlabel('Success Ratio (1-FAR)')
    ax.set_ylabel('Probability of Detection (POD)')

    # restore default rc settings
    plt.rcdefaults()
    return ax