# _*_ coding: utf-8 _*_

# Copyright (c) 2019 NMC Developers.
# Distributed under the terms of the GPL V3 License.

"""
Objective analysis functions.
"""

import numpy as np
from numba import jit
from scipy.interpolate import RegularGridInterpolator
from nmc_met_base.arr import scale_vector
from nmc_met_base.regridding import hinterp


@jit
def stations_avg_distance(x, y, non_uniform=False):
    """
    calculate the station average distance,
    which can be used to define grid space.

    :param x:
    :param y:
    :param non_uniform:
    :return:
    """

    # check input vector
    if len(x) != len(y):
        raise Exception("x length is not equal to y length.")

    # for non-uniform point distribution, use area/npoints
    if non_uniform:
        area = (np.max(x)-np.min(x)) * (np.max(y)-np.min(y))
        return np.sqrt(area) * ((1.0 + np.sqrt(len(x)))/(len(x)-1.0))

    # compute minimum distance
    min_dist = np.full(len(x), 0.0)
    for i in range(len(x)):
        d = np.sqrt((x[i]-x)**2 + (y[i]-y)**2)
        min_dist[i] = np.min(d[d != 0])

    # return average distance
    return np.mean(min_dist)


@jit
def barnes(ix, iy, iz, gs=None, nyx=None, limit=None, radius=None,
           gamma=0.3, kappa=None, npasses=3, non_uniform=True,
           yxout=None, first_guess=None, missing=None,
           zrange=None, nonegative=False):
    """
    Implement barnes objective analysis.
    note: 1、not consider pole area, Near the poles,
             an approximate calculation of the distance along
             a great circle arc should be used.

    references:
    Koch, S., M. desJardins,and P. Kocin, 1983: An Interactive Barnes
      Objective Map Analysis Scheme for Use with Satellite and
      Convectional Data. Journal of Appl. Meteor., 22, 1487-1503.
    Barnes, S.L., 1994a: Applications of the Barnes objective analysis scheme
      Part I: Effects of undersampling, wave position, and station randomness.
      J. Atmos. Oceanic Technol. 11, 1433-1448.
    Barnes, S.L., 1994b: Applications of the Barnes objective analysis scheme
      Part II: Improving derivative estimates. J. Atmos. Oceanic Technol. 11,
      1449-1458.
    Barnes, S.L., 1994c: Applications of the Barnes objective analysis scheme
      Part III: Tuning for minimum error. J. Atmos. Oceanic Technol. 11,
      1459-1479.
    Narkhedkar, S. G., S. K. Sinha and A. K. Mitra (2008): Mesoscale
      objective analysis of daily rainfall with satellite and conventional
      data over Indian summer monsoon region. Geofizika, 25, 159-178.
    http://www.atmos.albany.edu/GEMHELP5.2/OABSFC.html

    :param ix: 1D array, station longitude
    :param iy: 1D array, station latitude
    :param iz: 1D array, station observations.
    :param gs: the result grid spacing, [ys, xs], where xs is the
               horizontal spacing between grid points and ys is
               the vertical spacing. Default is the average
               station spaces.
    :param nyx: the result grid size, [ny, nx], where nx is the output
                grid size in the x direction and ny is in the y
                direction. if not be specified, the size will be
                inferred from gs and bounds. if gs and nxy both are specified,
                nxy will overlap gs.
    :param limit: If present, limit must be a four-element array
                  containing the grid limits in x and y of the output
                  grid: [ymin, ymax, xmin, xmax]. If not specified, the
                  grid limits are set to the extent of x and y.
    :param radius: search radius, [y radius, x radius],
                  [40, 40] is default, with 'kappa' units, where kappa is the
                  scale length, which controls the rate of fall-off of the
                  weighting function. Search radius is the max distance that
                  a station may be from a grid point to be used in the analysis
                  for that point.  The search radius will be set so that
                  stations whose weighting factor would be less than
                  EXP (-SEARCH) will not be used.  SEARCH must be in the range
                  1 - 50, such that stations receiving a weight less than
                  EXP(-search) are considered negligible.  Typically a value
                  of 20 is used, which corresponds to a weight threshold of
                  approximately 2e-9. If a very small value is used, many grid
                  points will not have 3 stations within the search area and
                  will be set to the missing data value.
    :param gamma: is a numerical covergence parameter that controls the
                  difference between the weights on the first and second
                  passes, and lies between 0 and 1. Typically a value between
                  .2 and .3 is used. Gamma=0.3 is default.
                  gamma=0.2, minimum smoothing;
                  gamma=1.0, maximum smoothing.
    :param kappa: the scale length, Koch et al., 1983
    :param npasses: 3 passes is default.
                    Set the number of passes for the Barnes analysis to do
                    4 passes recommended for analysing fields where derivative
                    estimates are important (Ref: Barnes 1994b) 3 passes
                    recommended for all other fields (with gain set to 1.0)
                    (Ref: Barnes 1994c "Two pass Barnes Objective Analysis
                    schemes now in use probably should be replaced
                    by appropriately tuned 3 pass or 4 pass schemes") 2 passes
                    only recommended for "quick look" type analyses.
    :param non_uniform: When the data spacing is severely non-uniform,
                        Koch et al. (1983) suggested the data spacing Dn,
                        which has the following form:
                          sqrt(area){(1+sqrt(N))/(N-1)}
    :param yxout: the latitudes and longitudes on the grid where interpolated
                  values are desired (in degrees), list [yout[:], xout[:]]
    :param first_guess: use a model grid as a first guess field for the
                        analysis, which is a dictionary
                        {'data': [ny, nx], 'x': x[nx], 'y': y[ny]}
    :param missing: if set, remove  missing data.
    :param zrange: if set, z which are in zrange are used.
    :param nonegative: if True, negative number were set to 0.0 for return.
    :return: output grid, which is a dictionary
             {'data': [ny, nx], 'x': x[nx], 'y': y[ny]}
    """

    # keep origin data
    x = ix.copy()
    y = iy.copy()
    z = iz.copy()

    # check z shape
    if (len(x) != len(z)) or (len(y) != len(z)):
        raise Exception('z, x, y dimension mismatch.')

    # remove missing values
    if missing is not None:
        index = z != missing
        z = z[index]
        x = x[index]
        y = y[index]

    # control z value range
    if zrange is not None:
        index = np.logical_and(z >= zrange[0], z <= zrange[1])
        z = z[index]
        x = x[index]
        y = y[index]

    # check observation number
    if len(z) < 3:
        return None

    # domain definitions
    if limit is None:
        limit = [np.min(y), np.max(y), np.min(x), np.max(x)]

    # calculate data spacing
    deltan = stations_avg_distance(x, y, non_uniform=non_uniform)

    # gamma parameters
    if gamma < 0.2:
        gamma = 0.2
    if gamma > 1.0:
        gamma = 1.0

    # kappa parameters (the scale length, Koch et al., 1983)
    if kappa is None:
        kappa = 5.052 * (deltan * 2.0 / np.pi) * (deltan * 2.0 / np.pi)

    # search radius
    if radius is None:
        radius = [40., 40.] * kappa

    # define grid size
    #
    # Peterson and Middleton (1963) stated that a wave whose
    # horizontal wavelength does not exceed at least 2*deltan
    # cannot resolved, since five data points are required to
    # describe a wave. Hence deltax(i.e., gs) not be larger than
    # half of deltan. Since a very small grid resolution may
    # produce an unrealistic noisy derivative and if the
    # derivative fields are to represent only resolvable features,
    # the grid length should not be much smaller than deltan.
    # Thus a constraint that deltan/3 <= deltax <= deltan/2 was
    # imposed by Barnes in his interactive scheme.
    if gs is None:
        if nyx is not None:
            gs = [(limit[1] - limit[0])/nyx[0],
                  (limit[3] - limit[2])/nyx[1]]
        else:
            gs = [deltan, deltan] * 0.4
            nyx = [int((limit[1] - limit[0])/gs[0]),
                   int((limit[3] - limit[2])/gs[1])]
    else:
        nyx = [int((limit[1] - limit[0])/gs[0]),
               int((limit[3] - limit[2])/gs[1])]

    # result grid x and y coordinates
    if yxout is None:
        nyx = [len(yxout[0]), len(yxout[1])]
        gs = [yxout[0][1]-yxout[0][0], yxout[1][1]-yxout[1][0]]
    else:
        yxout = [
            scale_vector(
                np.arange(nyx[0], dtype=np.float), limit[0], limit[1]),
            scale_vector(
                np.arange(nyx[1], dtype=np.float), limit[2], limit[3])]

    # define grid
    yout = yxout[0]
    ny = yout.size
    xout = yxout[1]
    nx = xout.size
    g0 = np.full((ny, nx), np.nan)

    # first pass
    indices = []
    distances = []
    for j in range(ny):
        for i in range(nx):
            # points in search radius
            rd = (
                ((xout[i] - x) / radius[0]) ** 2 +
                ((yout[j] - y) / radius[1]) ** 2)
            if np.count_nonzero(rd <= 1.0) < 1:
                indices.append(None)
                distances.append(None)
                continue

            # extract points in search radius
            index = np.nonzero(rd <= 1.0)
            xx = x[index]
            yy = y[index]
            zz = z[index]

            # compute the square distance
            d = (xout[i] - xx) * (xout[i] - xx) +\
                (yout[j] - yy) * (yout[j] - yy) *\
                np.cos(yout[j]) * np.cos(yout[j])

            # compute weights
            w = np.exp(-1.0 * d / kappa)

            # compute grid value
            g0[j, i] = np.sum(w * zz) / np.sum(w)

            # append index and w for computation efficiency
            indices.append(index)
            distances.append(d)

    # initializing first guess with give field
    if first_guess is not None:
        g0 = hinterp(first_guess['data'], first_guess['x'],
                     first_guess['y'], xout, yout)

    # second and more pass
    points = np.vstack((y, x)).T
    for k in range(npasses-1):
        # initializing corrected grid
        g1 = g0.copy()

        # interpolating to points
        interp_func = RegularGridInterpolator((yout, xout), g0)
        z1 = interp_func(points)

        # pass
        num = 0
        for j in range(ny):
            for i in range(nx):
                if indices[num] is None:
                    num += 1
                    continue

                # compute grid value
                index = indices[num]
                zz = z[index] - z1[index]
                d = distances[num]
                w = np.exp(-d / (gamma * kappa))
                g1[j, i] = g0[i, j] + np.sum(w*zz)/np.sum(w)
                num += 1

        # update g0
        g0 = g1.copy()

    # set negative value to zero
    if nonegative:
        g0[g0 < 0] = 0.0

    # return grid
    return {'data': g0, 'x': xout, 'y': yout}
