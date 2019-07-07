# _*_ coding: utf-8 _*_

# Copyright (c) 2019 NMC Developers.
# Distributed under the terms of the GPL V3 License.

"""
Geodesy calculation.
"""

import numpy as np
from numba import jit
import nmc_met_base.constants as const


def haversine_np(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    All args must be of equal length.

    :param lon1: point 1 longitudes.
    :param lat1: point 1 latitudes.
    :param lon2: point 2 longitudes.
    :param lat2: point 2 latitudes.
    :return: great circle distance in meters.
    """

    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    return 6371.e3 * c


def area_weighted_mean(lon, lat, data):
    """Calculate the mean of gridded data on a sphere.
    Data points on the Earth's surface are often represented as a grid. As the
    grid cells do not have a constant area they have to be weighted when
    calculating statistical properties (e.g. mean).
    This function returns the weighted mean assuming a perfectly spherical
    globe.

    refer to:
    https://github.com/atmtools/typhon/blob/master/typhon/geographical.py

    Parameters:
        lon (ndarray): Longitude (M) angles [degree].
        lat (ndarray): Latitude (N) angles [degree].
        data ()ndarray): Data array (N x M).
    Returns:
        float: Area weighted mean.
    """
    # Calculate coordinates and steradian (in rad).
    lon = np.deg2rad(lon)
    lat = np.deg2rad(lat)
    dlon = np.diff(lon)
    dlat = np.diff(lat)

    # Longitudal mean
    middle_points = (data[:, 1:] + data[:, :-1]) / 2
    norm = np.sum(dlon)
    lon_integral = np.sum(middle_points * dlon, axis=1) / norm

    # Latitudal mean
    lon_integral *= np.cos(lat)  # Consider varying grid area (N-S).
    middle_points = (lon_integral[1:] + lon_integral[:-1]) / 2
    norm = np.sum(np.cos((lat[1:] + lat[:-1]) / 2) * dlat)

    return np.sum(middle_points * dlat) / norm


def stations_mean_distance(lon, lat):
    """
    Determine the mean separation distances of the observing stations.
    https://www.atmos.illinois.edu/~jtrapp/Ex3.1.pdf

    Arguments:
        lon {numpy array} -- longitude array.
        lat {numpy array} -- latitude array.
    """

    # check input vector
    if len(lon) != len(lat):
        raise Exception("lon length is not equal to lat length.")

    # compute minimu distance
    min_dist = np.full(len(lon), 0.0)
    for i in range(len(lat)):
        dx = const.Re * np.cos(lat) * (lon - lon[i]) * const.d2r
        dy = const.Re * (lat - lat[i]) * const.d2r
        d = np.sqrt(dx*dx + dy*dy)
        min_dist[i] = np.min(d[d != 0])

    # return mean distance
    return np.mean(min_dist)

