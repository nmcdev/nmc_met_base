# _*_ coding: utf-8 _*_

# Copyright (c) 2019 NMC Developers.
# Distributed under the terms of the GPL V3 License.

"""
  Manipulating grid field and perform mathematical algorithm, like
  partial differential, regridding and so on.

References:
* https://bitbucket.org/tmiyachi/pymet
"""

import numpy as np
from numba import jit
from scipy import interpolate, ndimage
from pyproj import Geod
from nmc_met_base import constants, arr

NA = np.newaxis
a0 = constants.Re
g = constants.g0
PI = constants.pi
d2r = PI/180.


def calc_dx_dy(lon, lat, shape='WGS84', radius=6370997.):
    """
    This definition calculates the distance between grid points
    that are in a latitude/longitude format.

    Using pyproj GEOD; different Earth Shapes
        https://jswhit.github.io/pyproj/pyproj.Geod-class.html
    Common shapes: 'sphere', 'WGS84', 'GRS80'

    :param lon: 1D or 2D longitude array.
    :param lat: 1D or 2D latitude array.
    :param shape: earth shape.
    :param radius: earth radius.
    :return: dx, dy; 2D arrays of distances between grid points
        in the x and y direction in meters

    :Example:
    >>> lat = np.arange(90,-0.1,-0.5)
    >>> lon = np.arange(0,360.1,0.5)
    >>> dx, dy = calc_dx_dy(lon, lat)
    """

    # check longitude and latitude
    if lon.ndim == 1:
        longitude, latitude = np.meshgrid(lon, lat)
    else:
        longitude = lon
        latitude = lat

    if radius != 6370997.:
        gg = Geod(a=radius, b=radius)
    else:
        gg = Geod(ellps=shape)

    dx = np.empty(latitude.shape)
    dy = np.zeros(longitude.shape)

    for i in range(latitude.shape[1]):
        for j in range(latitude.shape[0] - 1):
            _, _, dx[j, i] = gg.inv(
                longitude[j, i], latitude[j, i], longitude[j + 1, i],
                latitude[j + 1, i])
    dx[j + 1, :] = dx[j, :]

    for i in range(latitude.shape[1] - 1):
        for j in range(latitude.shape[0]):
            _, _, dy[j, i] = gg.inv(
                longitude[j, i], latitude[j, i], longitude[j, i + 1],
                latitude[j, i + 1])
    dy[:, i + 1] = dy[:, i]

    return dx, dy


def dvardx(var, lon, lat, xdim, ydim, cyclic=True, sphere=True):
    """
    calculate center finite difference along x or longitude.
    https://bitbucket.org/tmiyachi/pymet/src/8df8e3ff2f899d625939448d7e96755dfa535357/pymet/grid.py

    :param var: ndarray, grid values.
    :param lon: array_like, longitude
    :param lat: array_like, latitude
    :param xdim: the longitude dimension index
    :param ydim: the latitude dimension index
    :param cyclic: east-west boundary is cyclic
    :param sphere: sphere coordinate
    :return: ndarray

    :Examples:
     >>> var.shape
     (24, 73, 72)
     >>> lon = np.arange(0, 180, 2.5)
     >>> lat = np.arange(-90, 90.1, 2.5)
     >>> result = dvardx(var, lon, lat, 2, 1, cyclic=False)
     >>> result.shape
     (24, 73, 72)
    """

    var = np.array(var)
    ndim = var.ndim
    var = np.rollaxis(var, xdim, ndim)
    if cyclic and sphere:
        dvar = np.concatenate(((var[..., 1] - var[..., -1])[..., NA],
                               (var[..., 2:] - var[..., :-2]),
                               (var[..., 0] - var[..., -2])[..., NA]), axis=-1)
        dx = np.r_[(lon[1] + 360 - lon[-1]), (lon[2:] - lon[:-2]),
                   (lon[0] + 360 - lon[-2])]
    else:
        dvar = np.concatenate(((var[..., 1] - var[..., 0])[..., NA],
                               (var[..., 2:] - var[..., :-2]),
                               (var[..., -1] - var[..., -2])[..., NA]),
                              axis=-1)
        dx = np.r_[(lon[1] - lon[0]), (lon[2:] - lon[:-2]),
                   (lon[-1] - lon[-2])]

    dvar = np.rollaxis(dvar, ndim - 1, xdim)
    if sphere:
        dx = a0 * PI / 180. * arr.expand(dx, ndim, xdim) * \
             arr.expand(np.cos(lat * d2r), ndim, ydim)
    else:
        dx = arr.expand(dx, ndim, xdim)
    out = dvar / dx

    return out


def dvardy(var, lat, ydim, sphere=True):
    """
    calculate center finite difference along y or latitude.
    https://bitbucket.org/tmiyachi/pymet/src/8df8e3ff2f899d625939448d7e96755dfa535357/pymet/grid.py

    :param var: ndarray, grid values.
    :param lat: array_like, latitude
    :param ydim: the latitude dimension index
    :param sphere: sphere coordinate
    :return: ndarray

    :Examples:
     >>> var.shape
     (24, 73, 144)
     >>> lat = np.arange(-90, 90.1, 2.5)
     >>> result = dvardy(var, lat, 1)
     >>> result.shape
     (24, 73, 144)

    """

    var = np.array(var)
    ndim = var.ndim
    var = np.rollaxis(var, ydim, ndim)

    dvar = np.concatenate([(var[..., 1] - var[..., 0])[..., NA],
                           (var[..., 2:]-var[..., :-2]),
                           (var[..., -1] - var[..., -2])[..., NA]],
                          axis=-1)
    dy = np.r_[(lat[1]-lat[0]), (lat[2:]-lat[:-2]), (lat[-1]-lat[-2])]

    if sphere:
        dy = a0*PI/180.*dy
    out = dvar/dy
    out = np.rollaxis(out, ndim-1, ydim)

    return out


def dvardp(var, lev, zdim, punit=100.):
    """
    calculate center finite difference along vertical coordinate.
    https://bitbucket.org/tmiyachi/pymet/src/8df8e3ff2f899d625939448d7e96755dfa535357/pymet/grid.py

    :param var: ndarray, grid values.
    :param lev: 1d-array, isobaric levels.
    :param zdim: the vertical dimension index.
    :param punit: pressure level units.
    :return: ndarray.
    """

    var = np.array(var)
    ndim = var.ndim
    lev = lev * punit

    # roll lat dim axis to last
    var = np.rollaxis(var, zdim, ndim)
    dvar = np.concatenate([(var[..., 1] - var[..., 0])[..., NA],
                           (var[..., 2:] - var[..., :-2]),
                           (var[..., -1] - var[..., -2])[..., NA]],
                          axis=-1)
    dp = np.r_[np.log(lev[1] / lev[0]) * lev[0],
               np.log(lev[2:] / lev[:-2]) * lev[1:-1],
               np.log(lev[-1] / lev[-2]) * lev[-1]]

    out = dvar / dp

    # reroll lat dim axis to original dim
    out = np.rollaxis(out, ndim - 1, zdim)

    return out


def d2vardx2(var, lon, lat, xdim, ydim, cyclic=True, sphere=True):
    """
    calculate second center finite difference along x or longitude.
    https://bitbucket.org/tmiyachi/pymet/src/8df8e3ff2f899d625939448d7e96755dfa535357/pymet/grid.py

    :param var: ndarray, grid values.
    :param lon: array_like, longitude
    :param lat: array_like, latitude
    :param xdim: the longitude dimension index
    :param ydim: the latitude dimension index
    :param cyclic: east-west boundary is cyclic
    :param sphere: sphere coordinate
    :return: ndarray
    """

    var = np.array(var)
    ndim = var.ndim

    # roll lon dim axis to last
    var = np.rollaxis(var, xdim, ndim)

    if cyclic and sphere:
        dvar = np.concatenate(((var[..., 1]-2*var[..., 0] +
                                var[..., -1])[..., NA],
                               (var[..., 2:]-2*var[..., 1:-1] + var[..., :-2]),
                               (var[..., 0]-2*var[..., -1] +
                                var[..., -2])[..., NA]), axis=-1)
        dx = np.r_[(lon[1]+360-lon[-1]), (lon[2:]-lon[:-2]),
                   (lon[0]+360-lon[-2])]
    else:  # edge is zero
        dvar = np.concatenate(((var[..., 0]-var[..., 0])[..., NA],
                               (var[..., 2:]-2*var[..., 1:-1]+var[..., :-2]),
                               (var[..., 0]-var[..., 0])[..., NA]), axis=-1)
        dx = np.r_[(lon[1]-lon[0]), (lon[2:]-lon[:-2]), (lon[-1]-lon[-2])]

    dvar = np.rollaxis(dvar, ndim-1, xdim)
    if sphere:
        dx2 = a0 ** 2 * (PI/180.) ** 2 * arr.expand(dx ** 2, ndim, xdim) * \
              arr.expand(np.cos(lat * d2r) ** 2, ndim, ydim)
    else:
        dx2 = arr.expand(dx ** 2, ndim, xdim)
    out = 4.*dvar/dx2

    # reroll lon dim axis to original dim
    out = np.rollaxis(out, ndim-1, xdim)

    return out


def d2vardy2(var, lat, ydim, sphere=True):
    """
    calculate second center finite difference along y or latitude.
    https://bitbucket.org/tmiyachi/pymet/src/8df8e3ff2f899d625939448d7e96755dfa535357/pymet/grid.py

    :param var: ndarray, grid values.
    :param lat: array_like, latitude
    :param ydim: the latitude dimension index
    :param sphere: sphere coordinate
    :return: ndarray
    """

    var = np.array(var)
    ndim = var.ndim

    # roll lat dim axis to last
    var = np.rollaxis(var, ydim, ndim)

    # edge is zero
    dvar = np.concatenate([(var[..., 0] - var[..., 0])[..., NA],
                           (var[..., 2:] - 2*var[..., 1:-1] + var[..., :-2]),
                           (var[..., 0] - var[..., 0])[..., NA]], axis=-1)
    dy = np.r_[(lat[1]-lat[0]), (lat[2:]-lat[:-2]), (lat[-1]-lat[-2])]

    if sphere:
        dy2 = a0**2 * dy**2
    else:
        dy2 = dy**2
    out = 4.*dvar/dy2

    # reroll lat dim axis to original dim
    out = np.rollaxis(out, ndim-1, ydim)

    return out


def dvardvar(var1, var2, dim):
    """
    Calculate d(var1)/d(var2) along axis=dim.
    https://bitbucket.org/tmiyachi/pymet/src/8df8e3ff2f899d625939448d7e96755dfa535357/pymet/grid.py

    :param var1: numpy nd array, denominator of derivative
    :param var2: numpy nd array, numerator of derivative
    :param dim: along dimension.
    :return:
    """

    var1, var2 = np.array(var1), np.array(var2)
    ndim = var1.ndim

    # roll dim axis to last
    var1 = np.rollaxis(var1, dim, ndim)
    var2 = np.rollaxis(var2, dim, ndim)

    dvar1 = np.concatenate([(var1[..., 1] - var1[..., 0])[..., NA],
                            (var1[..., 2:] - var1[..., :-2]),
                            (var1[..., -1] - var1[..., -2])[..., NA]], axis=-1)
    dvar2 = np.concatenate([(var2[..., 1] - var2[..., 0])[..., NA],
                            (var2[..., 2:] - var2[..., :-2]),
                            (var2[..., -1] - var2[..., -2])[..., NA]], axis=-1)

    out = dvar1 / dvar2

    # reroll lat dim axis to original dim
    out = np.rollaxis(out, ndim - 1, dim)

    return out


def div(u, v, lon, lat, xdim, ydim, cyclic=True, sphere=True):
    """
    Calculate horizontal divergence.

    :param u: ndarray, u-component wind.
    :param v: ndarray, v-component wind.
    :param lon: array_like, longitude.
    :param lat: array_like, latitude.
    :param xdim: the longitude dimension index
    :param ydim: the latitude dimension index
    :param cyclic: east-west boundary is cyclic
    :param sphere: sphere coordinate
    :return: ndarray
    """

    u, v = np.array(u), np.array(v)
    ndim = u.ndim

    out = dvardx(u, lon, lat, xdim, ydim, cyclic=cyclic,
                 sphere=sphere) + dvardy(v, lat, ydim, sphere=sphere)
    if sphere:
        out = out - v * arr.expand(np.tan(lat * d2r), ndim, ydim) / a0

    out = np.rollaxis(out, ydim, 0)
    out[0, ...] = 0.
    out[-1, ...] = 0.
    out = np.rollaxis(out, 0, ydim + 1)

    return out


def rot(u, v, lon, lat, xdim, ydim, cyclic=True, sphere=True):
    """
    Calculate vertical vorticity.

    :param u: ndarray, u-component wind.
    :param v: ndarray, v-component wind.
    :param lon: array_like, longitude.
    :param lat: array_like, latitude.
    :param xdim: the longitude dimension index
    :param ydim: the latitude dimension index
    :param cyclic: east-west boundary is cyclic
    :param sphere: sphere coordinate
    :return: ndarray
    """

    u, v = np.array(u), np.array(v)
    ndim = u.ndim

    out = dvardx(v, lon, lat, xdim, ydim, cyclic=cyclic,
                 sphere=sphere) - dvardy(u, lat, ydim, sphere=sphere)
    if sphere:
        out = out + u * arr.expand(np.tan(lat * d2r), ndim, ydim) / a0

    out = np.rollaxis(out, ydim, 0)
    out[0, ...] = 0.
    out[-1, ...] = 0.
    out = np.rollaxis(out, 0, ydim + 1)

    return out


def laplacian(var, lon, lat, xdim, ydim, cyclic=True, sphere=True):
    """
    Calculate laplacian operation on sphere.

    :param var: ndarray, grid values.
    :param lon: array_like, longitude
    :param lat: array_like, latitude
    :param xdim: the longitude dimension index
    :param ydim: the latitude dimension index
    :param cyclic: east-west boundary is cyclic
    :param sphere: sphere coordinate
    :return: ndarray
    """

    var = np.asarray(var)
    ndim = var.ndim

    if sphere:
        out = d2vardx2(var, lon, lat, xdim, ydim,
                       cyclic=cyclic, sphere=sphere) + \
              d2vardy2(var, lat, ydim, sphere=sphere) - \
              arr.expand(np.tan(lat * d2r), ndim, ydim) * \
              dvardy(var, lat, ydim)/a0
    else:
        out = d2vardx2(var, lon, lat, xdim, ydim,
                       cyclic=cyclic, sphere=sphere) + \
              d2vardy2(var, lat, ydim, sphere=sphere)

    return out


def grad(var, lon, lat, xdim, ydim, cyclic=True, sphere=True):
    """
    Calculate gradient operator.

    :param var: ndarray, grid values.
    :param lon: array_like, longitude
    :param lat: array_like, latitude
    :param xdim: the longitude dimension index
    :param ydim: the latitude dimension index
    :param cyclic: east-west boundary is cyclic
    :param sphere: sphere coordinate
    :return: ndarray
    """

    var = np.asarray(var)

    outu = dvardx(var, lon, lat, xdim, ydim, cyclic=cyclic, sphere=sphere)
    outv = dvardy(var, lat, ydim, sphere=sphere)

    return outu, outv


def skgrad(var, lon, lat, xdim, ydim, cyclic=True, sphere=True):
    """
    Calculate skew gradient.

    :param var: ndarray, grid values.
    :param lon: array_like, longitude
    :param lat: array_like, latitude
    :param xdim: the longitude dimension index
    :param ydim: the latitude dimension index
    :param cyclic: east-west boundary is cyclic
    :param sphere: sphere coordinate
    :return: ndarray
    """

    var = np.asarray(var)

    outu = -dvardy(var, lat, ydim, sphere=sphere)
    outv = dvardx(var, lon, lat, xdim, ydim, cyclic=cyclic, sphere=sphere)

    return outu, outv


def gradient_sphere(f, *varargs):
    """
    Return the gradient of a 2-dimensional array on a sphere given a latitude
      and longitude vector.

    The gradient is computed using central differences in the interior
      and first differences at the boundaries. The returned gradient hence has
      the same shape as the input array.

    https://github.com/scavallo/python_scripts/blob/master/utils/weather_modules.py

    :param f: A 2-dimensional array containing samples of a scalar function.
    :param varargs: latitude, longitude and so on.
    :return: dfdx and dfdy arrays of the same shape as `f`
             giving the derivative of `f` with
             respect to each dimension.

    :Example:
    temperature = temperature(pressure,latitude,longitude)
    levs = pressure vector
    lats = latitude vector
    lons = longitude vector
    >>> tempin = temperature[5,:,:]
    >>> dfdlat, dfdlon = gradient_sphere(tempin, lats, lons)
    >>> dfdp, dfdlat, dfdlon = gradient_sphere(temperature, levs, lats, lons)
    """

    r_earth = 6371200.
    N = f.ndim          # number of dimensions
    n = len(varargs)    # number of arguments
    argsin = list(varargs)

    if N != n:
        raise SyntaxError(
            "dimensions of input must match the remaining arguments")

    df = np.gradient(f)

    if n == 2:
        lats = argsin[0]
        lons = argsin[1]
        dfdy = df[0]
        dfdx = df[1]
    elif n == 3:
        levs = argsin[0]
        lats = argsin[1]
        lons = argsin[2]
        dfdz = df[0]
        dfdy = df[1]
        dfdx = df[2]
    else:
        raise SyntaxError("invalid number of arguments")

    otype = f.dtype.char
    if otype not in ['f', 'd', 'F', 'D']:
        otype = 'd'

    latarr = np.zeros_like(f).astype(otype)
    lonarr = np.zeros_like(f).astype(otype)
    if N == 2:
        nlat, nlon = np.shape(f)
        for jj in range(0, nlat):
            latarr[jj, :] = lats[jj]
        for ii in range(0, nlon):
            lonarr[:, ii] = lons[ii]
    else:
        nz, nlat, nlon = np.shape(f)
        for jj in range(0, nlat):
            latarr[:, jj, :] = lats[jj]
        for ii in range(0, nlon):
            lonarr[:, :, ii] = lons[ii]

    # use central differences on interior and first differences on endpoints

    dlats = np.zeros_like(lats).astype(otype)
    dlats[1:-1] = (lats[2:] - lats[:-2])
    dlats[0] = (lats[1] - lats[0])
    dlats[-1] = (dlats[-2] - dlats[-1])

    dlons = np.zeros_like(lons).astype(otype)
    dlons[1:-1] = (lons[2:] - lons[:-2])
    dlons[0] = (lons[1] - lons[0])
    dlons[-1] = (dlons[-2] - dlons[-1])

    dlatarr = np.zeros_like(f).astype(otype)
    dlonarr = np.zeros_like(f).astype(otype)
    if N == 2:
        for jj in range(0, nlat):
            dlatarr[jj, :] = dlats[jj]
        for ii in range(0, nlon):
            dlonarr[:, ii] = dlons[ii]
    elif N == 3:
        for jj in range(0, nlat):
            dlatarr[:, jj, :] = dlats[jj]
        for ii in range(0, nlon):
            dlonarr[:, :, ii] = dlons[ii]

    dlatsrad = dlatarr * (PI / 180.)
    dlonsrad = dlonarr * (PI / 180.)
    latrad = latarr * (PI / 180.)

    if n == 2:
        dx1 = r_earth * dlatsrad
        dx2 = r_earth * np.cos(latrad) * dlonsrad
        dfdy = dfdy / dx1
        dfdx = dfdx / dx2

        return dfdy, dfdx
    elif n == 3:
        dx1 = r_earth * dlatsrad
        dx2 = r_earth * np.cos(latrad) * dlonsrad
        dfdy = dfdy / dx1
        dfdx = dfdx / dx2

        zin = levs
        dz = np.zeros_like(zin).astype(otype)
        dz[1:-1] = (zin[2:] - zin[:-2]) / 2.0
        dz[0] = (zin[1] - zin[0])
        dz[-1] = (zin[-1] - zin[-2])
        dx3 = np.ones_like(f).astype(otype)
        for kk in range(0, nz):
            dx3[kk, :, :] = dz[kk]

        dfdz = dfdz / dx3
        return dfdz, dfdy, dfdx


def vint(var, bottom, top, lev, zdim, punit=100.):
    """
    Calculate vertical integration.

    :param var: array_like.
    :param bottom: bottom boundary of integration.
    :param top: top boundary of integration.
    :param lev: isobaric levels.
    :param zdim: vertical dimension.
    :param punit: levels units.
    :return: array_like.
    """

    var = np.ma.asarray(var)
    lev = np.asarray(lev)
    ndim = var.ndim

    lev = lev[(lev <= bottom) & (lev >= top)]
    lev_m = np.r_[bottom, (lev[1:] + lev[:-1])/2., top]
    dp = lev_m[:-1] - lev_m[1:]

    # roll lat dim axis to last
    var = arr.mrollaxis(var, zdim, ndim)
    out = var[..., (lev <= bottom) & (lev >= top)] * dp / g * punit
    if bottom > top:
        out = out.sum(axis=-1)
    else:
        out = -out.sum(axis=-1)
    return out


def total_col(infld, pres, temp, hght):
    """
    Compute column integrated value of infld.
    https://github.com/scavallo/classcode/blob/master/utils/weather_modules.py

    :param infld:  Input 3D field to column integrate
    :param pres: Input 3D air pressure (Pa)
    :param temp: Input 3D temperature field (K)
    :param hght: Input 3D geopotential height field (m
    :return:  Output total column integrated value
    """

    [iz, iy, ix] = np.shape(infld)
    density = pres / (287 * temp)
    tmp = pres[0, :, :].squeeze()

    coltot = np.zeros_like(tmp).astype('f')
    for jj in range(0, iy):
        for ii in range(0, ix):
            colnow = infld[:, jj, ii] * density[:, jj, ii]
            hghtnow = hght[:, jj, ii].squeeze()
            coltot[jj, ii] = np.trapz(colnow[::-1], hghtnow[::-1])

    return coltot


def vmean(var, bottom, top, lev, zdim):
    """
    Calculate vertical mean.

    :param var: array_like.
    :param bottom: bottom boundary of integration.
    :param top: top boundary of integration.
    :param lev: isobaric levels.
    :param zdim: vertical dimension.
    :return: array_like.
    """

    var = np.ma.asarray(var)
    lev = np.asarray(lev)
    ndim = var.ndim

    lev = lev[(lev <= bottom) & (lev >= top)]
    lev_m = np.r_[bottom, (lev[1:] + lev[:-1])/2., top]
    dp = lev_m[:-1] - lev_m[1:]

    # roll lat dim axis to last
    var = arr.mrollaxis(var, zdim, ndim)
    out = var[..., (lev <= bottom) & (lev >= top)] * dp
    out = out.sum(axis=-1)/(dp.sum())

    return out


def vinterp(var, oldz, newz, zdim, logintrp=True, bounds_error=True):
    """
    perform vertical linear interpolation.

    :param var: array_like variable.
    :param oldz: original vertical level.
    :param newz: new vertical level.
    :param zdim: the dimension of vertical.
    :param logintrp: log linear interpolation.
    :param bounds_error: options for scipy.interpolate.interp1d.
    :return:
    """

    var = np.array(var)
    ndim = var.ndim

    new_z = np.array(newz)
    old_z = np.array(oldz)
    if logintrp:
        old_z = np.log(old_z)
        new_z = np.log(new_z)
    old_zn = var.shape[zdim]
    new_zn = len(new_z)

    # roll z dim axis to last
    var = np.rollaxis(var, zdim, ndim)
    old_shape = var.shape
    new_shape = list(old_shape)
    new_shape[-1] = new_zn
    var = var.reshape(-1, old_zn)
    if old_z.ndim == ndim:
        old_z = np.rollaxis(old_z, zdim, ndim).reshape(-1, old_zn)
        f = interpolate.interp1d(old_z, var, axis=-1, kind='linear',
                                 bounds_error=bounds_error)
        out = f(new_z)
    elif old_z.ndim == 1:
        f = interpolate.interp1d(old_z, var, kind='linear',
                                 bounds_error=bounds_error)
        out = f(new_z)

    # reroll lon dim axis to original dim
    out = out.reshape(new_shape)
    out = np.rollaxis(out, ndim - 1, zdim)

    return out


def _grid_smooth_bes(x):
    """
    Bessel function.  (copied from RIP)

    :param x: float number
    :return: bessel function value.
    """

    rint = 0.0
    for i in range(1000):
        u = i * 0.001 - 0.0005
        rint = rint + np.sqrt(1 - u*u) * np.cos(x*u)*0.001
    return 2.0 * x * rint / (4.0 * np.arctan(1.0))


def grid_smooth(field, radius=6, method='CRES', **kwargs):
    """
    Perform grid field smooth filter.
    refer to
    https://github.com/Unidata/IDV/blob/master/src/ucar/unidata/data/grid/GridUtil.java

    * Apply a weigthed smoothing function to the grid.
      The smoothing types are:

    SMOOTH_CRESSMAN: the smoothed value is given by a weighted average of
      values at surrounding grid points.  The weighting function is the
      Cressman weighting function:
             w = ( D**2 - d**2 ) / ( D**2 + d**2 )
      In the above, d is the distance (in grid increments) of the neighboring
      point to the smoothing point, and D is the radius of influence
      [in grid increments]

    SMOOTH_CIRCULAR: the weighting function is the circular apperture
      diffraction function (following a suggestion of Barnes et al. 1996):
              w = bessel(3.8317*d/D)/(3.8317*d/D)

    SMOOTH_RECTANGULAR: the weighting function is the product of the
      rectangular aperture diffraction function in the x and y
      directions (the function used in Barnes et al. 1996):
              w = [sin(pi*x/D)/(pi*x/D)]*[sin(pi*y/D)/(pi*y/D)]
      Adapted from smooth.f written by Mark Stoelinga in his RIP package

    :param field: 2D array variable.
    :param radius: if type is CRES, CIRC or RECT, radius of window
                     in grid units (in grid increments)
                   if type is GWFS, radius is the standard deviation
                     of gaussian function, larger for smoother
    :param method: string value, smooth type:
                   SM9S, 9-point smoother
                   GWFS, Gaussian smoother
                   CRES, Cressman smoother, default
                   CIRC, Barnes circular apperture diffraction function
                   RECT, Barnes rectangular apperture diffraction function
    :param kwargs: parameters for scipy.ndimage.filters.convolve function.
    :return: 2D array like smoothed field.
    """

    # construct kernel
    if method == 'SM9S':
        kernel = [[0.3, 0.5, 0.3], [0.5, 1, 0.5], [0.3, 0.5, 0.3]]
    elif method == 'GWFS':
        return ndimage.filters.gaussian_filter(field, radius, **kwargs)
    elif method == 'CRES':
        width = np.int(np.ceil(radius)*2+1)
        center = np.ceil(radius)
        kernel = np.zeros((width, width))
        for jj in range(width):
            for ii in range(width):
                x = ii - center
                y = jj - center
                d = np.sqrt(x*x + y*y)
                if d > radius:
                    continue
                kernel[jj, ii] = (radius*radius - d*d)/(radius*radius + d*d)
    elif method == 'CIRC':
        width = np.int(np.ceil(radius) * 2 + 1)
        center = np.ceil(radius)
        kernel = np.zeros((width, width))
        for jj in range(width):
            for ii in range(width):
                x = ii - center
                y = jj - center
                d = np.sqrt(x * x + y * y)
                if d > radius:
                    continue
                if d == 0.:
                    kernel[jj, ii] = 0.5
                else:
                    kernel[jj, ii] = _grid_smooth_bes(
                        3.8317*d/radius)/(3.8317*d/radius)
    elif method == 'RECT':
        width = np.int(np.ceil(radius) * 2 + 1)
        center = np.ceil(radius)
        kernel = np.zeros((width, width))
        for jj in range(width):
            for ii in range(width):
                x = ii - center
                y = jj - center
                d = np.sqrt(x * x + y * y)
                if d > radius:
                    continue
                kernel[jj, ii] = (np.sin(PI*x/radius)/(PI*x/radius)) * \
                                 (np.sin(PI*y/radius)/(PI*y/radius))
    else:
        return field

    # return smoothed field
    return ndimage.filters.convolve(field, kernel, **kwargs)


@jit
def grid_smooth_area_average(in_field, lon, lat, radius=400.e3):
    """
    Smoothing grid field with circle area average.

    :param in_field: 2D or multiple dimension array grid field,
                     the rightest dimension [..., lat, lon].
    :param lon: 1D array longitude.
    :param lat: 1D array latitude.
    :param radius: smooth radius, [m]
    :return: smoothed grid field.
    """

    # set constants
    deg_to_rad = np.arctan(1.0)/45.0
    earth_radius = 6371000.0

    # reshape field to 3d array
    old_shape = in_field.shape
    if np.ndim(in_field) == 2:
        ndim = 1
    else:
        ndim = np.product(old_shape[0:-2])
    field = in_field.reshape(ndim, *old_shape[-2:])

    # grid coordinates
    x, y = np.meshgrid(lon, lat)

    # define output field
    out_field = np.full_like(field, np.nan)

    # loop every grid point
    lat1 = np.cos(lat * deg_to_rad)
    lat2 = np.cos(y * deg_to_rad)
    for j in range(lat.size):
        dlat = (y - lat[j]) * deg_to_rad
        a1 = (np.sin(dlat/2.0))**2
        b1 = lat1[j] * lat2
        for i in range(lon.size):
            # great circle distance
            dlon = (x - lon[i]) * deg_to_rad
            a = np.sqrt(a1+b1*(np.sin(dlon/2.0))**2)
            dist = earth_radius * 2.0 * np.arcsin(a)
            dist = dist <= radius

            # compute average
            if np.any(dist):
                for k in range(ndim):
                    temp = field[k, :, :]
                    out_field[k, j, i] = np.mean(temp[dist])

    # return smoothed field
    return out_field.reshape(old_shape)


def grid_subset(lon, lat, bound):
    """
    Get the upper and lower bound of a grid subset.

    :param lon: 1D array, longitude.
    :param lat: 1D array, latitude.
    :param bound: subset boundary, [lonmin, lonmax, latmin, latmax]
    :return: subset boundary index.
    """

    # latitude lower and upper index
    latli = np.argmin(np.abs(lat - bound[2]))
    latui = np.argmin(np.abs(lat - bound[3]))

    # longitude lower and upper index
    lonli = np.argmin(np.abs(lon - bound[0]))
    lonui = np.argmin(np.abs(lon - bound[1]))

    # return subset boundary index
    return lonli, lonui+1, latli, latui+1


def vertical_cross(in_field, lon, lat, line_points, npts=100):
    """
    Interpolate 2D or multiple dimensional grid data to vertical cross section.

    :param in_field: 2D or multiple dimensional grid data,
                     the rightest dimension [..., lat, lon].
    :param lon: grid data longitude.
    :param lat: grid data latitude.
    :param line_points: cross section line points,
                        should be [n_points, 2] array.
    :param npts: the point number of great circle line.
    :return: cross section [..., n_points], points
    """

    if np.ndim(in_field) < 2:
        raise ValueError("in_field must be at least 2 dimension")

    # reshape field to 3d array
    old_shape = in_field.shape
    if np.ndim(in_field) == 2:
        field = in_field.reshape(1, *old_shape)
    else:
        field = in_field.reshape(np.product(old_shape[0:-2]), *old_shape[-2:])

    # get great circle points
    points = None
    n_line_points = line_points.shape[0]
    geod = Geod("+ellps=WGS84")
    for i in range(n_line_points-1):
        seg_points = geod.npts(
            lon1=line_points[i, 0], lat1=line_points[i, 1],
            lon2=line_points[i+1, 0], lat2=line_points[i+1, 1], npts=npts)
        if points is None:
            points = np.array(seg_points)
        else:
            points = np.vstack((points, np.array(seg_points)))

    # convert to pixel coordinates
    x = np.interp(points[:, 0], lon, np.arange(len(lon)))
    y = np.interp(points[:, 1], lat, np.arange(len(lat)))

    # loop every level
    zdata = []
    for i in range(field.shape[0]):
        zdata.append(
            ndimage.map_coordinates(np.transpose(field[i, :, :]),
                                    np.vstack((x, y))))

    # reshape zdata
    zdata = np.array(zdata)
    if np.ndim(in_field) > 2:
        zdata = zdata.reshape(np.append(old_shape[0:-2], points.shape[0]))

    # return vertical cross section
    return zdata, points


def interpolate1d(x, z, points, mode='linear', bounds_error=False):
    """
    1D interpolation routine.

    :param x: 1D array of x-coordinates on which to interpolate
    :param z: 1D array of values for each x
    :param points: 1D array of coordinates where interpolated values are sought
    :param mode: Determines the interpolation order. Options are
        'constant' - piecewise constant nearest neighbour interpolation
        'linear' - bilinear interpolation using the two
                   nearest neighbours (default)
    :param bounds_error: Boolean flag. If True (default) an exception will
                         be raised when interpolated values are requested
                         outside the domain of the input data. If False, nan
                         is returned for those values
    :return: 1D array with same length as points with interpolated values

    :Notes:
        Input coordinates x are assumed to be monotonically increasing,
        but need not be equidistantly spaced.
        z is assumed to have dimension M where M = len(x).
    """

    # Check inputs
    #
    # make sure input vectors are numpy array
    x = np.array(x)
    # Input vectors should be monotoneously increasing.
    if (not np.min(x) == x[0]) and (not max(x) == x[-1]):
        raise Exception('Input vector x must be monotoneously increasing.')
    # Input array Z's dimensions
    z = np.array(z)
    if not len(x) == len(z):
        raise Exception('Input array z must have same length as x')

    # Get interpolation points
    in_points = np.array(points)
    xi = in_points[:]

    # Check boundary
    if bounds_error:
        if np.min(xi) < x[0] or np.max(xi) > x[-1]:
            raise Exception('Interpolation points was out of the domain.')

    # Identify elements that are outside interpolation domain or NaN
    outside = (xi < x[0]) + (xi > x[-1])
    outside += np.isnan(xi)

    inside = -outside
    xi = xi[inside]

    # Find upper neighbours for each interpolation point
    idx = np.searchsorted(x, xi, side='left')

    # Internal check (index == 0 is OK)
    msg = 'Interpolation point outside domain. This should never happen.'
    if len(idx) > 0:
        if not max(idx) < len(x):
            raise RuntimeError(msg)

    # Get the two neighbours for each interpolation point
    x0 = x[idx - 1]
    x1 = x[idx]

    z0 = z[idx - 1]
    z1 = z[idx]

    # Coefficient for weighting between lower and upper bounds
    alpha = (xi - x0) / (x1 - x0)

    if mode == 'linear':
        # Bilinear interpolation formula
        dx = z1 - z0
        zeta = z0 + alpha * dx
    else:
        # Piecewise constant (as verified in input_check)

        # Set up masks for the quadrants
        left = alpha < 0.5

        # Initialise result array with all elements set to right neighbour
        zeta = z1

        # Then set the left neighbours
        zeta[left] = z0[left]

    # Self test
    if len(zeta) > 0:
        mzeta = np.nanmax(zeta)
        mz = np.nanmax(z)
        msg = ('Internal check failed. Max interpolated value %.15f '
               'exceeds max grid value %.15f ' % (mzeta, mz))
        if not (np.isnan(mzeta) or np.isnan(mz)):
            if not mzeta <= mz:
                raise RuntimeError(msg)

    # Populate result with interpolated values for points inside domain
    # and NaN for values outside
    r = np.zeros(len(points))
    r[inside] = zeta
    r[outside] = np.nan

    return r


def interpolate2d(x, y, Z, points, mode='linear', bounds_error=False):
    """
    Interpolating from 2D field to points.
    Refer to
    https://github.com/inasafe/python-safe/blob/master/safe/engine/interpolation2d.py.
    * provides piecewise constant (nearest neighbour) and
    * bilinear interpolation is fast (based on numpy vector operations)
    * depends only on numpy
    * guarantees that interpolated values never exceed the four nearest
    * neighbours handles missing values in domain sensibly using NaN
    * is unit tested with a range of common and corner cases

    :param x: 1D array of x-coordinates of the mesh on which to interpolate
    :param y: 1D array of y-coordinates of the mesh on which to interpolate
    :param Z: 2D array of values for each x, y pair
    :param points: Nx2 array of coordinates where interpolated values
                   are sought
    :param mode: Determines the interpolation order. Options are
        'constant' - piecewise constant nearest neighbour interpolation
        'linear' - bilinear interpolation using the four nearest
                   neighbours (default)
    :param bounds_error: Boolean flag. If True (default) an exception will
                         be raised when interpolated values are requested
                         outside the domain of the input data. If False, nan
                         is returned for those values.
    :return: 1D array with same length as points with interpolated values

    :Notes:
        Input coordinates x and y are assumed to be monotonically increasing,
        but need not be equidistantly spaced.

        Z is assumed to have dimension M x N, where M = len(x) and N = len(y).
        In other words it is assumed that the x values follow the first
        (vertical) axis downwards and y values the second (horizontal) axis
        from left to right.

        2D bilinear interpolation aims at obtaining an interpolated value z at
        a point (x,y) which lies inside a square formed by points (x0, y0),
        (x1, y0), (x0, y1) and (x1, y1) for which values z00, z10, z01 and
        z11 are known.

        This obtained be first applying equation (1) twice in in the
        x-direction to obtain interpolated points q0 and q1 for (x, y0)
        and (x, y1), respectively.
        q0 = alpha*z10 + (1-alpha)*z00         (2)
          and
        q1 = alpha*z11 + (1-alpha)*z01         (3)

        Then using equation (1) in the y-direction on the results from
          (2) and (3)
          z = beta*q1 + (1-beta)*q0              (4)
          where beta = (y-y0)/(y1-y0)            (4a)

        Substituting (2) and (3) into (4) yields
          z = alpha*beta*z11 + beta*z01 - alpha*beta*z01 +
              alpha*z10 + z00 - alpha*z00 - alpha*beta*z10 - beta*z00 +
              alpha*beta*z00
            = alpha*beta*(z11 - z01 - z10 + z00) +
              alpha*(z10 - z00) + beta*(z01 - z00) + z00
        which can be further simplified to
          z = alpha*beta*(z11 - dx - dy - z00) + alpha*dx + beta*dy + z00  (5)
        where
          dx = z10 - z00
          dy = z01 - z00
        Equation (5) is what is implemented in the function
          interpolate2d above.
    """

    # Check inputs
    #
    # make sure input vectors are numpy array
    x = np.array(x)
    y = np.array(y)
    # Input vectors should be monotoneously increasing.
    if (not np.min(x) == x[0]) and (not max(x) == x[-1]):
        raise Exception('Input vector x must be monotoneously increasing.')
    if (not np.min(y) == y[0]) and (not max(y) == y[-1]):
        raise Exception('Input vector y must be monotoneously increasing.')
    # Input array Z's dimensions
    Z = np.array(Z)
    m, n = Z.shape
    if not(len(x) == m and len(y) == n):
        raise Exception(
            'Input array Z must have dimensions corresponding to the '
            'lengths of the input coordinates x and y')

    # Get interpolation points
    in_points = np.array(points)
    xi = in_points[:, 0]
    eta = in_points[:, 1]

    # Check boundary
    if bounds_error:
        if np.min(xi) < x[0] or np.max(xi) > x[-1] or \
                np.min(eta) < y[0] or np.max(eta) > y[-1]:
            raise RuntimeError('Interpolation points was out of the domain.')

    # Identify elements that are outside interpolation domain or NaN
    outside = (xi < x[0]) + (eta < y[0]) + (xi > x[-1]) + (eta > y[-1])
    outside += np.isnan(xi) + np.isnan(eta)
    inside = -outside
    xi = xi[inside]
    eta = eta[inside]

    # Find upper neighbours for each interpolation point
    idx = np.searchsorted(x, xi, side='left')
    idy = np.searchsorted(y, eta, side='left')

    # Internal check (index == 0 is OK)
    msg = 'Interpolation point outside domain. This should never happen.'
    if len(idx) > 0:
        if not max(idx) < len(x):
            raise RuntimeError(msg)
    if len(idy) > 0:
        if not max(idy) < len(y):
            raise RuntimeError(msg)

    # Get the four neighbours for each interpolation point
    x0 = x[idx - 1]
    x1 = x[idx]
    y0 = y[idy - 1]
    y1 = y[idy]

    z00 = Z[idx - 1, idy - 1]
    z01 = Z[idx - 1, idy]
    z10 = Z[idx, idy - 1]
    z11 = Z[idx, idy]

    # Coefficients for weighting between lower and upper bounds
    oldset = np.seterr(invalid='ignore')  # Suppress warnings
    alpha = (xi - x0) / (x1 - x0)
    beta = (eta - y0) / (y1 - y0)
    np.seterr(**oldset)  # Restore

    if mode == 'linear':
        # Bilinear interpolation formula
        dx = z10 - z00
        dy = z01 - z00
        z = z00 + alpha * dx + beta * dy + alpha * beta * (z11 - dx - dy - z00)
    else:
        # Piecewise constant (as verified in input_check)
        # Set up masks for the quadrants
        left = alpha < 0.5
        right = -left
        lower = beta < 0.5
        upper = -lower

        lower_left = lower * left
        lower_right = lower * right
        upper_left = upper * left

        # Initialise result array with all elements set to upper right
        z = z11

        # Then set the other quadrants
        z[lower_left] = z00[lower_left]
        z[lower_right] = z10[lower_right]
        z[upper_left] = z01[upper_left]

    # Self test
    if len(z) > 0:
        mz = np.nanmax(z)
        mZ = np.nanmax(Z)
        msg = ('Internal check failed. Max interpolated value %.15f '
               'exceeds max grid value %.15f ' % (mz, mZ))
        if not (np.isnan(mz) or np.isnan(mZ)):
            if not mz <= mZ:
                raise RuntimeError(msg)

    # Populate result with interpolated values for
    #   points inside domain and NaN for values outside
    r = np.zeros(len(points))
    r[inside] = z
    r[outside] = np.nan

    return r
