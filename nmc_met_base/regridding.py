# _*_ coding: utf-8 _*_

# Copyright (c) 2019 NMC Developers.
# Distributed under the terms of the GPL V3 License.

"""
  Regridding from one grid to another.
"""

import numpy as np
from numpy.lib.stride_tricks import as_strided
from numba import jit
import scipy.interpolate
import scipy.ndimage


def hinterp(data, x, y, xout, yout, grid=True, **kargs):
    """
    Regridding multiple dimensions data. Interpolation occurs
    in the 2 rightest most indices of grid data.

    :param grid: input grid, multiple array.
    :param x: input grid x coordinates, 1D vector, must be increase order.
    :param y: input grid y coordinates, 1D vector, must be increase order.
    :param xout: output points x coordinates, 1D vector.
    :param yout: output points y coordinates, 1D vector.
    :param grid: output points is a grid.
    :param kargs: keyword arguments for np.interp.
    :return: interpolated grid.

    :Example:
    >>> data = np.arange(40).reshape(2,5,4)
    >>> x = np.linspace(0,9,4)
    >>> y = np.linspace(0,8,5)
    >>> xout = np.linspace(0,9,7)
    >>> yout = np.linspace(0,8,9)
    >>> odata = hinterp(data, x, y, xout, yout)

    """

    # check grid
    if grid:
        xxout, yyout = np.meshgrid(xout, yout)
    else:
        xxout = xout
        yyout = yout

    # interpolated location
    xx = np.interp(xxout, x, np.arange(len(x), dtype=np.float), **kargs)
    yy = np.interp(yyout, y, np.arange(len(y), dtype=np.float), **kargs)

    # perform bilinear interpolation
    xx0 = np.floor(xx).astype(int)
    xx1 = xx0 + 1
    yy0 = np.floor(yy).astype(int)
    yy1 = yy0 + 1

    ixx0 = np.clip(xx0, 0, data.shape[-1] - 1)
    ixx1 = np.clip(xx1, 0, data.shape[-1] - 1)
    iyy0 = np.clip(yy0, 0, data.shape[-2] - 1)
    iyy1 = np.clip(yy1, 0, data.shape[-2] - 1)

    Ia = data[..., iyy0, ixx0]
    Ib = data[..., iyy1, ixx0]
    Ic = data[..., iyy0, ixx1]
    Id = data[..., iyy1, ixx1]

    wa = (xx1 - xx) * (yy1 - yy)
    wb = (xx1 - xx) * (yy - yy0)
    wc = (xx - xx0) * (yy1 - yy)
    wd = (xx - xx0) * (yy - yy0)

    return wa * Ia + wb * Ib + wc * Ic + wd * Id


def rebin(a, factor, func=None):
    """Aggregate data from the input array ``a`` into rectangular tiles.

    The output array results from tiling ``a`` and applying `func` to
    each tile. ``factor`` specifies the size of the tiles. More
    precisely, the returned array ``out`` is such that::

        out[i0, i1, ...] = func(a[f0*i0:f0*(i0+1), f1*i1:f1*(i1+1), ...])

    If ``factor`` is an integer-like scalar, then
    ``f0 = f1 = ... = factor`` in the above formula. If ``factor`` is a
    sequence of integer-like scalars, then ``f0 = factor[0]``,
    ``f1 = factor[1]``, ... and the length of ``factor`` must equal the
    number of dimensions of ``a``.

    The reduction function ``func`` must accept an ``axis`` argument.
    Examples of such function are

      - ``numpy.mean`` (default),
      - ``numpy.sum``,
      - ``numpy.product``,
      - ...

    The following example shows how a (4, 6) array is reduced to a
    (2, 2) array

    >>> import numpy as np
    >>> a = np.arange(24).reshape(4, 6)
    >>> rebin(a, factor=(2, 3), func=np.sum)
    array([[ 24,  42],
           [ 96, 114]])

    If the elements of `factor` are not integer multiples of the
    dimensions of `a`, the remaining cells are discarded.

    >>> rebin(a, factor=(2, 2), func=np.sum)
    array([[16, 24, 32],
           [72, 80, 88]])

    """

    a = np.asarray(a)
    dim = a.ndim
    if np.isscalar(factor):
        factor = dim*(factor,)
    elif len(factor) != dim:
        raise ValueError('length of factor must be {} (was {})'
                         .format(dim, len(factor)))
    if func is None:
        func = np.mean
    for f in factor:
        if f != int(f):
            raise ValueError('factor must be an int or a tuple of ints '
                             '(got {})'.format(f))

    new_shape = [n//f for n, f in zip(a.shape, factor)]+list(factor)
    new_strides = [s*f for s, f in zip(a.strides, factor)]+list(a.strides)
    aa = as_strided(a, shape=new_shape, strides=new_strides)
    return func(aa, axis=tuple(range(-dim, 0)))


def congrid(a, newdims, method='linear', centre=False, minusone=False):
    """
    Arbitrary resampling of source array to new dimension sizes.
    Currently only supports maintaining the same number of dimensions.
    To use 1-D arrays, first promote them to shape (x,1).

    Uses the same parameters and creates the same co-ordinate lookup points
    as IDL''s congrid routine, which apparently originally came from a VAX/VMS
    routine of the same name.

    http://scipy-cookbook.readthedocs.io/items/Rebinning.html

    :param a:
    :param newdims:
    :param method: neighbour - closest value from original data
                   nearest and linear - uses n x 1-D interpolations using
                                        scipy.interpolate.interp1d
                   (see Numerical Recipes for validity of
                    use of n 1-D interpolations)
                   spline - uses ndimage.map_coordinates
    :param centre: True - interpolation points are at the centres of the bins
                   False - points are at the front edge of the bin
    :param minusone:
        For example- inarray.shape = (i,j) & new dimensions = (x,y)
        False - inarray is resampled by factors of (i/x) * (j/y)
        True - inarray is resampled by(i-1)/(x-1) * (j-1)/(y-1)
        This prevents extrapolation one element beyond bounds of input array.
    :return:
    """

    if not (a.dtype in [np.float64, np.float32]):
        a = np.cast[float](a)

    m1 = np.cast[int](minusone)
    ofs = np.cast[int](centre) * 0.5
    old = np.array(a.shape)
    ndims = len(a.shape)
    if len(newdims) != ndims:
        print("[congrid] dimensions error. "
              "This routine currently only support "
              "rebinning to the same number of dimensions.")
        return None
    newdims = np.asarray(newdims, dtype=float)
    dimlist = []

    if method == 'neighbour':
        for i in range(ndims):
            base = np.indices(newdims)[i]
            dimlist.append(
                (old[i] - m1) / (newdims[i] - m1) * (base + ofs) - ofs)
        cd = np.array(dimlist).round().astype(int)
        newa = a[list(cd)]
        return newa

    elif method in ['nearest', 'linear']:
        # calculate new dims
        for i in range(ndims):
            base = np.arange(newdims[i])
            dimlist.append(
                (old[i] - m1) / (newdims[i] - m1) * (base + ofs) - ofs)
        # specify old dims
        olddims = [np.arange(i, dtype=np.float) for i in list(a.shape)]

        # first interpolation - for ndims = any
        mint = scipy.interpolate.interp1d(olddims[-1], a, kind=method)
        newa = mint(dimlist[-1])

        trorder = [ndims - 1] + np.arange(ndims - 1)
        for i in range(ndims - 2, -1, -1):
            newa = newa.transpose(trorder)

            mint = scipy.interpolate.interp1d(olddims[i], newa, kind=method)
            newa = mint(dimlist[i])

        if ndims > 1:
            # need one more transpose to return to original dimensions
            newa = newa.transpose(trorder)

        return newa
    elif method in ['spline']:
        nslices = [slice(0, j) for j in list(newdims)]
        newcoords = np.mgrid[nslices]

        newcoords_dims = np.arange(np.rank(newcoords))
        # make first index last
        newcoords_dims.append(newcoords_dims.pop(0))
        newcoords_tr = newcoords.transpose(newcoords_dims)
        # makes a view that affects newcoords
        newcoords_tr += ofs

        deltas = (np.asarray(old) - m1) / (newdims - m1)
        newcoords_tr *= deltas

        newcoords_tr -= ofs

        newa = scipy.ndimage.map_coordinates(a, newcoords)
        return newa
    else:
        print("Congrid error: Unrecognized interpolation type.\n",
              "Currently only \'neighbour\', \'nearest\',\'linear\',",
              "and \'spline\' are supported.")
        return None


@jit
def box_average(field, lon, lat, olon, olat, width=None, rm_nan=False):
    """
    Remap high resolution field to coarse with box_average.
    Accelerated by numba, but slower than rebin.

    :param field: 2D array (nlat, nlon) for input high resolution.
    :param lon: 1D array, field longitude coordinates.
    :param lat: 1D array, field latitude coordinates.
    :param olon: 1D array, out coarse field longitude coordinates.
    :param olat: 1D array, out coarse field latitude coordinates.
    :param width: box width.
    :param rm_nan: remove nan values when calculate average.
    :return: 2D array (nlat1, nlon1) for coarse field.
    """

    # check box width
    if width is None:
        width = (olon[1] - olon[0]) / 2.0

    # define output grid
    out_field = np.full((olat.size, olon.size), np.nan)

    # loop every grid point
    for j in np.arange(olat.size):
        for i in np.arange(olon.size):
            # searchsorted is fast.
            lon_min = np.searchsorted(lon, olon[i] - width)
            lon_max = np.searchsorted(lon, olon[i] + width) + 1
            lat_min = np.searchsorted(lat, olat[j] - width)
            lat_max = np.searchsorted(lat, olat[j] + width) + 1
            temp = field[lat_min:lat_max, lon_min:lon_max]
            if rm_nan:
                if np.all(np.isnan(temp)):
                    out_field[j, i] = np.nan
                else:
                    temp = temp[~np.isnan(temp)]
                    out_field[j, i] = np.mean(temp)
            else:
                out_field[j, i] = np.mean(temp)

    # return
    return out_field


@jit
def box_max_avg(field, lon, lat, olon, olat,
                width=None, number=1, rm_nan=False):
    """
    Remap high resolution field to coarse with box_max_avg.
    Same as box_avg, but average the "number" hightest values.
    Accelerated by numba, but slower than rebin.

    :param field: 2D array (nlat, nlon) for input high resolution.
    :param lon: 1D array, field longitude coordinates.
    :param lat: 1D array, field latitude coordinates.
    :param olon: 1D array, out coarse field longitude coordinates.
    :param olat: 1D array, out coarse field latitude coordinates.
    :param width: box width.
    :param number: select the number of largest value.
    :param rm_nan: remove nan values when calculate average.
    :return: 2D array (nlat1, nlon1) for coarse field.
    """

    # check box width
    if width is None:
        width = (olon[1] - olon[0]) / 2.0

    # define output grid
    out_field = np.full((olat.size, olon.size), np.nan)

    # loop every grid point
    for j in np.arange(olat.size):
        for i in np.arange(olon.size):
            # searchsorted is fast.
            lon_min = np.searchsorted(lon, olon[i] - width)
            lon_max = np.searchsorted(lon, olon[i] + width)
            lat_min = np.searchsorted(lat, olat[j] - width)
            lat_max = np.searchsorted(lat, olat[j] + width)
            temp = field[lat_min:lat_max, lon_min:lon_max]
            if rm_nan:
                if np.all(np.isnan(temp)):
                    out_field[j, i] = np.nan
                else:
                    temp = temp[~np.isnan(temp)]
                    if temp.size < number:
                        out_field[j, i] = np.mean(temp)
                    else:
                        out_field[j, i] = np.mean(np.sort(temp)[-number:])
            else:
                if temp.size < number:
                    out_field[j, i] = np.mean(temp)
                else:
                    out_field[j, i] = np.mean(np.sort(temp)[-number:])

    # return
    return out_field
