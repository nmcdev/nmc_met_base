# _*_ coding: utf-8 _*_

# Copyright (c) 2019 NMC Developers.
# Distributed under the terms of the GPL V3 License.

"""
  Array manipulating functions.
"""

import numpy as np
import xarray as xr
from typing import Union, Callable


def conform_dims(dims, r, ndim):
    """
    Expands an array or scalar so that it conforms to the shape of
      the given dimension sizes.

    :param dims: An array of dimension sizes of which r will be conformed to.
    :param r: An numpy array whose dimensions must be a subset of dims.
    :param ndim: An array of dimension indexes to indicate which dimension
                 sizes indicated by dims match the dimensions in r.
    :return: This function will create a new variable that has dimensions dims
             and the same type as r.
             The values of r will be copied to all of the other dimensions.

    :Example:
    >>> x = np.arange(12).reshape(3,4)
    >>> x = conform_dims([2,3,5,4,2],x,[1,3])
    >>> print(x.shape)
    (2, 3, 5, 4, 2)
    """

    # reshape r to conform the number of dimension
    sz = r.shape
    cdim = np.ones(len(dims), dtype=np.int)
    cdim[ndim] = sz
    rr = np.reshape(r, cdim)

    # repeat r to conform the dimension
    for i, item in enumerate(dims):
        if cdim[i] == 1:
            rr = np.repeat(rr, item, axis=i)

    # return
    return rr


def unshape(a):
    """
    Convert multiple dimension array to 2d array, keep 1st dim unchanged.

    :param a: array_like, > 2D.
    :return: 2d array, old shape

    >>> a = np.arange(40).reshape(2,4,5)
    >>> a.shape
     (2, 4, 5)
    >>> b, oldshape = tools.unshape(a)
    >>> b.shape
     (2, 20)
    >>> c = tools.deunshape(b, oldshape)
    >>> c.shape
     (2, 4, 5)

    """

    if np.ndim(a) < 2:
        raise ValueError("a must be at least 2 dimension")

    oldshape = a.shape
    array2d = a.reshape(oldshape[0], -1)
    return array2d, oldshape


def deunshape(a, oldshape):
    """
    restore a to old shape.

    :param a: array_like
    :param oldshape: return array shape
    :return: ndarray

    :example:
     >>> a = np.arange(40).reshape(2,4,5)
     >>> a.shape
     (2, 4, 5)
     >>> b, oldshape = unshape(a)
     >>> b.shape
     (2, 20)
     >>> c = deunshape(b, oldshape)
     >>> c.shape
     (2, 4, 5)
    """

    arraynd = a.reshape(oldshape)
    return arraynd


def expand(a, ndim, axis=0):
    """
    expand 1D array to ndim array.

    :param a: 1D array_like
    :param ndim: number of dimensions
    :param axis: position of 1D array
    :return: narray.

    :Example:
     >>> x = np.array([1, 2, 3])
     >>> y = expand(x, 3, axis=1)
     >>> y.shape
     (1, 3, 1)
     >>> y
     array([[[1],
             [2],
             [3]]])
    """

    if axis < 0:
        axis = ndim + axis
    res = np.asarray(a)
    if res.ndim != 1:
        raise ValueError("input array must be one dimensional array")
    idx = [x for x in range(ndim)]
    idx.remove(axis)
    for i in idx:
        res = np.expand_dims(res, axis=i)
    return res


def mrollaxis(a, axis, start=0):
    """
    numpy.rollaxis 's MaskedArray version.

    :param a: array_like
    :param axis: moved axis
    :param start: moved start position.
    :return: ndarray
    """

    if not hasattr(a, 'mask'):
        return np.rollaxis(a, axis, start=start)
    else:
        mask = np.ma.getmaskarray(a)
        data = np.ma.getdata(a)
        mask = np.rollaxis(mask, axis, start=start)
        data = np.rollaxis(data, axis, start=start)
        out = np.ma.asarray(data)
        out.mask = mask
        return out


def scale_vector(in_vector, min_range, max_range,
                 vector_min=None, vector_max=None):
    """
    This is a utility routine to scale the elements of
    a vector or an array into a given data range. nan values is not changed.

    :param in_vector: The input vector or array to be scaled.
    :param min_range: The minimum output value of the scaled vector.
    :param max_range: The maximum output value of the scaled vector.
    :param vector_min: Set this value to the minimum value of the vector,
                       before scaling (vector_min < vector).
                       The default value is Min(vector).
    :param vector_max: Set this value to the maximum value of the vector,
                       before scaling (vector_max < maxvalue).
                       The default value is Max(vector).
    :return: A vector or array of the same size as the input,
             scaled into the data range given by `min_range` and
             `max_range'. The input vector is confined to the data
             range set by `vector_min` and `vector_max` before
             scaling occurs.
    """

    # make sure numpy array
    vector = np.array(in_vector)

    # check keyword parameters
    if vector_min is None:
        vector_min = np.nanmin(vector)
    if vector_max is None:
        vector_max = np.nanmax(vector)

    # Calculate the scaling factors
    scale_factor = [(
        (min_range * vector_max) -
        (max_range * vector_min)) / (vector_max - vector_min),
        (max_range - min_range) / (vector_max - vector_min)]

    # return the scaled vector
    return vector * scale_factor[1] + scale_factor[0]


def matching(in_a, in_b, nan=True):
    """
    Keeping array a's values with b's sort.

    :param in_a: nd array.
    :param in_b: nd array.
    :param nan: do not involve nan values.
    :return: the same length a array.

    :Examples:
    >>> aa = np.array([3, 4, 2, 10, 7, 3, 6])
    >>> bb = np.array([ 5,  7,  3, 9, 6, np.nan, 11])
    >>> print(matching(aa, bb))
    """
    a = in_a.flatten()
    b = in_b.flatten()
    if nan:
        index = np.logical_and(np.isfinite(a), np.isfinite(b))
        a[index][np.argsort(b[index])] = np.sort(a[index])
    else:
        a[np.argsort(b)] = np.sort(a)
    a.shape = in_a.shape
    return a


def plug_array(small,small_lat,small_lon,large,large_lat,large_lon):
    """
    Plug a small array into a large array, assuming they have the same lat/lon
    resolution.
    
    Args:
        small ([type]): 2D array to be inserted into "large"
        small_lat ([type]): 1D array of lats
        small_lon ([type]): 1D array of lons
        large ([type]): 2D array for "small" to be inserted into
        large_lat ([type]): 1D array of lats
        large_lon ([type]): 1D array of lons
    """
    
    small_minlat = min(small_lat)
    small_maxlat = max(small_lat)
    small_minlon = min(small_lon)
    small_maxlon = max(small_lon)
    
    if small_minlat in large_lat:
        minlat = np.where(large_lat==small_minlat)[0][0]
    else:
        minlat = min(large_lat)
    if small_maxlat in large_lat:
        maxlat = np.where(large_lat==small_maxlat)[0][0]
    else:
        maxlat = max(large_lat)
    if small_minlon in large_lon:
        minlon = np.where(large_lon==small_minlon)[0][0]
    else:
        minlon = min(large_lon)
    if small_maxlon in large_lon:
        maxlon = np.where(large_lon==small_maxlon)[0][0]
    else:
        maxlon = max(large_lon)
    
    large[minlat:maxlat+1,minlon:maxlon+1] = small
    
    return large


def filter_numeric_nans(data,thresh, repl_val, high_or_low) :
    """
    Filter numerical nans above or below a specified value''

    Args:
        data ([type]): array to filter '''
        thresh ([type]):  threshold value to filter above or below '''
        repl_val ([type]): replacement value'''
        high_or_low ([type]): [description]
    """
    
    dimens = np.shape(data)    
    temp = np.reshape(data,np.prod(np.size(data)), 1)    
    if high_or_low=='high':        	
	    inds = np.argwhere(temp > thresh) 	
	    temp[inds] = repl_val	  
    elif high_or_low=='low':    
        inds = np.argwhere(temp < thresh) 
        temp[inds] = repl_val	  
    elif high_or_low =='both':
       	inds = np.argwhere(temp > thresh) 	
        temp[inds] = repl_val
        del inds
        inds = np.argwhere(temp < -thresh) 	
        temp[inds] = -repl_val	                 
    else:
        inds = np.argwhere(temp > thresh) 
        temp[inds] = repl_val	  
    
    # Turn vector back into array
    data = np.reshape(temp,dimens,order='F').copy()
    
    return data    


def find_nearest_index(array,val):
    # Return the index of the value closest to the one passed in the array
    return np.abs(array - val).argmin()


def find_nearest_value(array,val):
    # Return the value closest to the one passed in the array
    return array[np.abs(array - val).argmin()]


def check_xarray(arr):
    """
    Check if the passed array is an xarray dataaray by a simple try & except block.
    https://github.com/tomerburg/metlib/blob/master/diagnostics/met_functions.py
    
    Returns:
        Returns 0 if false, 1 if true.
    """
    try:
        temp_val = arr.values
        return 1
    except:
        return 0


def data_array_or_dataset_var(X: Union[xr.DataArray, xr.Dataset], var=None) -> xr.DataArray:
    """
    refer to https://github.com/bgroenks96/pyclimdex/blob/master/climdex/utils.py
    
    If X is a Dataset, selects variable 'var' from X and returns the corresponding
    DataArray. If X is already a DataArray, returns X unchanged.
    """
    if isinstance(X, xr.Dataset):
        assert var is not None, 'var name must be supplied for Dataset input'
        return X[var]
    elif isinstance(X, xr.DataArray):
        return X
    else:
        raise Exception('unrecognized data type: {}'.format(type(X)))

