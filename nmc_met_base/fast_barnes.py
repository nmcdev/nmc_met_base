
# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# Copyright (c) 2022 MeteoSwiss, Bruno Zuercher.
# Published under the BSD-3-Clause license.
#------------------------------------------------------------------------------

# 2023/02/25, from https://github.com/MeteoSwiss/fast-barnes-py

from math import log, cos, sin, tan, pi, sqrt, exp
import numpy as np
from numba import njit


"""
A hyper-fast kd-tree implementation tailored for its application in 'radius' Barnes
interpolation algorithm and thus providing only radius-search functionality.
Uses Numba JIT compiler and numpy arrays to achieve the best performance with the
downside that Python class formalism cannot be used and has to be emulated using
data containers and 'external' functions acting on them.

Assuming the sample points are given by the array `pts`, repeated radius searches on
the corresponding kd-tree can be conducted in this way:

    
# create kd-tree 'instance'
kd_tree = create_kdtree(pts)

# create kd-tree search 'object'
radius = 12.5
kd_tree_search = prepare_search(radius, *kd_tree)

# extract array indices and their distances from returned tuple
res_index, res_sqr_dist, _, _, _ = kd_tree_search

# perform searches using search 'object'
search_pts = np.asarray([[3.2, 7.1], [9.8, -1.6], ...])
for pt in serach_pts:
    radius_search(pt, *kd_tree_search)
    
    # handle results
    for k in range(res_index[-1]):
        # do something with `res_index[k]` and `res_sqr_dist[k]`
        ...
    

Created on Sat May 28 15:35:23 2022
@author: Bruno Zürcher
"""


###############################################################################

@njit
def create_kdtree(pts):
    """
    Creates a balanced kd-tree 'instance' given the N point coordinates `pts`.
    
    In absence of a class concept supported by Numba, the kd-tree 'instance' merely
    consists of a tuple containing the tree structure and the points, which were
    passed to construct it.
    
    The tree is described by N nodes, which themselves are simple integer arrays of
    length 3. Array element 0 contains the index of the root node of the left
    sub-tree, element 1 the index of the root node of the right sub-tree and
    element 2 the index of the parent node. No node is represented by -1.
    The index of the root node of the whole kd-tree is stored in the additionally
    appended N+1-th array element.

    Parameters
    ----------
    pts : numpy ndarray
        A 2-dimensional array of size N x 2 containing the x- and y-coordinates
        (or if you like the longitude/latitude) of the N sample points.

    Returns
    -------
    tree : numpy ndarray
        A (N+1) x 3 integer array contiaining the index description of the kd-tree.
    pts : numpy ndarray
        The original point array that was used to construct the kd-tree.
    """

    num = len(pts)
    index_map = np.arange(num+1)
    tree = np.full((num+1,3), -1, dtype=np.int32)

    root = _median_sort(pts, index_map, 0, 0, num, tree)

    # reorder tree using reverse mapping
    tree[index_map] = tree.copy()
    # map tree entries using forward map; but first map -1 to -1
    index_map[-1] = -1
    for k in range(num):
        for i in range(3):
            tree[k,i] = index_map[tree[k,i]]

    # map also root index
    root = index_map[root]
    # and store root index in tree
    tree[-1,0] = root

    return tree, pts


@njit
def _median_sort(pts, index_map, cInd, frm, to, tree):
    """
    Determines median node by using "select median" algorithm, which establishes
    only partial sort.
    """
    if to-frm == 1:
        return frm
    
    # first sort specified array range with respect to coordinate index
    _partial_sort(pts, index_map, cInd, frm, to)
    median_ind = (frm + to) // 2
    median_pivot = pts[index_map[median_ind], cInd]
    # find 'left-most' node with same median value
    while median_ind > frm and pts[index_map[median_ind-1], cInd] == median_pivot:
        median_ind -= 1

    # recursively call median sort of left and right part
    if frm != median_ind:
        left_median = _median_sort(pts, index_map, 1-cInd, frm, median_ind, tree)
        tree[median_ind,0] = left_median
        tree[left_median,2] = median_ind

    if median_ind+1 != to:
        right_median = _median_sort(pts, index_map, 1-cInd, median_ind+1, to, tree)
        tree[median_ind,1] = right_median
        tree[right_median,2] = median_ind

    return median_ind


@njit
def _partial_sort(pts, index_map, c_ind, frm, to):
    """
    Partially sorts the given array by splitting it in an left sub-array which contains all
    elements smaller-equal than the median and a right sub-array which contains all elements
    greater-equal than the median.
    By construction, it is ensured that all elements from the left sub-array (only!), which are
    equal to the median, occur at the very end of the sub-array, just neighboring the median
    element.
    """
    # find median with adapted Hoare's and Wirth's method
    median_ind = (frm + to) // 2
    left = frm
    right = to - 1
    while left < right:
        # extract pivot value
        median_pivot = pts[index_map[median_ind], c_ind]
        # swap pivot node to beginning of relevant range
        h = index_map[left]
        index_map[left] = index_map[median_ind]
        index_map[median_ind] = h
        i = left + 1
        j = right
        while i <= j:
            # invariant: for all r with left+1 <= r < i: arr[r] < median_pivot
            #        and for all s with j < s <= right: median_pivot <= arr[s]
            while i <= right and pts[index_map[i], c_ind] < median_pivot:  i += 1
            # now holds: either i > right or median_pivot <= arr[i]
            while j > left and median_pivot <= pts[index_map[j], c_ind]:   j -= 1
            # now holds: either j <= left or arr[j] < median_pivot
            if i < j:
                # i.e. (i <= right and j > left) and (median_pivot <= arr[i] and arr[j] < median_pivot)
                # swap elements
                h = index_map[i]
                index_map[i] = index_map[j]
                index_map[j] = h
                i += 1
                j -= 1
                # invariant is reestablished
        # here we have j+1 == i and invariant, i.e.
        #       for all r with left+1 <= r <= j: arr[r] < median_pivot
        #   and for all s with j < s <= right: median_pivot <= arr[s]

        # reinsert pivot node at its correct place
        h = index_map[left]
        index_map[left] = index_map[j]
        index_map[j] = h

        if j < median_ind:    left = i
        elif j > median_ind:  right = j-1
        else:
            # j == medianIndex, i.e. we actually found median already and have it at the right place and
            # also the correct order of the sub arrays
            break


# -----------------------------------------------------------------------------

def _print_tree(tree, ind, pts):
    """
    Auxiliary function that traverses tree and prints tree nodes in infix order.
    Prints node index, its coordinates and then the indices of left and right child,
    nodes as well as index of parent node.
    """
    if tree[ind,0] >= 0:
        _print_tree(tree, tree[ind,0], pts)
    print('%4d: (%7.2f, %6.2f)  [lft: %4d, rgt: %4d, par: %4d]'
          % (ind, pts[ind,0], pts[ind,1], tree[ind,0], tree[ind,1], tree[ind,2]))
    if tree[ind,1] >= 0:
        _print_tree(tree, tree[ind,1], pts)


# -----------------------------------------------------------------------------

@njit
def prepare_search(radius, tree, pts):
    """
    Creates a radius search 'object' that can be used to retrieve points from
    the kd-tree given by tuple (tree, pts), refer to create_kdtree() function.
    
    This preparation step consists of creating two reusable arrays, that will
    contain the indices of the points that lie within the search radius around
    the search point and their square distances from it.
    
    The resulting tuple contains all information that is required to perform a
    radius search around a specific point.

    Parameters
    ----------
    radius : float
        The search radius to be used (using Euclidean norm).
    tree : numpy ndarray
        The index description of the kd-tree.
    pts : numpy ndarray
        The original point array.

    Returns
    -------
    res_index : numpy ndarray
        Will be used to store the indices of the retrieved point.
    res_sqr_dist : numpd ndarray
        Will be used to store the respective square distances.
    tree : numpy ndarray
        The index description of the kd-tree.
    pts : numpy ndarray
        The original point array.
    radius_sqr : float
        The square of the specified search radius.
    """

    num = len(pts)

    # encode number of valid res_index array elements at index -1
    res_index = np.full(num+1, -1, dtype=np.int32)
    res_sqr_dist = np.full(num, 999999.9, dtype=np.float64)
    
    radius_sqr = radius**2

    return res_index, res_sqr_dist, tree, pts, radius_sqr


# -----------------------------------------------------------------------------

@njit
def radius_search(search_pt, res_index, res_sqr_dist, tree, pts, radius_sqr):
    """
    Performs a radius search around the point `search_pt`. The remaining arguments
    that are passed to this function is the data tuple returned from prepare_search().
    The retrieved sample points are in general collected in an unordered way.
    The number of valid array entries is stored in array element `res_index[-1]`.
    
    The results are stored in the reusable arrays `res_index` and `res_sqr_dist` and
    thus are only valid before the next invocation of radius_search().
    
    Take care when using radius_search() in a multithreaded environment. Nevertheless,
    if each thread uses its own radius search 'object' (based on the same underlying
    kd-tree), parallel processing should be possible.

    Parameters
    ----------
    search_pt : numpy ndarray
        The coordinates of the search point.
    res_index : numpy ndarray
        The index array that specifies the points that lie within the search radius.
    res_sqr_dist : numpy ndarray
        The corresponding square distance to the search point.
    tree : numpy ndarray
        The index description of the kd-tree.
    pts : numpy ndarray
        The original point array.
    radius_sqr : float
        The square of the specified search radius.

    Returns
    -------
    None.
    """

    # reset number of valid res_index array elements
    res_index[-1] = 0
    _do_radius_search(tree, pts, radius_sqr, res_index, res_sqr_dist, tree[-1,0], search_pt, 0)


@njit
def _do_radius_search(tree, pts, radius_sqr, res_index, res_sqr_dist, node_ind, coor, c_ind):
    """ The recursive kd-tree radius search implementation. """
    if coor[c_ind] < pts[node_ind,c_ind]:
        # go to the left side
        if tree[node_ind,0] >= 0:
            _do_radius_search(tree, pts, radius_sqr, res_index, res_sqr_dist, tree[node_ind,0], coor, 1-c_ind)

        # check whether further tests are required
        if (coor[c_ind] - pts[node_ind,c_ind])**2 <= radius_sqr:
            # check this node against search radius
            sqr_dist = (coor[0]-pts[node_ind,0])**2 + (coor[1]-pts[node_ind,1])**2
            if sqr_dist <= radius_sqr:
                _append(res_sqr_dist, sqr_dist, res_index, node_ind)

            # check also nodes on the right side of the hyperplane
            if tree[node_ind,1] >= 0:
                _do_radius_search(tree, pts, radius_sqr, res_index, res_sqr_dist, tree[node_ind,1], coor, 1-c_ind)

    else:
        # go the the right side
        if tree[node_ind,1] >= 0:
            _do_radius_search(tree, pts, radius_sqr, res_index, res_sqr_dist, tree[node_ind,1], coor, 1-c_ind)

        # check whether further tests are required
        if (coor[c_ind] - pts[node_ind,c_ind])**2 <= radius_sqr:
            # check this node against search radius
            sqr_dist = (coor[0]-pts[node_ind,0])**2 + (coor[1]-pts[node_ind,1])**2
            if sqr_dist <= radius_sqr:
                _append(res_sqr_dist, sqr_dist, res_index, node_ind)

            # check also nodes on the left side of the hyperplane
            if tree[node_ind,0] >= 0:
                _do_radius_search(tree, pts, radius_sqr, res_index, res_sqr_dist, tree[node_ind,0], coor, 1-c_ind)


@njit
def _append(res_sqr_dist, sqr_dist, res_index, node_ind):
    """
    Appends the sample point with index `node_ind` that has a distance of `sqr_dist`
    from the search point to the result arrays.
    """
    cur_len = res_index[-1]
    res_sqr_dist[cur_len] = sqr_dist
    res_index[cur_len] = node_ind
    # increase number of valid elements
    res_index[-1] += 1


###############################################################################

"""
Implements Lambert conformal conic projection.

Uses Numba JIT compiler and numpy arrays to achieve the best performance with the
downside that Python class formalism cannot be used and has to be emulated using
data containers and 'external' functions acting on them.

Assuming the lon/lat-coordinates of the points to be transformed are given by the
array `pts`, the corresponding Lambert map coordinates can be computed by:


# create Lambert projection 'instance'
lambert_proj = lambert_conformal.create_proj(11.5, 34.5, 42.5, 65.5)

# map lonlat sample point coordinates to Lambert coordinate space
pts = np.asarray([[8.55, 47.37], [13.41, 52.52], ...])
lam_pts = lambert_conformal.to_map(pts, pts.copy(), *lambert_proj)


The reverse transform can be invoked by calling lambert_conformal.to_geo().

Refer to Snyder J. (1987), Map Projections: A Working Manual, US Geological
Survey Professional Paper 1395, US Government Printing Office, Washington.    


Created on Sat Jun  4 13:18:04 2022
@author: Bruno Zürcher
"""

RAD_PER_DEGREE = pi / 180.0
HALF_RAD_PER_DEGREE = RAD_PER_DEGREE / 2.0


@njit
def create_proj(center_lon, center_lat, lat1, lat2):
    """
    Creates a Lambert conformal projection 'instance' using the specified parameters.
    
    In absence of a class concept supported by Numba, the projection 'instance'
    merely consists of a tuple containing the relevant Lambert projection constants.

    Parameters
    ----------
    center_lon : float
        The center meridian, which points in the upward direction of the map [°E].
    center_lat : float
        The center parallel, which defines together with the center meridian
        the origin of the map [°N].
    lat1 : float
        The latitude of the first (northern) standard parallel [°N].
    lat2 : float
        The latitude of the second (southern) standard parallel [°S].

    Returns
    -------
    center_lon : float
        The center meridian, as above.
    n : float
        The cone constant, i.e. the ratio of the angle between meridians to the true
        angle, as described in Snider.
    n_inv : float
        The inverse of n, as described in Snider.
    F : float
        A constant used for mapping, as described in Snider.
    rho0 : float
        Unscaled distance from the cone tip to the first standard parallel, as
        described in Snider.
    """

    if lat1 != lat2:
        n = log(cos(lat1 *RAD_PER_DEGREE) / cos(lat2 *RAD_PER_DEGREE)) / log(tan((90.0+ lat2)*HALF_RAD_PER_DEGREE) / tan((90.0+ lat1)*HALF_RAD_PER_DEGREE))
    else:
        n = sin(lat1 * RAD_PER_DEGREE)
        
    n_inv = 1.0 / n
    F = cos(lat1 * RAD_PER_DEGREE) * tan((90.0+lat1)*HALF_RAD_PER_DEGREE) ** n / n
    rho0 = F / tan((90.0+center_lat)*HALF_RAD_PER_DEGREE) ** n
    
    return (center_lon, n, n_inv, F, rho0)


# -----------------------------------------------------------------------------

@njit
def get_scale(lat, center_lon, n, n_inv, F, rho0):
    """
    Returns the local scale factor for the specified latitude.
    This quantity specifies the degree of length distortion compared to that at
    the standard latitudes of this Lambert conformal projection instance, where
    it is by definition 1.0.
    """
    return F * n / (np.cos(lat*RAD_PER_DEGREE) * np.power(np.tan((90.0+lat)*HALF_RAD_PER_DEGREE), n))


# -----------------------------------------------------------------------------

@njit
def to_map(geoc, mapc, center_lon, n, n_inv, F, rho0):
    """
    Maps the geographic coordinates given by the numpy array `geoc` onto the Lambert
    map given by the tuple `(center_lon, n, n_inv, F, rho0)` and stores the result
    in the preallocated numpy array `mapc`.
    """
    rho = F / np.power(np.tan((90.0+geoc[:,1])*HALF_RAD_PER_DEGREE), n)
    arg = n * (geoc[:,0] - center_lon) * RAD_PER_DEGREE
    mapc[:,0] = rho * np.sin(arg) / RAD_PER_DEGREE
    mapc[:,1] = (rho0 - rho*np.cos(arg)) / RAD_PER_DEGREE
    return mapc


@njit
def to_map2(geox, geoy, mapx, mapy, center_lon, n, n_inv, F, rho0):
    """
    Maps the geographic coordinates given by the separated numpy arrays `geox` and `geoy`
    onto the Lambert map given by the tuple `(center_lon, n, n_inv, F, rho0)` and
    stores the result in the preallocated numpy arrays `mapx` and `mapy`.
    """
    rho = F / np.power(np.tan((90.0+geoy)*HALF_RAD_PER_DEGREE), n)
    arg = n * (geox - center_lon) * RAD_PER_DEGREE
    mapx[:] = rho * np.sin(arg) / RAD_PER_DEGREE
    mapy[:] = (rho0 - rho*np.cos(arg)) / RAD_PER_DEGREE
    
#------------------------------------------------------------------------------

@njit
def to_geo(mapc, geoc, center_lon, n, n_inv, F, rho0):
    """
    Maps the Lambert map coordinates given by the numpy array `mapc` to the
    geographic coordinate system and stores the result in the preallocated numpy
    array `geoc`. The Lambert projection is given by the tuple
    `(center_lon, n, n_inv, F, rho0)`.
    """
    x = mapc[:,0] * RAD_PER_DEGREE
    y = mapc[:,1] * RAD_PER_DEGREE
    arg = rho0 - y
    rho = np.sqrt(x**2 + arg**2)
    if (n < 0.0):
        rho = np.negative(rho)
    theta = np.arctan2(x, arg)
    geoc[:,1] = np.arctan(np.power(F/rho, n_inv)) / HALF_RAD_PER_DEGREE - 90.0
    geoc[:,0] = center_lon + theta / n / RAD_PER_DEGREE
    return geoc
    

@njit
def to_geo2(mapx, mapy, geox, geoy, center_lon, n, n_inv, F, rho0):
    """
    Maps the Lambert map coordinates given by the numpy arrays `mapx` and `mapy` to
    the geographic coordinate system and stores the result in the preallocated numpy
    arrays `geox` and `geoy`. The Lambert projection is given by the tuple
    `(center_lon, n, n_inv, F, rho0)`.
    """
    x = mapx * RAD_PER_DEGREE
    arg = rho0 - mapy * RAD_PER_DEGREE
    rho = np.sqrt(x**2 + arg**2)
    if (n < 0.0):
        rho = np.negative(rho)
    theta = np.arctan2(x, arg)
    geoy[:] = np.arctan(np.power(F/rho, n_inv)) / HALF_RAD_PER_DEGREE - 90.0
    geox[:] = center_lon + theta / n / RAD_PER_DEGREE 
    

###############################################################################

"""
Module that provides different Barnes interpolation algorithms that use
the distance metric of the Euclidean plane.
To attain competitive performance, the code is written using Numba's
just-in-time compiler and thus has to use the respective programming idiom,
which is sometimes not straightforward to read at a first glance. Allocated
memory is as far as possible reused in order to reduce the workload imposed
on the garbage collector.

Created on Sat May 14 13:10:47 2022
@author: Bruno Zürcher
"""

def barnes(pts, val, sigma, x0, step, size, method='optimized_convolution',
    num_iter=4, min_weight=0.001):
    """
    Computes the Barnes interpolation for observation values `val` taken at sample
    points `pts` using Gaussian weights for the width parameter `sigma`.
    The underlying grid embedded in a Euclidean space is given with start point
    `x0`, regular grid steps `step` and extension `size`.

    Parameters
    ----------
    pts : numpy ndarray
        A 2-dimensional array of size N x 2 containing the x- and y-coordinates
        (or if you like the longitude/latitude) of the N sample points.
    val : numpy ndarray
        A 1-dimensional array of size N containing the N observation values.
    sigma : float
        The Gaussian width parameter to be used.
    x0 : numpy ndarray
        A 1-dimensional array of size 2 containing the coordinates of the
        start point of the grid to be used.
    step : float
        The regular grid point distance.
    size : tuple of 2 int values
        The extension of the grid in x- and y-direction.
    method : {'optimized_convolution', 'convolution', 'radius', 'naive'}
        Designates the Barnes interpolation method to be used. The possible
        implementations that can be chosen are 'naive' for the straightforward
        implementation (algorithm A from paper), 'radius' to consider only sample
        points within a specific radius of influence, both with an algorithmic
        complexity of O(N x W x H).
        The choice 'convolution' implements algorithm B specified in the paper
        and 'optimized_convolution' is its optimization by appending tail values
        to the rectangular kernel. The latter two algorithms reduce the complexity
        down to O(N + W x H).
        The default is 'optimized_convolution'.
    num_iter : int, optional
        The number of performed self-convolutions of the underlying rect-kernel.
        Applies only if method is 'optimized_convolution' or 'convolution'.
        The default is 4.
    min_weight : float, optional
        Choose radius of influence such that Gaussian weight of considered sample
        points is greater than `min_weight`.
        Applies only if method is 'radius'. Recommended values are 0.001 and less.
        The default is 0.001, which corresponds to a radius of 3.717 * sigma.

    Returns
    -------
    numpy ndarray
        A 2-dimensional array containing the resulting field of the performed
        Barnes interpolation.
    """

    if method == 'optimized_convolution':
        return _interpolate_opt_convol(pts, val, sigma, x0, step, size, num_iter)
        
    elif method == 'convolution':
        return _interpolate_convol(pts, val, sigma, x0, step, size, num_iter)
    
    elif method == 'radius':
        return _interpolate_radius(pts, val, sigma, x0, step, size, min_weight)
        
    elif method == 'naive':
        return _interpolate_naive(pts, val, sigma, x0, step, size)
        
    else:
        raise RuntimeError("encountered invalid Barnes interpolation method: " + method)
    

# -----------------------------------------------------------------------------

@njit
def _normalize_values(val):
    """
    Offsets the observation values such that they are centered over 0.
    """
    offset = (np.amin(val) + np.amax(val)) / 2.0
    # center range of observation values around 0
    val -= offset
    return offset


# -----------------------------------------------------------------------------

@njit
def _inject_data(vg, wg, pts, val, x0, step, size):
    """
    Injects the observations values and weights, respectively, into the
    corresponding fields as described by algorithm B.1.
    """
    for k in range(len(pts)):
        xc = (pts[k,0]-x0[0]) / step
        yc = (pts[k,1]-x0[1]) / step
        if (xc < 0.0 or yc < 0.0 or xc >= size[1]-1 or yc >= size[0]-1):
            continue
        xi = int(xc)
        yi = int(yc)
        xw = xc - xi
        yw = yc - yi
        
        w = (1.0-xw)*(1.0-yw)
        vg[yi, xi] += w*val[k]
        wg[yi, xi] += w

        w =  xw*(1.0-yw)
        vg[yi, xi+1] +=w*val[k]
        wg[yi, xi+1] += w
        
        w = xw*yw
        vg[yi+1, xi+1] += w*val[k]
        wg[yi+1, xi+1] += w
        
        w = (1.0-xw)*yw
        vg[yi+1, xi] += w*val[k]
        wg[yi+1, xi] += w


# -----------------------------------------------------------------------------

@njit
def _interpolate_opt_convol(pts, val, sigma, x0, step, size, num_iter):
    """ 
    Implements algorithm B presented in section 4 of the paper but optimized for
    a rectangular window with a tail value alpha.
    """
    offset = _normalize_values(val)

    # the grid fields to store the convolved values and weights
    vg = np.zeros(size, dtype=np.float64)
    wg = np.zeros(size, dtype=np.float64)
    
    # inject obs values into grid
    _inject_data(vg, wg, pts, val, x0, step, size)
        
    # prepare convolution
    half_kernel_size = get_half_kernel_size_opt(sigma, step, num_iter)
    kernel_size = 2*half_kernel_size + 1
        
    tail_value = get_tail_value(sigma, step, num_iter)
    
    # execute algorithm B
    # convolve rows in x-direction
    h_arr = np.empty(size[1], dtype=np.float64)
    for j in range(size[0]):
        # convolve row values
        vg[j,:] = _accumulate_tail_array(vg[j,:].copy(), h_arr, size[1], kernel_size, num_iter, tail_value)
            
        # convolve row weights
        wg[j,:] = _accumulate_tail_array(wg[j,:].copy(), h_arr, size[1], kernel_size, num_iter, tail_value)
        
    # convolve columns in y- direction
    h_arr = np.empty(size[0], dtype=np.float64)
    for i in range(size[1]):
        # convolve column values
        vg[:,i] = _accumulate_tail_array(vg[:,i].copy(), h_arr, size[0], kernel_size, num_iter, tail_value)
        
        # convolve column weights
        wg[:,i] = _accumulate_tail_array(wg[:,i].copy(), h_arr, size[0], kernel_size, num_iter, tail_value)
        
    # compute limit wg array value for which weight > 0.0022, i.e. grid points with greater distance
    #   than 3.5*sigma will evaluate to NaN
    # since we dropped common factors in our computation, we have to revert their cancellation in the
    #   following computation
    factor = (kernel_size+2*tail_value) ** (2*num_iter) * (step/sigma) ** 2 / 2 / pi * 0.0022

    # set smaller weights to NaN with overall effect that corresponding quotient is NaN, too
    for j in range(size[0]):
        for i in range(size[1]):
            if wg[j,i] < factor: wg[j,i] = np.NaN

    # yet to be considered:
    # - add offset again to resulting quotient
    # - and apply quantization operation:
    #   here by casting double to float and thus drop around 29 least significant bits
    return (vg / wg + offset).astype(np.float32)

    
@njit
def _accumulate_tail_array(in_arr, h_arr, arr_len, rect_len, num_iter, alpha):
    """
    Computes the `num_iter`-fold convolution of the specified 1-dim array ìn_arr`
    with a rect-kernel of length rect_len and tail values `alpha`. To obtain the
    actual convolution with a corresponding uniform distribution, the result would have
    to be scaled with a factor 1/rect_len^num_iter. But this scaling is not implemented,
    since these factors are canceled when the resulting fields are divided with
    each other.
    """
    # the half window size T
    h0 = (rect_len-1) // 2
    h0_1 = h0 + 1
    h1 = rect_len - h0
    for i in range(num_iter):
        # accumulates values under regular part of window (without tails!)
        accu = 0.0
        # phase a: window center still outside array
        # accumulate first h0 elements
        for k in range(-h0, 0):
            accu += in_arr[k+h0]
        # phase b: window center inside array but window does not cover array completely
        # accumulate remaining rect_len elements and write their value into array
        for k in range(0, h1):
            accu += in_arr[k+h0]
            h_arr[k] = accu + alpha*in_arr[k+h0_1]
        # phase c: window completely contained in array
        # add difference of border elements and write value into array
        for k in range(h1, arr_len-h0_1):
            accu += (in_arr[k+h0] - in_arr[k-h1])
            h_arr[k] = accu + alpha*(in_arr[k-h1]+in_arr[k+h0_1])
        # phase c': very last element
        k = arr_len-h0_1
        accu += (in_arr[k+h0] - in_arr[k-h1])
        h_arr[k] = accu + alpha*in_arr[k-h1]
        # phase d (mirroring phase b): window center still inside array but window does not cover array completely
        # de-accumulate elements and write value into array
        for k in range(arr_len-h0, arr_len):
            accu -= in_arr[k-h1]
            h_arr[k] = accu + alpha*in_arr[k-h1]
        # phase e (mirroring phase a): window center left array
        # unnecessary since value is not written

        # h_arr contains convolution result of this pass
        # swap arrays and start over next convolution
        h = in_arr
        in_arr = h_arr
        h_arr = h
    
    return in_arr

    
@njit
def get_half_kernel_size_opt(sigma, step, num_iter):
    """ Computes the half kernel size T for the optimized convolution algorithm. """
    s = sigma / step
    return int((sqrt(1.0+12*s*s/num_iter) - 1) / 2)


@njit
def get_tail_value(sigma, step, num_iter):
    """ Computes the tail value alpha for the optimized convolution algorithm. """
    half_kernel_size = get_half_kernel_size_opt(sigma, step, num_iter)
    kernel_size = 2*half_kernel_size + 1

    sigma_rect_sqr = (half_kernel_size+1)*half_kernel_size/3.0*step**2
    # slightly rearranged expression from equ. (12)
    return  0.5*kernel_size*(sigma**2/num_iter - sigma_rect_sqr) \
        / (((half_kernel_size+1)*step)**2 - sigma**2/num_iter)


# -----------------------------------------------------------------------------

@njit
def _interpolate_convol(pts, val, sigma, x0, step, size, num_iter):
    """ 
    Implements algorithm B presented in section 4 of the paper.
    """
    offset = _normalize_values(val)

    # the grid fields to store the convolved values and weights
    vg = np.zeros(size, dtype=np.float64)
    wg = np.zeros(size, dtype=np.float64)
    
    # inject obs values into grid
    _inject_data(vg, wg, pts, val, x0, step, size)
        
    # prepare convolution
    half_kernel_size = get_half_kernel_size(sigma, step, num_iter)
    kernel_size = 2*half_kernel_size + 1
        
    # execute algorithm B
    # convolve rows in x-direction
    h_arr = np.empty(size[1], dtype=np.float64)
    for j in range(size[0]):
        # convolve row values
        vg[j,:] = _accumulate_array(vg[j,:].copy(), h_arr, size[1], kernel_size, num_iter)
            
        # convolve row weights
        wg[j,:] = _accumulate_array(wg[j,:].copy(), h_arr, size[1], kernel_size, num_iter)
        
    # convolve columns in y- direction
    h_arr = np.empty(size[0], dtype=np.float64)
    for i in range(size[1]):
        # convolve column values
        vg[:,i] = _accumulate_array(vg[:,i].copy(), h_arr, size[0], kernel_size, num_iter)
        
        # convolve column weights
        wg[:,i] = _accumulate_array(wg[:,i].copy(), h_arr, size[0], kernel_size, num_iter)
        
    
    # compute limit wg array value for which weight > 0.0022, i.e. grid points with greater distance
    #   than 3.5*sigma will evaluate to NaN
    # since we dropped common factors in our computation, we have to revert their cancellation in the
    #   following computation
    sigma_eff = get_sigma_effective(sigma, step, num_iter)
    factor = float(kernel_size) ** (2*num_iter) * (step/sigma_eff) ** 2 / 2 / pi * 0.0022
    
    # set smaller weights to NaN with overall effect that corresponding quotient is NaN, too
    for j in range(size[0]):
        for i in range(size[1]):
            if wg[j,i] < factor: wg[j,i] = np.NaN
    
    # yet to be considered:
    # - add offset again to resulting quotient
    # - and apply quantization operation:
    #   here by temporary casting double to float and thus drop around 29 least significant bits
    return (vg / wg + offset).astype(np.float32)


@njit
def _accumulate_array(in_arr, h_arr, arr_len, rect_len, num_iter):
    """
    Computes the `num_iter`-fold convolution of the specified 1-dim array ìn_arr`
    with a rect-kernel of length rect_len. To obtain the actual convolution with
    a corresponding uniform distribution, the result would have to be scaled with
    a factor 1/rect_len^num_iter. But this scaling is not implemented, since these
    factors are canceled when the resulting fields are divided with each other.
    """
    # the half window size T
    h0 = (rect_len-1) // 2
    h1 = rect_len - h0
    for i in range(num_iter):
        # accumulates values under regular part of window (without tails!)
        accu = 0.0
        # phase a: window center still outside array
        # accumulate first h0 elements
        for k in range(-h0, 0):
            accu += in_arr[k+h0]
        # phase b: window center inside array but window does not cover array completely
        # accumulate remaining rect_len elements and write their value into array
        for k in range(0, h1):
            accu += in_arr[k+h0]
            h_arr[k] = accu
        # phase c: window completely contained in array
        # add difference of border elements and write value into array
        for k in range(h1, arr_len-h0):
            accu += (in_arr[k+h0] - in_arr[k-h1])
            h_arr[k] = accu
        # phase d (mirroring phase b): window center still inside array but window does not cover array completely
        # de-accumulate elements and write value into array
        for k in range(arr_len-h0, arr_len):
            accu -= in_arr[k-h1]
            h_arr[k] = accu
        # phase e (mirroring phase a): window center left array
        # unnecessary since value is not written

        # h_arr contains convolution result of this pass
        # swap arrays and start over next convolution
        h = in_arr
        in_arr = h_arr
        h_arr = h
    
    return in_arr


@njit
def get_half_kernel_size(sigma, step, num_iter):
    """ Computes the half kernel size T for the convolution algorithm. """
    return int(sqrt(3.0/num_iter)*sigma/step + 0.5)


@njit
def get_sigma_effective(sigma, step, num_iter):
    """
    Computes the effective variance of the `num_iter`-fold convolved rect-kernel
    of length 2*T+1.
    """
    half_kernel_size = get_half_kernel_size(sigma, step, num_iter)
    return sqrt(num_iter / 3.0 * half_kernel_size*(half_kernel_size+1)) * step
    

# -----------------------------------------------------------------------------

@njit
def _interpolate_radius(pts, val, sigma, x0, step, size, min_weight):
    """ 
    Implements the radius algorithm to compute the Barnes interpolation.
    """
    offset = _normalize_values(val)

    grid_value = np.zeros(size, dtype=np.float64)

    # construct kd-tree 'instance' with given points
    kd_tree = create_kdtree(pts)
    
    # create kd-tree search 'instance'
    search_radius = sqrt(-2.0*log(min_weight)) * sigma
    kd_radius_search = prepare_search(search_radius, *kd_tree)
    # extract array indices and their distances from returned tuple
    res_index, res_sqr_dist, _, _, _ = kd_radius_search
    
    scale = 2*sigma**2
    c = np.empty(2, dtype=np.float64)
    for j in range(size[0]):
        # compute y-coordinate of grid point
        c[1] = x0[1] + j*step
        for i in range(size[1]):
            # compute x-coordinate of grid point
            c[0] = x0[0] + i*step
            
            # loop over all observation points and compute numerator and denominator of equ. (1)
            weighted_sum = 0.0
            weight_total = 0.0
            radius_search(c, *kd_radius_search)
            for k in range(res_index[-1]):
                weight = exp(-res_sqr_dist[k]/scale)
                weighted_sum += weight*val[res_index[k]]
                weight_total += weight
                
            # set grid points with greater distance than 3.5*sigma to NaN, i.e.
            #   points with weight < 0.0022, 
            if weight_total >= 0.0022:
                grid_value[j,i] = weighted_sum / weight_total + offset
            else:
                grid_value[j,i] = np.NaN
            
    return grid_value


# -----------------------------------------------------------------------------

@njit
def _interpolate_naive(pts, val, sigma, x0, step, size):
    """ Implements the naive algorithm A to compute the Barnes interpolation. """
    offset = _normalize_values(val)
    
    grid_value = np.zeros(size, dtype=np.float64)
    
    scale = 2*sigma**2
    for j in range(size[0]):
        # compute y-coordinate of grid point
        yc = x0[1] + j*step
        for i in range(size[1]):
            # compute x-coordinate of grid point
            xc = x0[0] + i*step
            
            # use numpy to directly compute numerator and denominator of equ. (1)
            sqr_dist = (pts[:,0]-xc)**2 + (pts[:,1]-yc)**2
            weight = np.exp(-sqr_dist/scale)
            weighted_sum = np.dot(weight, val)
            weight_total = np.sum(weight)
            
            if weight_total > 0.0:
                grid_value[j,i] = weighted_sum / weight_total + offset
            else:
                grid_value[j,i] = np.NaN
            
    return grid_value


###############################################################################

"""
Module that provides two different Barnes interpolation algorithms acting on
the unit sphere S^2 and thus using the spherical distance metric.
To attain competitive performance, the code is written using Numba's
just-in-time compiler and thus has to use the respective programming idiom,
which is sometimes not straightforward to read at a first glance. Allocated
memory is as far as possible reused in order to reduce the workload imposed
on the garbage collector.

Created on Sat May 14 20:49:17 2022
@author: Bruno Zürcher
"""

def barnes_S2(pts, val, sigma, x0, step, size, method='optimized_convolution', num_iter=4, resample=True):
    """
    Computes the Barnes interpolation for observation values `val` taken at sample
    points `pts` using Gaussian weights for the width parameter `sigma`.
    The underlying grid is embedded on the unit sphere S^2 and thus inherits the
    spherical distance measure (taken in degrees). The grid is given by the start
    point `x0`, regular grid steps `step` and extension `size`.

    Parameters
    ----------
    pts : numpy ndarray
        A 2-dimensional array of size N x 2 containing the x- and y-coordinates
        (or if you like the longitude/latitude) of the N sample points.
    val : numpy ndarray
        A 1-dimensional array of size N containing the N observation values.
    sigma : float
        The Gaussian width parameter to be used.
    x0 : numpy ndarray
        A 1-dimensional array of size 2 containing the coordinates of the
        start point of the grid to be used.
    step : float
        The regular grid point distance.
    size : tuple of 2 int values
        The extension of the grid in x- and y-direction.
    method : {'optimized_convolution_S2', 'naive_S2'}
        Designates the Barnes interpolation method to be used. The possible
        implementations that can be chosen are 'naive_S2' for the straightforward
        implementation (algorithm A from the paper) with an algorithmic complexity
        of O(N x W x H).
        The choice 'optimized_convolution_S2' implements the optimized algorithm B
        specified in the paper by appending tail values to the rectangular kernel.
        The latter algorithm has a reduced complexity of O(N + W x H).
        The default is 'optimized_convolution_S2'.
    num_iter : int, optional
        The number of performed self-convolutions of the underlying rect-kernel.
        Applies only if method is 'optimized_convolution_S2'.
        The default is 4.
    resample : bool, optional
        Specifies whether to resample Lambert grid field to lonlat grid.
        Applies only if method is 'optimized_convolution_S2'.
        The default is True.

    Returns
    -------
    numpy ndarray
        A 2-dimensional array containing the resulting field of the performed
        Barnes interpolation.
    """    

    if method == 'optimized_convolution_S2':
        return _interpolate_opt_convol_S2(pts, val, sigma, x0, step, size, num_iter, resample)
        
    elif method == 'naive_S2':
        return _interpolate_naive_S2(pts, val, sigma, x0, step, size)
        
    else:
        raise RuntimeError("encountered invalid Barnes interpolation method: " + str(method))
    

# -----------------------------------------------------------------------------

@njit
def _interpolate_opt_convol_S2(pts, val, sigma, x0, step, size, num_iter, resample):
    """ 
    Implements the optimized convolution algorithm B for the unit sphere S^2.
    """
    # # the used Lambert projection
    # lambert_proj = get_lambert_proj()
    
    # # the *fixed* grid in Lambert coordinate space
    # lam_x0 = np.asarray([-32.0, -2.0])
    # lam_size = (int(44.0/step), int(64.0/step))
    
    # # map lonlat sample point coordinatess to Lambert coordinate space
    # lam_pts = lambert_conformal.to_map(pts, pts.copy(), *lambert_proj)
    
    # # call ordinary 'optimized_convolution' algorithm
    # lam_field = interpolation._interpolate_opt_convol(lam_pts, val, sigma, lam_x0, step, lam_size, num_iter)
    
    # if resample:
    #     return _resample(lam_field, lam_x0, x0, step, size, *lambert_proj)
    # else:
    #     return lam_field
    
    
    
    # split commented code above in two separately 'measurable' sub-routines
    
    # the convolution part taking place in Lambert space
    res1 = interpolate_opt_convol_S2_part1(pts, val, sigma, x0, step, size, num_iter)
    
    # the resampling part that performs back-projection from Lambert to lonlat space
    if resample:
        return interpolate_opt_convol_S2_part2(*res1)
    else:
        return res1[0]


@njit
def interpolate_opt_convol_S2_part1(pts, val, sigma, x0, step, size, num_iter):
    """ The convolution part of _interpolate_opt_convol_S2(), allowing to measure split times. """
    # the used Lambert projection
    lambert_proj = get_lambert_proj()
    
    # the *fixed* grid in Lambert coordinate space
    lam_x0 = np.asarray([-32.0, -2.0])
    lam_size = (int(44.0/step), int(64.0/step))
    
    # map lonlat sample point coordinates to Lambert coordinate space
    lam_pts = to_map(pts, pts.copy(), *lambert_proj)
    
    # call ordinary 'optimized_convolution' algorithm
    lam_field = _interpolate_opt_convol(lam_pts, val, sigma, lam_x0, step, lam_size, num_iter)

    return (lam_field, lam_x0, x0, step, size, lambert_proj)


@njit
def interpolate_opt_convol_S2_part2(lam_field, lam_x0, x0, step, size, lambert_proj):
    """ The back-projection part of _interpolate_opt_convol_S2(), allowing to measure split times. """
    return _resample(lam_field, lam_x0, x0, step, size, *lambert_proj)

    
@njit
def get_lambert_proj():
    """ Return the Lambert projection that is used for our test example. """
    return create_proj(11.5, 34.5, 42.5, 65.5)

    
@njit
def _resample(lam_field, lam_x0, x0, step, size, center_lon, n, n_inv, F, rho0):
    """ Resamples the Lambert grdi field to the specified lonlat grid. """
    # x-coordinate in lon-lat grid is constant over all grid lines
    geox = np.empty(size[1], dtype=np.float64)
    for i in range(size[1]):
        geox[i] = x0[0] + i*step
        
    # memory for coordinates in Lambert space
    mapx = np.empty(size[1], dtype=np.float64)
    mapy = np.empty(size[1], dtype=np.float64)
    
    # memory for the corresponding Lambert grid indices 
    indx = np.empty(size[1], dtype=np.int32)
    indy = np.empty(size[1], dtype=np.int32)
    
    # memory for the resulting field in lonlat space
    res_field = np.empty(size, dtype=np.float32)
    
    # for each line in lonlat grid 
    for j in range(size[0]):
        # compute corresponding locations in Lambert space
        to_map2(geox, j*step + x0[1], mapx, mapy, center_lon, n, n_inv, F, rho0)
        # compute corresponding Lambert grid indices
        mapx -= lam_x0[0]
        mapx /= step
        mapy -= lam_x0[1]
        mapy /= step
        # the corresponding 'i,j'-integer indices of the lower left grid point
        indx[:] = mapx.astype(np.int32)
        indy[:] = mapy.astype(np.int32)
        # and compute bilinear weights
        mapx -= indx    # contains now the weights
        mapy -= indy    # contains now the weights
        
        # compute bilinear interpolation of the 4 neighboring grid point values 
        for i in range(size[1]):
            res_field[j,i] = (1.0-mapy[i])*(1.0-mapx[i])*lam_field[indy[i],indx[i]] + \
                mapy[i]*(1.0-mapx[i])*lam_field[indy[i]+1,indx[i]] + \
                mapy[i]*mapx[i]*lam_field[indy[i]+1,indx[i]+1] + \
                (1.0-mapy[i])*mapx[i]*lam_field[indy[i],indx[i]+1]
        
    return res_field
    

# -----------------------------------------------------------------------------

@njit
def _interpolate_naive_S2(pts, val, sigma, x0, step, size):
    """ Implements the naive Barnes interpolation algorithm A for the unit sphere S^2. """
    offset = _normalize_values(val)
    
    grid_val = np.zeros(size, dtype=np.float64)
    
    scale = 2*sigma**2
    for j in range(size[0]):
        # compute y-coordinate of grid point
        yc = x0[1] + j*step
        for i in range(size[1]):
            # compute x-coordinate of grid point
            xc = x0[0] + i*step
            
            # use numpy to directly compute numerator and denominator of equ. (1)
            dist = _dist_S2(xc, yc, pts[:,0], pts[:,1])
            weight = np.exp(-dist*dist/scale)
            weighted_sum = np.dot(weight, val)
            weight_total = np.sum(weight)
            
            if weight_total > 0.0:
                grid_val[j,i] = weighted_sum / weight_total + offset
            else:
                grid_val[j,i] = np.NaN
            
    return grid_val


@njit
def _dist_S2(lon0, lat0, lon1, lat1):
    """ Computes spherical distance between the 2 specified points. Input and output in degrees. """
    lat0_rad = lat0 * RAD_PER_DEGREE
    lat1_rad = lat1 * RAD_PER_DEGREE
    arg = np.sin(lat0_rad)*np.sin(lat1_rad) + np.cos(lat0_rad)*np.cos(lat1_rad)*np.cos((lon1-lon0)*RAD_PER_DEGREE)
    arg[arg > 1.0] = 1.0
    return np.arccos(arg) / RAD_PER_DEGREE

