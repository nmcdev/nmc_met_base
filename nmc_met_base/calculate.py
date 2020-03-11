# _*_ coding: utf-8 _*_

# Copyright (c) 2019 NMC Developers.
# Distributed under the terms of the GPL V3 License.

"""
  Calculating functions.
"""

import numpy as np
from nmc_met_base import arr, constants


def center_finite_diff_n(grid, dim=1, r=None, map_scale=None,
                         cyclic=False, second=False):
    """
    Performs a centered finite difference operation on the given dimension.

    using:
    Central finite difference scheme second order for first derivatives
      (u[i+1]-u[i-1])/(2dx)
    Central finite difference scheme second order for second derivatives
      (u[i+1]+u[i-1]-2*u[i])/(dx*dx)

    reference:
    http://www.cfm.brown.edu/people/jansh/resources/APMA1180/fd.pdf

    notice: for second derivatives, ensure equal interval.

    :param grid: a multi-dimensional numpy array.
    :param r: A scalar, one-dimensional, or multi-dimensional array containing
              the coordinates along which grid is to be difference. Does need
              not be equally spaced from a computational point of view.
                  >scalar: r assumed to be the (constant) distance between
                           adjacent points.
                  >one-dimensional (and the same size as the dimension of
                           grid): applied to all dimensions of grid.
                  >multi-dimensional: then it must be the same size as grid.
    :param dim: A scalar integer indicating which dimension of grid to
                calculate the center finite difference on.
                Dimension numbering starts at 1, default=1.
    :param map_scale:  map scale coefficient, a scalar, one-dimensional,
                       or multi-dimensional array like r.
    :param cyclic: cyclic or periodic boundary.
    :param second: calculate second derivatives, default is first derivatives.
    :return: finite difference array.
    """

    # move specified dimension to the first
    p = np.arange(grid.ndim)
    p[-1] = dim - 1
    p[dim-1] = -1
    grid = np.transpose(grid, p)

    # construct shift vector
    sf = np.arange(grid.ndim)
    sf[0] = -1
    sb = np.arange(grid.ndim)
    sb[0] = 1

    # check coordinates
    if r is not None:
        if len(r) == 1:
            rr = np.arange(grid.shape[0], dtype=np.float) * r
        else:
            rr = r
        if np.ndim(rr) == 1:
            rr = arr.conform_dims(grid.shape, rr, [0])
        else:
            rr = np.transpose(rr, p)

        if map_scale is not None:    # check map scale
            mps = map_scale
            if np.ndim(mps) == 1:
                mps = arr.conform_dims(grid.shape, mps, [0])
            if np.ndim(mps) > 1:
                mps = np.transpose(mps, p)
            rr *= mps

    #
    # Compute center finite difference
    #

    # first derivative
    if not second:
        # value difference
        dgrid = np.roll(grid, -1, -1) - np.roll(grid, 1, -1)

        # grid space
        if r is not None:
            drr = np.roll(rr, -1, -1) - np.roll(rr, 1, -1)

        # deal boundary
        if cyclic:
            dgrid[..., 0] = grid[..., 1] - grid[..., -1]
            dgrid[..., -1] = grid[..., 0] - grid[..., -2]
            if r is not None:
                drr[..., 0] = 2*(rr[..., 1] - rr[..., 0])
                drr[..., -1] = 2*(rr[..., -1] - rr[..., -2])
        else:
            dgrid[..., 0] = grid[..., 1] - grid[..., 0]
            dgrid[..., -1] = grid[..., -1] - grid[..., -2]
            if r is not None:
                drr[..., 0] = rr[..., 1] - rr[..., 0]
                drr[..., -1] = rr[..., -1] - rr[..., -2]
    else:
        # value difference
        dgrid = np.roll(grid, -1, -1) - 2*grid + np.roll(grid, 1, -1)

        # grid space
        if r is not None:
            drr = (np.roll(rr, -1, -1) - rr) * (rr - np.roll(rr, 1, -1))

        # deal boundary
        if cyclic:
            dgrid[..., 0] = grid[..., 1] + grid[..., -1] - 2*grid[..., 0]
            dgrid[..., -1] = grid[..., 0] + grid[..., -2] - 2*grid[..., -1]
            if r is not None:
                drr[..., 0] = (rr[..., 1] - rr[..., 0]) * \
                              (rr[..., -1] - rr[..., -2])
                drr[..., -1] = drr[..., 0]
        else:
            dgrid[..., 0] = grid[..., 0] + grid[..., -2] - 2 * grid[..., 1]
            dgrid[..., -1] = grid[..., -1] + grid[..., -3] - 2 * grid[..., -2]
            if r is not None:
                drr[..., 0] = (rr[..., 1] - rr[..., 0]) * \
                              (rr[..., 2] - rr[..., 1])
                drr[..., -1] = (rr[..., -1] - rr[..., -2]) * \
                               (rr[..., -2] - rr[..., -3])

    # compute derivatives
    if r is not None:
        dgrid /= drr

    # restore grid array
    grid = np.transpose(grid, p)
    dgrid = np.transpose(dgrid, p)

    # return
    return dgrid


def calculate_distance_2d(lat1,lat2,lon1,lon2):
    # Calculates dx and dy for 2D arrays
    #=ACOS(COS(RADIANS(90-Lat1)) *COS(RADIANS(90-Lat2)) +SIN(RADIANS(90-Lat1)) *SIN(RADIANS(90-Lat2)) *COS(RADIANS(Long1-Long2))) *6371
    step1 = np.cos(np.radians(90.0-lat1))
    step2 = np.cos(np.radians(90.0-lat2))
    step3 = np.sin(np.radians(90.0-lat1))
    step4 = np.sin(np.radians(90.0-lat2))
    step5 = np.cos(np.radians(lon1-lon2))
    dist = np.arccos(step1 * step2 + step3 * step4 * step5) * constants.Re
    
    return dist


def compute_gradient(var,lats,lons):
    """
    Computes the horizontal gradient of a 2D scalar variable
    
    Returns:
        Returns ddx, ddy (x and y components of gradient) in units of (unit)/m
    """

    #Pull in lat & lon resolution
    latres = abs(lats[1]-lats[0])
    lonres = abs(lons[1]-lons[0])    
    
    #compute the length scale for each gridpoint as a 2D array
    lons2,lats2 = np.meshgrid(lons,lats)
    dx = calculate_distance_2d(lats2,lats2,lons2-(lonres),lons2+(lonres))
    dy = calculate_distance_2d(lats2-(latres),lats2+(latres),lons2,lons2)
    
    #Compute the gradient of the variable
    dvardy,dvardx = np.gradient(var)
    ddy = np.multiply(2,np.divide(dvardy,dy))
    ddx = np.multiply(2,np.divide(dvardx,dx))

    return ddx,ddy


def spatial_anomaly(varin,slice_option):
    """
    Computes the spatial anomaly of varin

    Input:    
        varin:        3D array of variable to compute anomaly of
        slice_option: 1 to compute anomaly of second dimension
                      2 to compute anomaly of third dimension
    Output:
        varanom: Anomaly of varin
        varanom_std = Standardized anomaly of varin

    Steven Cavallo
    March 2014
    University of Oklahoma
    """
    iz, iy, ix = varin.shape
        
    mvar = np.ma.masked_array(varin,np.isnan(varin)) 
    
    tmp = np.zeros_like(varin).astype('f')  
    tmp_std = np.zeros_like(varin).astype('f')  
    
    if slice_option == 1:
        var_mean = np.mean(mvar,2)
        var_std = np.std(mvar,2)
        for kk in range(0,iz): 
            for jj in range(0,iy):     
                tmp[kk,jj,:] = varin[kk,jj,:] - var_mean[kk,jj]    
                tmp_std[kk,jj,:] = var_std[kk,jj]
    else:
        var_mean = np.mean(mvar,1)
        var_std = np.std(mvar,1)
        for kk in range(0,iz): 
            for ii in range(0,ix):     
                tmp[kk,:,ii] = varin[kk,:,ii] - var_mean[kk,ii]    
                tmp_std[kk,:,ii] = var_std[kk,ii]    
            
    varanom = tmp
    varanom_std = tmp/tmp_std
    
    return varanom, varanom_std
