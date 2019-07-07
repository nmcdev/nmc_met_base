# _*_ coding: utf-8 _*_

# Copyright (c) 2019 NMC Developers.
# Distributed under the terms of the GPL V3 License.

"""
  Calculate grid derivative.
"""

import numpy as np
from nmc_met_base.arr import conform_dims


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
            rr = conform_dims(grid.shape, rr, [0])
        else:
            rr = np.transpose(rr, p)

        if map_scale is not None:    # check map scale
            mps = map_scale
            if np.ndim(mps) == 1:
                mps = conform_dims(grid.shape, mps, [0])
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
