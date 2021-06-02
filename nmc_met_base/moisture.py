# _*_ coding: utf-8 _*_

# Copyright (c) 2020 NMC Developers.
# Distributed under the terms of the GPL V3 License.

"""
Compute moisture parameters.
"""

import xarray as xr
import numpy as np

from nmc_met_base.grid import interp_3D_to_surface


def cal_ivt(q, u, v, lon, lat, lev, surf_pres=None):
    """
    Calculate integrated water vapor transport.

    Args:
        q (numpy array): Specific humidity, g/kg, [nlev, nlat, nlon]
        u (numpy array): u component wind, m/s, [nlev, nlat, nlon]
        v (numpy array): v component wind, m/s, [nlev, nlat, nlon]
        lon (numpy array): vertical level, hPa, [nlev]
        lat (numpy array): longitude, [nlon]
        lev (numpy array): latitude, [nlat]
        surf_pres (numpy array, optional): surface pressure, hPa, [nlev, nlat, nlon]. Defaults to None.
    """

    # compute water vapor transport
    qu = q * u
    qv = q * v

    # set up full grid levels
    pCoord, _, _ = np.meshgrid(lev, lat, lon, indexing='ij')

    # mask the grid points under the ground
    if surf_pres is not None:
        # 将三维格点场插值到地面上, 这里地面的高度使用地面气压来指示
        qus = interp_3D_to_surface(qu, lon, lat, lev, surf_pres)
        qvs = interp_3D_to_surface(qv, lon, lat, lev, surf_pres)

        # 判断三维格点是否在地面之下, 如果在地面之下, 其物理量用地面替代, 并且高度也设置为地面气压
        # 这样在后面积分过程中, 地面以下的积分为零值
        for ilevel, level in enumerate(lev):
            qu[ilevel, ...] = np.where(surf_pres >= level, qu[ilevel, ...], qus)
            qv[ilevel, ...] = np.where(surf_pres >= level, qv[ilevel, ...], qvs)
            pCoord[ilevel, ...] = np.where(surf_pres >= level, level, surf_pres)

    # compute the vertical integration, using trapezoid rule
    # 由于垂直坐标用的是百帕, 而比湿用的是g/kg, 因此转换单位100*0.001=0.1
    iqu = np.zeros((lat.size, lon.size))
    iqv = np.zeros((lat.size, lon.size))
    for ilevel, level in enumerate(lev[0:-1]):
        iqu += np.abs(pCoord[ilevel,...]-pCoord[ilevel+1,...])*(qu[ilevel,...]+qu[ilevel+1,...])*0.5*0.1/9.8
        iqv += np.abs(pCoord[ilevel,...]-pCoord[ilevel+1,...])*(qv[ilevel,...]+qv[ilevel+1,...])*0.5*0.1/9.8

    return iqu, iqv

