# _*_ coding: utf-8 _*_

# Copyright (c) 2019 NMC Developers.
# Distributed under the terms of the GPL V3 License.

"""
  Compute dynamic physical parameters on lon/lat grid.

  refer
  https://github.com/keltonhalbert/wrftools/blob/master/wrftools/variables/winds.py
  https://github.com/tomerburg/metlib/blob/master/diagnostics/met_functions.py
  https://bitbucket.org/tmiyachi/pymet
"""

import numpy as np
import xarray as xr
from scipy import ndimage
from nmc_met_base import arr, constants, calculate
from nmc_met_base.grid import calc_dx_dy, dvardx, dvardy, d2vardx2, \
                              d2vardy2, dvardp, gradient_sphere, rot
from nmc_met_base.thermal import pottemp_3D, stability, vapor_pressure, \
                                 relh_temp, specific_humidity, theta


NA = np.newaxis
a0 = constants.Re
g = constants.g0
PI = constants.pi
d2r = PI/180.
Rd = constants.rd


def advection(var,u,v,lats,lons):
    """
    Compute the magnitude of horizontal advection of a scalar quantity by the wind
    
    Args:
        var ([type]): 2D scalar field (e.g. temperature)
        u ([type]): 2D arrays of u & v wind components, in meters per second
        v ([type]): 2D arrays of u & v wind components, in meters per second
    
    Returns:
        Returns in units of (scalar unit) per second
    """
    
    #Check if input is an xarray dataarray
    use_xarray = arr.check_xarray(var)

    #Compute the gradient of the variable
    ddx,ddy = calculate.compute_gradient(var,lats,lons)
    
    #Compute advection (-v dot grad var)
    #adv = -1 * ((ddx*u) + (ddy*v))
    adv = np.add(np.multiply(ddx,u),np.multiply(ddy,v))
    adv = np.multiply(-1.0,adv)
    
    #Convert back to xarray dataset, if initially passed as one
    if use_xarray == True:
        adv = xr.DataArray(adv, coords=[lats, lons], dims=['lat', 'lon'])
    
    return adv


def divergence(u,v,lats,lons):
    """
    Compute the horizontal divergence of a vector
    
    Args:
        var ([type]): 2D scalar field (e.g. temperature)
        u ([type]): 2D arrays of u & v wind components, in meters per second
        v ([type]): 2D arrays of u & v wind components, in meters per second
    
    Returns:
        Returns in units of per second
    """
    
    #Check if input is an xarray dataarray
    use_xarray = arr.check_xarray(u)

    #Compute the gradient of the wind
    dudx = calculate.compute_gradient(u,lats,lons)[0]
    dvdy = calculate.compute_gradient(v,lats,lons)[1]

    #div = dudx + dvdy #dv/dx - du/dy
    div = np.add(dudx,dvdy)
    
    #Convert back to xarray dataset, if initially passed as one
    if use_xarray == True:
        div = xr.DataArray(div, coords=[lats, lons], dims=['lat', 'lon'])
    
    return div


def relvort(u,v,lats,lons):
    """
    Compute the relative vertical vorticity of the wind
    
    Args:
        u ([type]): 2D arrays of u wind components, in meters per second
        v ([type]): 2D arrays of v wind components, in meters per second
    
    Returns:
        Returns in units of per second
    """
    
    #Check if input is an xarray dataarray
    use_xarray = arr.check_xarray(u)
    
    #Compute the gradient of the wind
    dudy = calculate.compute_gradient(u,lats,lons)[1]
    dvdx = calculate.compute_gradient(v,lats,lons)[0]

    #Compute relative vorticity (dv/dx - du/dy)
    vort = np.subtract(dvdx,dudy)
    
    #Account for southern hemisphere
    tlons, tlats = np.meshgrid(lons,lats)
    vort[tlats<0] = vort[tlats<0] * -1
    
    #Convert back to xarray dataset, if initially passed as one
    if use_xarray == True:
        vort = xr.DataArray(vort, coords=[lats, lons], dims=['lat', 'lon'])
    
    return vort


def absvort(u,v,lats,lons):
    """
    Compute the absolute vertical vorticity of the wind

    Args:
        u ([type]): 2D arrays of u wind components, in meters per second
        v ([type]): 2D arrays of v wind components, in meters per second
    
    Returns:
        Returns in units of per second
    """
    
    #Check if input is an xarray dataarray
    use_xarray = arr.check_xarray(u)
    
    #Compute relative vorticity
    vort = relvort(u,v,lats,lons)
    
    #Compute the Coriolis parameter (after converting lat to radians)
    cor2d = coriolis(lats,lons)

    #Compute absolute vorticity (relative + coriolis parameter)
    vort = np.add(vort,cor2d)
    
    #Convert back to xarray dataset, if initially passed as one
    if use_xarray == True:
        vort = xr.DataArray(vort, coords=[lats, lons], dims=['lat', 'lon'])
    
    return vort


def coriolis(lats,lons):
    """
    Computes and returns a 2D array of the Coriolis parameter
    
    Args:
        lats ([type]): 1D array of latitude and longitudes (degrees)
        lons ([type]): 2D array of latitude and longitudes (degrees)
    
    Returns:
        Returns coriolis in units of 1/s
    """

    #Compute the Coriolis parameter (after converting lat to radians)
    lons2,lats2 = np.meshgrid(lons,lats)
    sinlat = np.sin(lats2 * (constants.pi/180))
    cor = np.multiply(2.0,np.multiply(constants.omega,sinlat))
    
    return cor


def geo(hght,lats,lons):
    """
    Compute the u and v components of the geostrophic wind
    
    Args:
        hght ([type]): 2D scalar geopotential height field (m)
    
    Returns:
        Returns in units of meters per second
    """
    
    #Check if input is an xarray dataarray
    use_xarray = arr.check_xarray(hght)

    #Compute geopotential height gradient on pressure surface
    dzdx,dzdy = calculate.compute_gradient(hght,lats,lons)
    
    #2D array of Coriolis parameter for each lat/lon
    cor = coriolis(lats,lons)
    
    #Compute the geostrophic wind
    ug = (-1.0 * dzdy * g) / cor
    vg = (dzdx * g) / cor
    
    #Convert back to xarray dataset, if initially passed as one
    if use_xarray == True:
        ug = xr.DataArray(ug, coords=[lats, lons], dims=['lat', 'lon'])
        vg = xr.DataArray(vg, coords=[lats, lons], dims=['lat', 'lon'])
    
    return ug,vg


def ageo(hght,u,v,lats,lons):
    """
    Compute the u and v components of the ageostrophic wind
    
    Args:
        hght ([type]): 2D scalar geopotential height field (m)
        u ([type]):  2D arrays of u components of wind (m/s)
        v ([type]):  2D arrays of v components of wind (m/s)
    
    Returns:
        Returns in units of meters per second
    """
    
    #Check if input is an xarray dataarray
    use_xarray = arr.check_xarray(hght)

    #Compute the geostrophic wind
    ug,vg = geo(hght,lats,lons)
    
    #Compute the ageostrophic wind
    ua = u - ug
    va = v - vg
    
    #Convert back to xarray dataset, if initially passed as one
    if use_xarray == True:
        ua = xr.DataArray(ua, coords=[lats, lons], dims=['lat', 'lon'])
        va = xr.DataArray(va, coords=[lats, lons], dims=['lat', 'lon'])
    
    return ua,va


def qvect(temp,hght,lev,lats,lons,smooth,static_stability=1):
    """
    Compute the u and v components of the Q-vector
    
    Args:
        temp ([type]): 2D scalar temperature field (K)
        hght ([type]):  2D scalar geopotential height field (m)
        lev ([type]): Pressure level (hPa)
        lats ([type]): 1D arrays of lat
        lons ([type]): 1D arrays of lon
        smooth ([type]): integer representing sigma level of smoothing
        static_stability (int, optional): assumed to be 1, unless provided. Defaults to 1.
    
    Returns:
        Returns in units of meters per second
    """
    
    #Check if input is an xarray dataarray
    use_xarray = arr.check_xarray(temp)
    
    #Smooth data
    hght = ndimage.gaussian_filter(hght,sigma=smooth,order=0)
    temp = ndimage.gaussian_filter(temp,sigma=smooth,order=0)
    
    #Convert pressure to Pa
    levPa = lev * 100.0

    #Compute the geostrophic wind
    ug,vg = geo(hght,lats,lons)
    
    #Compute the constant out front
    const = (-1.0 * Rd) / (levPa * static_stability)
    
    #Compute gradient quantities
    dtdx,dtdy = calculate.compute_gradient(temp,lats,lons)
    dudx,dudy = calculate.compute_gradient(ug,lats,lons)
    dvdx,dvdy = calculate.compute_gradient(vg,lats,lons)
    
    #Compute u,v components of Q-vector
    Qu = const * ((dudx*dtdx) + (dvdx*dtdy))
    Qv = const * ((dudy*dtdx) + (dvdy*dtdy))
    
    #Convert back to xarray dataset, if initially passed as one
    if use_xarray == True:
        Qu = xr.DataArray(Qu, coords=[lats, lons], dims=['lat', 'lon'])
        Qv = xr.DataArray(Qv, coords=[lats, lons], dims=['lat', 'lon'])
    
    return Qu,Qv
    

def ivt(temp,rh,levs,u,v,lats,lons):
    """
    Compute integrated vapor transport, assuming the pressure interval is constant
    
    Args:
        temp ([type]): 3D array (lev,lat,lon)  of temperature (K)
        rh ([type]): 3D array (lev,lat,lon) of relative humidity (in %)
        levs ([type]): 1D array of pressure levels (hPa)
        u ([type]):  3D array u-wind (m/s)
        v ([type]):  3D array v-wind (m/s)
    """
    
    #Check if input is an xarray dataarray
    use_xarray = arr.check_xarray(temp)
    
    #If using xarray, convert to numpy arrays
    if use_xarray == 1:
        try:
            temp = temp.values
        except:
            pass
        try:
            rh = rh.values
        except:
            pass
        try:
            u = u.values
        except:
            pass
        try:
            v = v.values
        except:
            pass
    
    #Get list of pressure levels in hPa, convert to Pa
    levs = levs * 100.0 #convert pressure to Pa
    
    #determine vertical dz in Pa, assuming levs array is uniform
    vint = (levs[1]-levs[0])
    
    nvert, nlat, nlon = np.shape(rh)
    pres = np.copy(rh) * 0.0
    
    #Arrange a 3D pressure array
    for k in range(0,nvert):
        pres[k] += levs[k]
    
    #saturated vapor pressure in Pa
    es = vapor_pressure(temp) * 100.0
    
    #get e from RH (in decimals) in Pa
    e = (rh / 100.0) * es
    
    #Approximate specific humidity q ~ w
    q = 0.622 * (e / pres)
    
    #Compute u and v components of IVT vector
    ut = np.trapz(u*q, axis=0, dx=vint) / -9.8
    vt = np.trapz(v*q, axis=0, dx=vint) / -9.8

    #Compute magnitude of IVT vector
    ivt = np.sqrt(np.add(np.square(ut),np.square(vt)))
    
    #Convert back to xarray dataset, if initially passed as one
    if use_xarray == True:
        ut  = xr.DataArray(ut, coords=[lats, lons], dims=['lat', 'lon'])
        vt  = xr.DataArray(vt, coords=[lats, lons], dims=['lat', 'lon'])
        ivt = xr.DataArray(ivt, coords=[lats, lons], dims=['lat', 'lon'])
    
    return ut, vt, ivt


def integrated_vapor(temp,rh,pressfc,levs,lats,lons):
    """
    Compute integrated vapor over a certain pressure layer, assuming the pressure
    interval is constant.
    
    Args:
        temp ([type]): 3D array (lev,lat,lon)  of temperature (K)
        rh ([type]): 3D array (lev,lat,lon) of relative humidity (in %)
        pressfc ([type]): Surface pressure (hPa)
        levs ([type]): 1D array of pressure levels (hPa)
    """
    
    #Check if input is an xarray dataarray
    use_xarray = arr.check_xarray(temp)
    
    #If using xarray, convert to numpy
    if use_xarray == 1:
        try:
            temp = temp.values
        except:
            pass
        try:
            rh = rh.values
        except:
            pass
    
    #Convert pressure to hPa
    levs = levs * 100.0
    
    #determine vertical dz in Pa, assuming levs array is uniform
    vint = (levs[1]-levs[0])
    
    #pres,lons0,lats0 = np.meshgrid(rh.lev,lons,lats)
    nvert, nlat, nlon = np.shape(rh)
    pres = np.copy(rh) * 0.0
    
    #Arrange a 3D pressure array
    for k in range(0,nvert):
        pres[k] += levs[k]
        
        #Mask by surface pressure
        tmp = rh[k]
        tmp[pressfc < levs[k]] = 0.0
        rh[k] = tmp
    
    #saturated vapor pressure in Pa
    es = vapor_pressure(temp) * 100.0
    
    #get e from RH (in decimals) in Pa
    e = (rh / 100.0) * es
    
    #Approximate specific humidity q ~ w
    w = 0.622 * (e / pres) #used to be 0.622
    q = w
    
    #Compute integrated vapor
    iv = np.trapz(q, axis=0, dx=vint) / -9.8
    
    #Return ut, vt as xarray DataArrays
    if use_xarray == 1:
        iv = xr.DataArray(iv, coords=[lats, lons], dims=['lat', 'lon'])
    
    return iv


def moisture_conv(u,v,temp,dwpt,pres,lats,lons,smth=0):
    
    #Smooth all fields
    u = ndimage.gaussian_filter(u,smth)
    v = ndimage.gaussian_filter(v,smth)
    temp = ndimage.gaussian_filter(temp,smth)
    dwpt = ndimage.gaussian_filter(dwpt,smth)
    pres = ndimage.gaussian_filter(pres,smth)
    
    #Compute relative humidity
    rh = relh_temp(temp,dwpt)
    
    #Compute q
    q = specific_humidity(temp,pres,rh)# * 1000.0
    
    #Compute moisture convergence
    term1 = advection(q,u,v,lats,lons)
    term2 = np.multiply(q,divergence(u,v,lats,lons))
    
    return (term1 - term2)


def pv_ertel(u,v,dthetadp,lats,lons):
    """
    Computes Ertel Potential Vorticity
    
    Args:
        u ([type]):  u-wind (m/s)
        v ([type]):  v-wind (m/s)
        dthetadp ([type]): change of theta with respect to pressure
    
    Returns:
        Returns EPV in units of PVU
    """

    #Compute absolute vorticity
    vort = absvort(u,v,lats,lons)
    
    #pv = -g(absvort)(dthetadp)
    pv = np.multiply(g,np.multiply(vort,dthetadp))
    pv = np.multiply(pv,-1.0)
    
    #convert to PVU
    pv = pv * 10**6

    return pv


def isentropic_transform(temp,u,v,lev,lats,lons,levs,tomask=1):
    """
    Transform a variable to isentropic coordinates, specifying a single isentropic level
    https://github.com/tomerburg/metlib/blob/master/diagnostics/met_functions.py
    
    Args:
        temp ([type]): 3D temperature array (K)
        u ([type]): 3D wind array (u)
        v ([type]): 3D wind array (v)
        lev ([type]): Desired isentropic level (K, scalar)
        lats ([type]): 1D lat array
        lons ([type]): 1D lon array
        levs ([type]): 1D pressure array
        tomask (int, optional): mask array values where the desired isentropic surface is below the
#                               ground. Yes=1, No=0. Default is yes (1). Defaults to 1.
    
    Returns:
        Returns a python list with the following quantities:
        [0] = 2D pressure array
        [1] = u-wind
        [2] = v-wind
        [3] = d(theta)/dp
        [4] = 2D array corresponding to the first k-index of where the
              theta threshold is exceeded.
    """
    
    #Check if input is an xarray dataarray
    use_xarray = arr.check_xarray(temp)
    
    #If using xarray, convert to numpy
    if use_xarray == 1:
        temp = temp.values
        u = u.values
        v = v.values
    
    #Subset data values to below 100 hPa
    tlev = float(lev)
    
    #Arrange a 3D pressure array of theta
    vtheta = np.copy(temp) * 0.0
    
    nvert, nlat, nlon = np.shape(temp)
    for k in range(0,nvert):
        vtheta[k] = theta(temp[k],levs[k])
        
    #Arrange 2D arrays of other values
    tpres = 0
    tu = 0
    tv = 0
    
    #Eliminate any NaNs to avoid issues
    temp = np.nan_to_num(temp)
    u = np.nan_to_num(u)
    v = np.nan_to_num(v)
    vtheta = np.nan_to_num(vtheta)
    
    #==================================================================
    
    #Step 0: Get 3d array of pressure
    levs3d = np.copy(temp) * 0.0
    
    nvert, nlat, nlon = np.shape(temp)
    for k in range(0,nvert):
        levs3d[k] = (vtheta[0] * 0.0) + levs[k]
    
    #------------------------------------------------------------------
    #Step 1: find first instances bottom-up of theta exceeding threshold
    
    #Check where the theta threshold is exceeded in the 3D array
    check_thres = np.where(vtheta >= tlev)
    check_ax1 = check_thres[0]
    check_ax2 = check_thres[1]
    check_ax3 = check_thres[2]
    
    #This is a 2D array corresponding to the first k-index of where the
    #theta threshold is exceeded.
    thres_pos = np.copy(vtheta[0]) * 0.0
    
    #Loop through all such positions and only record first instances
    for i in range(0,len(check_ax1)):
        pres_level = check_ax1[i]
        jval = check_ax2[i]
        ival = check_ax3[i]
        
        if thres_pos[jval][ival] == 0: thres_pos[jval][ival] = pres_level
    
    #------------------------------------------------------------------
    #Step 2: get the theta values corresponding to this axis
    
    #Convert the position of the theta threshold values to something readable
    thres_pos = thres_pos.astype('int64')
    thres_last = thres_pos - 1
    thres_last[thres_last < 0] = 0
    
    #replace NaNs, if any
    thres_pos = np.nan_to_num(thres_pos)
    thres_last = np.nan_to_num(thres_last)
    vtheta = np.nan_to_num(vtheta)
    
    #Get theta values where it's first exceeded and 1 vertical level below it
    ktheta = np.ndarray.choose(thres_pos,vtheta)
    ltheta = np.ndarray.choose(thres_last,vtheta)
    
    #Get the difference in theta between levels
    diffu = np.abs(np.subtract(tlev,ktheta))
    diffl = np.abs(np.subtract(tlev,ltheta))
    
    #Percentage from the lower level to the upper one
    perc = np.divide(diffl,np.add(diffu,diffl))
    
    #------------------------------------------------------------------
    #Step 3: find pressure at this level
    
    valu = np.ndarray.choose(thres_pos,levs3d)
    vall = np.ndarray.choose(thres_last,levs3d)
    
    #Adjustment factor
    fac = np.multiply(np.subtract(vall,valu),perc)
                    
    #New pressure
    tpres = np.subtract(vall,fac)
    
    #------------------------------------------------------------------
    #Step 3a: get d(theta)/dp array
    
    #d(theta)/dp = ktheta-ltheta / valu-vall
    dthetadp = np.divide(np.subtract(ktheta,ltheta),np.subtract(valu,vall))
    
    #Convert to units of K/Pa
    dthetadp = np.divide(dthetadp,100)
    
    #------------------------------------------------------------------
    #Step 4: find wind at this level
    
    uu = np.ndarray.choose(thres_pos,u)
    ul = np.ndarray.choose(thres_last,u)
    vu = np.ndarray.choose(thres_pos,v)
    vl = np.ndarray.choose(thres_last,v)
    
    fac = np.multiply(np.subtract(ul,uu),perc)
    tu = np.subtract(ul,fac)

    fac = np.multiply(np.subtract(vl,vu),perc)
    tv = np.subtract(vl,fac)
    
    #==================================================================
    # ALL RESUMES HERE
    #==================================================================
    
    if tomask == 1:
        pres = np.ma.masked_where(thres_pos <= 1.0,tpres)
        u = np.ma.masked_where(thres_pos <= 1.0,tu)
        v = np.ma.masked_where(thres_pos <= 1.0,tv)
    
    #Convert back to xarray, if specified initially
    if use_xarray == 1:
        pres = xr.DataArray(pres,coords=[lats,lons],dims=['lat','lon'])
        u = xr.DataArray(u,coords=[lats,lons],dims=['lat','lon'])
        v = xr.DataArray(v,coords=[lats,lons],dims=['lat','lon'])
    
    return pres,u,v,dthetadp,thres_pos


def avort(uwind, vwind, lon, lat):
    """
    Calculate absolute vorticity.
    refer to
    https://nbviewer.jupyter.org/url/fujita.valpo.edu/~kgoebber/NAM_vorticity.ipynb

    :param uwind: u direction wind.
    :param vwind: v direction wind.
    :param lon: grid longitude.
    :param lat: grid latitude.
    :return: relative vorticity.

    :Example:

    """

    # grid space
    dx, dy = calc_dx_dy(lon, lat)

    # relative vorticity
    dvdx = np.gradient(vwind, dx, axis=1)
    dudy = np.gradient(uwind, dy, axis=0)
    cor = 2 * (7.292 * 10 ** (-5)) * np.sin(np.deg2rad(lat))

    return dvdx - dudy + cor


def absvrt(uwnd, vwnd, lon, lat, xdim, ydim, cyclic=True, sphere=True):
    """
    Calculate absolute vorticity.

    :param uwnd: ndarray, u-component wind.
    :param vwnd: ndarray, v-component wind.
    :param lon: array_like, longitude.
    :param lat: array_like, latitude.
    :param xdim: the longitude dimension index
    :param ydim: the latitude dimension index
    :param cyclic: east-west boundary is cyclic
    :param sphere: sphere coordinate
    :return:
    """

    u, v = np.ma.getdata(uwnd), np.ma.getdata(vwnd)
    mask = np.ma.getmask(uwnd) | np.ma.getmask(vwnd)
    ndim = u.ndim

    vor = rot(u, v, lon, lat, xdim, ydim, cyclic=cyclic, sphere=sphere)
    f = arr.expand(constants.earth_f(lat), ndim, axis=ydim)
    out = f + vor

    out = np.ma.array(out, mask=mask)
    out = arr.mrollaxis(out, ydim, 0)
    out[0, ...] = np.ma.masked
    out[-1, ...] = np.ma.masked
    out = arr.mrollaxis(out, 0, ydim + 1)

    return out


def ertelpv(uwnd, vwnd, temp, lon, lat, lev, xdim, ydim, zdim,
            cyclic=True, punit=100., sphere=True):
    """
    Calculate Ertel potential vorticity.
    Hoskins, B.J., M.E. McIntyre and A.E. Robertson, 1985:
      On the use and significance of isentropic potential
      vorticity maps, `QJRMS`, 111, 877-946,
    <http://onlinelibrary.wiley.com/doi/10.1002/qj.49711147002/abstract>

    :param uwnd: ndarray, u component wind [m/s].
    :param vwnd: ndarray, v component wind [m/s].
    :param temp: ndarray, temperature [K].
    :param lon: array_like, longitude [degrees].
    :param lat: array_like, latitude [degrees].
    :param lev: array_like, pressure level [punit*Pa].
    :param xdim: west-east axis
    :param ydim: south-north axis
    :param zdim: vertical axis
    :param cyclic: west-east cyclic boundary
    :param punit: pressure level unit
    :param sphere: sphere coordinates.
    :return:
    """

    u, v, t = np.ma.getdata(uwnd), np.ma.getdata(vwnd), np.ma.getdata(temp)
    mask = np.ma.getmask(uwnd) | np.ma.getmask(vwnd) | np.ma.getmask(temp)
    ndim = u.ndim

    # potential temperature
    theta = pottemp_3D(t, lev, zdim, punit=punit)

    # partial derivation
    dthdp = dvardp(theta, lev, zdim, punit=punit)
    dudp = dvardp(u, lev, zdim, punit=punit)
    dvdp = dvardp(v, lev, zdim, punit=punit)

    dthdx = dvardx(theta, lon, lat, xdim, ydim, cyclic=cyclic, sphere=sphere)
    dthdy = dvardy(theta, lat, ydim, sphere=sphere)

    # absolute vorticity
    vor = rot(u, v, lon, lat, xdim, ydim, cyclic=cyclic, sphere=sphere)
    f = arr.expand(constants.earth_f(lat), ndim, axis=ydim)
    avor = f + vor

    out = -g * (avor*dthdp - (dthdx*dvdp-dthdy*dudp))

    out = np.ma.array(out, mask=mask)
    out = arr.mrollaxis(out, ydim, 0)
    out[0, ...] = np.ma.masked
    out[-1, ...] = np.ma.masked
    out = arr.mrollaxis(out, 0, ydim+1)

    return out


def vertical_vorticity_latlon(u, v, lats, lons, abs_opt=False):
    """
    Calculate the vertical vorticity on a latitude/longitude grid.

    :param u: 2 dimensional u wind arrays, dimensioned by (lats,lons).
    :param v: 2 dimensional v wind arrays, dimensioned by (lats,lons).
    :param lats: latitude vector
    :param lons: longitude vector
    :param abs_opt: True to compute absolute vorticity,
                    False for relative vorticity only
    :return: Two dimensional array of vertical vorticity.
    """

    dudy, dudx = gradient_sphere(u, lats, lons)
    dvdy, dvdx = gradient_sphere(v, lats, lons)

    if abs_opt:
        # 2D latitude array
        glats = np.zeros_like(u).astype('f')
        for jj in range(0, len(lats)):
            glats[jj, :] = lats[jj]

        # Coriolis parameter
        f = 2 * 7.292e-05 * np.sin(np.deg2rad(glats))
    else:
        f = 0.

    vert_vort = dvdx - dudy + f

    return vert_vort


def epv_sphere(theta, u, v, levs, lats, lons):
    """
    Computes the Ertel Potential Vorticity (PV) on a latitude/longitude grid.
    https://github.com/scavallo/python_scripts/blob/master/utils/weather_modules.py

    :param theta: 3D potential temperature array on isobaric levels
    :param u: 3D u components of the horizontal wind on isobaric levels
    :param v: 3D v components of the horizontal wind on isobaric levels
    :param levs: 1D pressure vectors
    :param lats: 1D latitude vectors
    :param lons: 1D longitude vectors
    :return: Ertel PV in potential vorticity units (PVU)
    """

    iz, iy, ix = theta.shape

    dthdp, dthdy, dthdx = gradient_sphere(theta, levs, lats, lons)
    dudp, dudy, dudx = gradient_sphere(u, levs, lats, lons)
    dvdp, dvdy, dvdx = gradient_sphere(v, levs, lats, lons)

    abvort = np.zeros_like(theta).astype('f')
    for kk in range(0, iz):
        abvort[kk, :, :] = vertical_vorticity_latlon(
            u[kk, :, :].squeeze(), v[kk, :, :].squeeze(),
            lats, lons, abs_opt=True)

    epv = (-9.81 * (-dvdp * dthdx - dudp * dthdy + abvort * dthdp)) * 1.0e6
    return epv


def eliassen_palm_flux_sphere(geop,theta,lats,lons,levs,normalize_option):
    """
    Computes the 3-D Eliassen-Palm flux vectors and divergence on a 
    latitude/longitude grid with any vertical coordinate 
   
    Computation is Equation 5.7 from:
    R. A. Plumb, On the Three-dimensional Propagation of Stationary Waves, J. Atmos. Sci., No. 3, 42 (1985).
  
    Input:    
        geop:      3D geopotential (m^2 s-2)
        theta:     3D potential temperature (K)
        lats,lons: 1D latitude and longitude vectors
        levs:      1D pressure vector (Pa)
        normalize_option: 0 = no normalizing, 1 = normalize by maximum value at each vertical level, 2 = standardize 
   
    Output:      
       Fx, Fy, Fz: Eliassen-Palm flux x, y, z vector components
       divF: Eliassen-Palm flux divergence
   
    Steven Cavallo
    March 2014
    University of Oklahoma
    """
    
    iz, iy, ix = geop.shape
    
    # First need to filter out numeric nans    
    geop = arr.filter_numeric_nans(geop,0,0,'low')    
    theta = arr.filter_numeric_nans(theta,200,0,'low')
    theta = arr.filter_numeric_nans(theta,10000,0,'high')

    theta_anom, theta_anom_std = calculate.spatial_anomaly(theta,1) 
    geop_anom , geop_anom_std = calculate.spatial_anomaly(geop,1)          
        
    latarr = np.zeros_like(geop).astype('f')    
    farr = np.zeros_like(geop).astype('f')    
    for kk in range(0,iz):
        for jj in range(0,iy):            
            latarr[kk,jj,:] = lats[jj]
            farr[kk,jj,:] = 2.0*constants.omega*np.sin(lats[jj]*(np.pi/180.0))
          
    psi = geop / farr     
    psi_anom, psi_anom_std = calculate.spatial_anomaly(psi,1) 
        
    pres = np.zeros_like(theta_anom).astype('f')   
    for kk in range(0,iz):      
        pres[kk,:,:] = levs[kk]
    
    coef = pres*np.cos(latarr*(np.pi/180.0))    
    arg1 = coef/( 2.0*np.pi*(constants.Re**2)*np.cos(latarr*(np.pi/180.0))*np.cos(latarr*(np.pi/180.0)) )
    arg2 = coef/( 2.0*np.pi*(constants.Re**2)*np.cos(latarr*(np.pi/180.0)) )
    arg3 = coef*(2.0*(constants.omega**2)*np.sin(latarr*(np.pi/180.0))*np.sin(latarr*(np.pi/180.0)))
   
    dthdz, yy, xx = gradient_sphere(theta, geop/g, lats, lons) 
    xx, dthdy, dthdx = gradient_sphere(theta_anom, geop/g, lats, lons)        
    dpsidz, dpsidy, dpsidx = gradient_sphere(psi_anom, geop/g, lats, lons)          
    d2psidxdz, d2psidxdy, d2psidx2 = gradient_sphere(dpsidx, geop/g, lats, lons)     
    aaa, d2psidxdz, ccc = gradient_sphere(dpsidz, geop/g, lats, lons) 

    N2 = (g/theta)*(dthdz)
    arg4 = arg3/(N2*constants.Re*np.cos(latarr*(np.pi/180.0)))        
    
    Fx = arg1*( dpsidx**2.0 - (psi_anom*d2psidx2))
    Fy = arg2*((dpsidy*dpsidx) - (psi_anom*d2psidxdy))
    Fz = arg4*((dpsidx*dpsidz) - (psi_anom*d2psidxdz))          
    
    Fx_z, Fx_y, Fx_x = gradient_sphere(Fx, geop/g, lats, lons)    
    Fy_z, Fy_y, Fy_x = gradient_sphere(Fy, geop/g, lats, lons)    
    Fz_z, Fz_y, Fz_x = gradient_sphere(Fz, geop/g, lats, lons)    
       
    divF = Fx_x + Fy_y + Fz_z
    
    if normalize_option == 1:        
        for kk in range(0,iz):  
            Fx[kk,:,:] =  Fx[kk,:,:] / np.nanmax(np.abs(Fx[kk,:,:]))
            Fy[kk,:,:] =  Fy[kk,:,:] / np.nanmax(np.abs(Fy[kk,:,:]))
            Fz[kk,:,:] =  Fz[kk,:,:] / np.nanmax(np.abs(Fz[kk,:,:]))
    if normalize_option == 2:        
        for kk in range(0,iz):  
            Fx[kk,:,:] =  Fx[kk,:,:] / np.nanstd(Fx[kk,:,:])
            Fy[kk,:,:] =  Fy[kk,:,:] / np.nanstd(Fy[kk,:,:])
            Fz[kk,:,:] =  Fz[kk,:,:] / np.nanstd(Fz[kk,:,:])    

    return Fx, Fy, Fz, divF


def tnflux2d(U, V, strm, lon, lat, xdim, ydim, cyclic=True, limit=100):
    """
    Takaya & Nakamura (2001) 计算水平等压面上的波活动度.
    Takaya, K and H. Nakamura, 2001: A formulation of a phase-independent
      wave-activity flux for stationary and migratory quasigeostrophic eddies
      on a zonally varying basic flow, `JAS`, 58, 608-627.
    http://journals.ametsoc.org/doi/abs/10.1175/1520-0469%282001%29058%3C0608%3AAFOAPI%3E2.0.CO%3B2

    :param U: u component wind [m/s].
    :param V: v component wind [m/s].
    :param strm: stream function [m^2/s].
    :param lon: longitude degree.
    :param lat: latitude degree.
    :param xdim: longitude dimension index.
    :param ydim: latitude dimension index.
    :param cyclic: east-west cyclic boundary.
    :param limit:
    :return:
    """

    U, V = np.asarray(U), np.asarray(V)
    ndim = U.ndim

    dstrmdx = dvardx(strm, lon, lat, xdim, ydim, cyclic=cyclic)
    dstrmdy = dvardy(strm, lat, ydim)
    d2strmdx2 = d2vardx2(strm, lon, lat, xdim, ydim, cyclic=cyclic)
    d2strmdy2 = d2vardy2(strm, lat, ydim)
    d2strmdxdy = dvardy(
        dvardx(strm, lon, lat, xdim, ydim, cyclic=cyclic),
        lat, ydim)

    tnx = U * (dstrmdx ** 2 - strm * d2strmdx2) + \
        V * (dstrmdx * dstrmdy - strm * d2strmdxdy)
    tny = U * (dstrmdx * dstrmdy - strm * d2strmdxdy) + \
        V * (dstrmdy ** 2 - strm * d2strmdy2)

    tnx = 0.5 * tnx / np.abs(U + 1j * V)
    tny = 0.5 * tny / np.abs(U + 1j * V)

    tnxy = np.sqrt(tnx ** 2 + tny ** 2)
    tnx = np.ma.asarray(tnx)
    tny = np.ma.asarray(tny)
    tnx[tnxy > limit] = np.ma.masked
    tny[tnxy > limit] = np.ma.masked
    tnx[U < 0] = np.ma.masked
    tny[U < 0] = np.ma.masked

    return tnx, tny


def tnflux3d(U, V, T, strm, lon, lat, lev, xdim, ydim, zdim,
             cyclic=True, limit=100, punit=100.):
    """
    Takaya & Nakamura (2001) 计算等压面上的波活动度.
    Takaya, K and H. Nakamura, 2001: A formulation of a phase-independent
      wave-activity flux for stationary and migratory quasigeostrophic eddies
      on a zonally varying basic flow, `JAS`, 58, 608-627.
    http://journals.ametsoc.org/doi/abs/10.1175/1520-0469%282001%29058%3C0608%3AAFOAPI%3E2.0.CO%3B2

    :param U: u component wind [m/s].
    :param V: v component wind [m/s].
    :param T: climate temperature [K].
    :param strm: stream function bias [m^2/s]
    :param lon: longitude degree.
    :param lat: latitude degree.
    :param lev: level pressure.
    :param xdim: longitude dimension index.
    :param ydim: latitude dimension index.
    :param zdim: level dimension index.
    :param cyclic: east-west cyclic boundary.
    :param limit:
    :param punit: level pressure unit.
    :return: east-west, south-north, vertical component.
    """

    U, V, T = np.asarray(U), np.asarray(V), np.asarray(T)
    ndim = U.ndim
    S = stability(T, lev, zdim, punit=punit)
    f = arr.expand(constants.earth_f(lat), ndim, axis=ydim)

    dstrmdx = dvardx(strm, lon, lat, xdim, ydim, cyclic=cyclic)
    dstrmdy = dvardy(strm, lat, ydim)
    dstrmdp = dvardp(strm, lev, zdim, punit=punit)
    d2strmdx2 = d2vardx2(strm, lon, lat, xdim, ydim, cyclic=cyclic)
    d2strmdy2 = d2vardy2(strm, lat, ydim)
    d2strmdxdy = dvardy(dstrmdx, lat, ydim)
    d2strmdxdp = dvardx(dstrmdp, lon, lat, xdim, ydim, cyclic=True)
    d2strmdydp = dvardy(dstrmdp, lat, ydim)

    tnx = U * (dstrmdx ** 2 - strm * d2strmdx2) + \
        V * (dstrmdx * dstrmdy - strm * d2strmdxdy)
    tny = U * (dstrmdx * dstrmdy - strm * d2strmdxdy) + \
        V * (dstrmdy ** 2 - strm * d2strmdy2)
    tnz = f ** 2 / S ** 2 * (
        U * (dstrmdx * dstrmdp - strm * d2strmdxdp) -
        V * (dstrmdy * dstrmdp - strm * d2strmdydp))

    tnx = 0.5 * tnx / np.abs(U + 1j * V)
    tny = 0.5 * tny / np.abs(U + 1j * V)

    tnxy = np.sqrt(tnx ** 2 + tny ** 2)
    tnx = np.ma.asarray(tnx)
    tny = np.ma.asarray(tny)
    tnz = np.ma.asarray(tnz)
    tnx[(U < 0) | (tnxy > limit)] = np.ma.masked
    tny[(U < 0) | (tnxy > limit)] = np.ma.masked
    tnz[(U < 0) | (tnxy > limit)] = np.ma.masked

    return tnx, tny, tnz


def w_to_omega(w, pres, tempk):
    """
    Compute vertical velocity on isobaric surfaces

    :param w: Input vertical velocity (m s-1)
    :param pres: Input half level pressures (full field) in Pa
    :param tempk: Input temperature (K)
    :return:
    """

    omeg = -((pres * g) / (Rd * tempk)) * w
    return omeg
