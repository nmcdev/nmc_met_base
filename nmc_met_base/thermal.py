# _*_ coding: utf-8 _*_

# Copyright (c) 2020 NMC Developers.
# Distributed under the terms of the GPL V3 License.

"""
  Calculate thermal parameters.

  refer to:
  https://github.com/scavallo/python_scripts/blob/master/utils/weather_modules.py
"""

import numpy as np
import xarray as xr
import scipy.ndimage as ndimage
from scipy.ndimage.filters import generic_filter as gf
from nmc_met_base import arr, constants, grid


# define constants
# https://github.com/tomerburg/metlib/blob/master/diagnostics/met_functions.py
Rd = constants.rd #J/kg/K
Rv = constants.rv #J/kg/K
RvRd = Rv / Rd
Cp = constants.cp #J/kg/K
g = constants.g0 #m/s2
Lv = constants.Lv #Latent heat of vaporization
Lf = constants.Lf
Talt = constants.Talt
Tfrez = constants.Tfrez
kappa = constants.kappa #Rd/Cp
eo = constants.eo
epsil = constants.epsil
rotation_rate = constants.omega #rotation period of Earth (1/s)
pi = np.pi #pi


def vapor_pressure(temp): 
    """
    Returns vapor pressure in units of hPa
    https://github.com/tomerburg/metlib/blob/master/diagnostics/met_functions.py
    
    Args:
        temp ([type]): 2D array of temp or dewpoints (K)
    
    Returns:
        e = Vapor pressure = dewpoint
        es = Saturated vapor pressure = temperature
    """
    #Calculate vapor pressure in hPa\
    part1 = np.multiply(17.67,np.subtract(temp,273.15))
    part2 = np.add(np.subtract(temp,273.15),243.5)
    part3 = np.divide(part1,part2)
    part4 = np.exp(part3)
    e = np.multiply(6.112,part4)
    
    return e


def claus_clap(temp):
    """
    Compute saturation vapor pressure

    :param temp: Input temperature (K)
    :return:  Output satuation vapor pressure (Pa)
    """

    esat = (eo * np.exp((Lv / Rv) * (1.0 / Tfrez - 1 / temp))) * 100.
    return esat


def claus_clap_ice(temp):
    """
    Compute saturation vapor pressure over ice

    :param temp: Input temperature (K)
    :return: Output satuation vapor pressure of ice (Pa)
    """

    a = 273.16 / temp
    exponent = -9.09718 * (a - 1.) - 3.56654 * np.log10(a) + \
        0.876793 * (1. - 1. / a) + np.log10(6.1071)
    esi = 10 ** exponent
    esi = esi * 100
    return esi


def sat_vap(temp):
    """
    Compute saturation vapor pressure

    :param temp: Input temperature (K)
    :return: Output satuation vapor pressure (Pa)
    """

    [iinds] = np.where(temp < 273.15)
    [linds] = np.where(temp >= 273.15)
    esat = np.zeros_like(temp).astype('f')

    nice = len(iinds)
    nliq = len(linds)

    tempc = temp - 273.15
    if nliq > 1:
        esat[linds] = 6.112 * np.exp(17.67 * tempc[linds] / (
            tempc[linds] + 243.12)) * 100.
    else:
        if nliq > 0:
            esat = 6.112 * np.exp(17.67 * tempc / (tempc + 243.12)) * 100.
    if nice > 1:
        esat[iinds] = 6.112 * np.exp(22.46 * tempc[iinds] / (
            tempc[iinds] + 272.62)) * 100.
    else:
        if nice > 0:
            esat = 6.112 * np.exp(22.46 * tempc / (tempc + 272.62)) * 100.
    return esat


def mixratio(temp,pres):
    """
    Calculate mixing ratio, returns in units of kg/kg
    https://github.com/tomerburg/metlib/blob/master/diagnostics/met_functions.py
    
    Args:
        temp ([type]): 2D array of temperature or dewpoint (K)
        pres ([type]): integer specifying pressure level (hPa)
    
    Returns:
        ws = saturated mixing ratio = temperature (@erified)
        w = mixing ratio = dewpoint
    """
    
    #Calculate vapor pressure in hPa
    e = vapor_pressure(temp)
    
    #w = 0.622 * (e / (pres))
    w = np.multiply(0.622,np.divide(e,pres))
    
    return w


def satur_mix_ratio(es, pres):
    """
    Compute saturation mixing ratio

    :param es: Input saturation vapor pressure (Pa)
    :param pres:  Input air pressure (Pa)
    :return: Output saturation mixing ratio
    """

    ws = 0.622 * (es / (pres - es))
    return ws


def td_to_mixrat(tdew, pres):
    """
    Convert from dew point temperature to water vapor mixing ratio.

    :param tdew: Input dew point temperature (K)
    :param pres:  Input pressure (Pa)
    :return: Output water vapor mixing ratio (kg/kg)
    """

    pres = pres / 100
    mixrat = eo / (pres * RvRd) * np.exp((Lv / Rv) * ((1 / Tfrez) - (1 / tdew)))
    return mixrat


def mixrat_to_td(qvap, pres):
    """
    Convert from water vapor mixing ratio to dewpoint temperature.

    :param qvap: Input water vapor mixing ratio (kg/kg)
    :param pres: Input pressure (Pa)
    :return: Output dewpoint temperature (K)
    """

    pres = pres / 100.
    evap = qvap * pres * RvRd
    tdew = 1 / ((1 / Tfrez) - (Rv / Lv) * np.log(evap / eo))
    return tdew


def virtual_temp_from_mixr(tempk, mixr):
    """
    Virtual Temperature

    :param tempk: Temperature (K)
    :param mixr: Mixing Ratio (kg/kg)
    :return:  Virtual temperature (K)
    """

    return tempk * (1.0 + 0.6 * mixr)


def wetbulb(temp,dwpt,pres):
    """
    Calculate wetbulb temperature, returns in Kelvin
    https://github.com/tomerburg/metlib/blob/master/diagnostics/met_functions.py

    Args:
        temp ([type]): 2D array of temperature (K)
        dwpt ([type]): 2D array of Dewpoints (K)
        pres ([type]): integer specifying pressure level (hPa)
    """
    #Calculate mixing ratios
    ws = mixratio(temp,pres)
    w = mixratio(dwpt,pres)

    #Formula used: Tw = T - (Lv/cp)(ws-w)\
    part1 = np.divide(Lv,Cp)
    part2 = np.subtract(ws,w)
    part3 = np.multiply(part1,part2)
    wb = np.subtract(temp,part3)
    
    return wb


def wetbulb_q(temp,q,pres):
    """
    Calculate wetbulb temperature, returns in Kelvin
    https://github.com/tomerburg/metlib/blob/master/diagnostics/met_functions.py
    
    Args:
        temp ([type]): 2D array of temperature (K)
        q ([type]): 2D array of specific humidity (K)
        pres ([type]): integer specifying pressure level (hPa)
    """
    
    #Calculate mixing ratios
    ws = mixratio(temp,pres)
    w = np.divide(q,np.add(q,1))
    w = np.divide(w,100.0)

    #Formula used: Tw = T - (Lv/cp)(ws-w) 
    part1 = np.divide(Lv,Cp)
    part2 = np.subtract(ws,w)
    part3 = np.multiply(part1,part2)
    wb = np.subtract(temp,part3)
    
    return wb


def latentc(tempk):
    """
    Latent heat of condensation (vapourisation)
    http://en.wikipedia.org/wiki/Latent_heat#Latent_heat_for_condensation_of_water

    :param tempk: Temperature (K)
    :return: L_w (J/kg)
    """

    tempc = tempk - 273.15
    return 1000 * (
        2500.8 - 2.36 * tempc + 0.0016 * tempc ** 2 -
        0.00006 * tempc ** 3)


def relh_temp(temp,dwpt):
    """
    Compute relative humidity given temperature and dewpoint
    https://github.com/tomerburg/metlib/blob/master/diagnostics/met_functions.py
    
    Args:
        temp ([type]): 2D arrays or integer of temperature (K)
        dwpt ([type]): 2D arrays or integer of dewpoint (K)
    
    Returns:
        Returns in units of percent (100 = 100%)
    """
    
    #Compute actual and saturated vapor pressures
    e = vapor_pressure(dwpt)
    es = vapor_pressure(temp)
    
    #Compute RH
    rh = relh(e,es)
    
    return rh


def relh(w,ws):
    """
    Compute relative humidity given mixing ratio or vapor pressure
    https://github.com/tomerburg/metlib/blob/master/diagnostics/met_functions.py
    
    Args:
        w ([type]): actual mixing ratios (or vapor pressures)
        ws ([type]): saturated mixing ratios (or vapor pressures)
    
    Returns:
        Returns in units of percent (100 = 100%)
    """
    
    #Compute RH
    rh = np.multiply(np.divide(w,ws),100.0)
    
    return rh


def theta(temp,pres):
    """
    Compute the potential temperature
    https://github.com/tomerburg/metlib/blob/master/diagnostics/met_functions.py
    
    Args:
        temp (2D array): scalar temperature field (Kelvin)
        pres (float): pressure level (hPa)
    
    Returns:
        Returns in units of Kelvin
    """

    #Compute theta using Poisson's equation
    refpres = np.divide(1000.0,pres)
    refpres = np.power(refpres,kappa)
    theta = np.multiply(temp,refpres)
    
    return theta


def temp_to_theta(temp, pres, p0=100000.):
    """
    Compute potential temperature.

    :param temp: Input temperature (K)
    :param pres: Input pressure (Pa)
    :param p0: reference pressure (Pa)
    :return: potential temperature (K)
    """

    return temp * (p0 / pres) ** 0.286


def theta_to_temp(theta, pres, p0=100000.):
    """
    Compute temperature.

    :param theta: Input potential temperature (K)
    :param pres: Input pressure (Pa)
    :param p0: reference pressure (Pa)
    :return: Output temperature (K)
    """

    return theta * (pres / p0) ** 0.286


def thetae(temp,dwpt,pres):
    """
    Compute the saturated potential temperature, following AMS methodology
    See: http://glossary.ametsoc.org/wiki/Equivalent_potential_temperature
    https://github.com/tomerburg/metlib/blob/master/diagnostics/met_functions.py
    
    Args:
        temp (2D array): scalar temperature field (Kelvin)
        dwpt (2D array): scalar dew point temperature field (Kelvin)
        pres (float): pressure level (hPa)
    
    Returns:
        Returns in units of Kelvin
    """
    
    #Calculate potential temp & saturated mixing ratio
    thta = theta(temp,pres)
    ws = mixratio(dwpt,pres)
    rh = np.divide(relh_temp(temp,dwpt),100.0)
    
    #Calculate potential temperature
    term1 = thta
    term2 = np.power(rh,np.divide(-1*np.multiply(Rv,ws),Cp))
    term3 = np.exp(np.divide(np.multiply(Lv,ws),np.multiply(Cp,temp)))
    
    thte = np.multiply(term1,np.multiply(term2,term3))
    
    return thte


def thetae_qv(thta, temp, qv):
    """
    Compute equivalent potential temperature

    :param thta: Input potential temperature of column (K)
    :param temp: Input temperature (K) at LCL
    :param qv: Input mixing ratio of column (kg kg-v1)
    :return: Output equivalent potential temperature (K)
    """

    thout = thta * np.exp((Lv * qv) / (Cp * temp))
    return thout


def dewpoint(temp,rh):
    """
    Calculates the dewpoint given RH and temperature
    https://github.com/tomerburg/metlib/blob/master/diagnostics/met_functions.py
    
    Returns:
        Returns dewpoint in Kelvin
    """

    #Source: http://andrew.rsmas.miami.edu/bmcnoldy/Humidity.html
    part1 = 243.04 * (np.log(rh/100.0) + ((17.625 * (temp-273.15)) / (243.04 + (temp-273.15))))
    part2 = 17.65 - np.log(rh/100.0) - ((17.625 * (temp-273.15)) / (243.04 + (temp-273.15)))
    
    Td = (part1 / part2) + 273.15
    
    return Td

def spechum_to_td(spechum, pres):
    """
    Convert from specific humidity to dewpoint temperature

    :param spechum: Input specific humidity in (kg/kg)
    :param pres: Input pressure (Pa)
    :return: Output dewpoint temperature (K)
    """

    qvap = (spechum / (1 - spechum))
    pres = pres / 100
    evap = qvap * pres * RvRd
    tdew = 1 / ((1 / Tfrez) - (Rv / Lv) * np.log(evap / eo))
    return tdew


def specific_humidity(temp,pres,rh):
    """
    Calculate specific humidity
    
    Args:
        temp ([type]): [description]
        pres ([type]): [description]
        rh ([type]): [description]
    
    Returns:
        Returns q in kg/kg
    """
    
    #saturated vapor pressure in Pa
    es = vapor_pressure(temp) * 100.0
    
    #get e from RH (in decimals) in Pa
    e = (rh / 100.0) * es
    
    #Approximate specific humidity q ~ w
    w = 0.622 * (e / pres) #used to be 0.622
    q = w
    
    return q


def moist_lapse(ws, temp):
    """
    Compute moist adiabatic lapse rate

    :param ws: Input saturation mixing ratio (kg kg-1)
    :param temp: Input air temperature (K)
    :return: Output moist adiabatic lapse rate
    """

    return (g / Cp) * ((1.0 + Lv * ws) / (Rd * temp)) / (
        1.0 + (ws * (Lv ** 2.0) / (Cp * Rv * temp ** 2.0)))


def gamma_w(tempk, pres, e=None):
    """
    Function to calculate the moist adiabatic lapse rate (deg K/Pa) based
    on the temperature, pressure, and rh of the environment.

    :param tempk: Temperature (K)
    :param pres: Input pressure (Pa)
    :param e: Input saturation vapor pressure (Pa)
    :return: The moist adiabatic lapse rate (Dec K/Pa)
    """

    es = sat_vap(tempk)
    ws = satur_mix_ratio(es, pres)

    if e is None:
        # assume saturated
        e = es

    w = satur_mix_ratio(e, pres)

    tempv = virtual_temp_from_mixr(tempk, w)
    latent = latentc(tempk)

    A = 1.0 + latent * ws / (Rd * tempk)
    B = 1.0 + epsil * latent * latent * ws / (Cp * Rd * tempk * tempk)
    Rho = pres / (Rd * tempv)
    gamma = (A / B) / (Cp * Rho)
    return gamma


def mslp(pres,hght,temp,lapse=6.5):
    """
    Approximate the MSLP from the Hypsometric equation
    https://github.com/tomerburg/metlib/blob/master/diagnostics/met_functions.py
    
    Args:
        pres ([type]): surface pressure (Pa)
        hght ([type]): surface height (m)
        temp ([type]): 2m temperature (K)
        lapse (float, optional): (either constant or array in the same dimension as the other
                                 variables. If none, assumed to be moist adiabatic lapse rate
                                 6.5 K/km.). Defaults to 6.5.
    """
    
    #Approximate the virtual temperature as the average temp in the layer
    #using a 6.5 K/km lapse rate
    tslp = (lapse/1000)*hght + temp
    Tavg = 0.5*(tslp + temp)
    
    #Use the hypsometric equation to solve for lower pressure
    mslp = pres * np.exp((hght * g) / (Rd * Tavg))
    
    return mslp


def dry_parcel_ascent(startpp, starttk, starttdewk, nsteps=101):
    """
    Lift a parcel dry adiabatically from startp to LCL.

    :param startpp: Pressure of parcel to lift in Pa
    :param starttk: Temperature of parcel at startp in K
    :param starttdewk: Dewpoint temperature of parcel at startp in K
    :param nsteps:
    :return: presdry, tempdry, pressure (Pa) and temperature (K) along
             dry adiabatic ascent of parcel tempiso is in K
             T_lcl, P_lcl, Temperature and pressure at LCL
    """

    assert starttdewk <= starttk

    startt = starttk - 273.15
    starttdew = starttdewk - 273.15
    startp = startpp / 100.

    if starttdew == startt:
        return np.array([startp]), np.array([startt]), np.array([starttdew]),

    Pres = np.logspace(np.log10(startp), np.log10(600), nsteps)

    # Lift the dry parcel
    T_dry = (starttk * (Pres / startp) ** (Rd / Cp)) - 273.15

    # Mixing ratio isopleth
    starte = sat_vap(starttdewk)
    startw = satur_mix_ratio(starte, startpp)
    ee = Pres * startw / (.622 + startw)
    T_iso = 243.5 / (17.67 / np.log(ee / 6.112) - 1.0)

    # Solve for the intersection of these lines (LCL).
    # interp requires the x argument (argument 2)
    # to be ascending in order!
    P_lcl = np.interp(0, T_iso - T_dry, Pres)
    T_lcl = np.interp(P_lcl, Pres[::-1], T_dry[::-1])

    presdry = np.linspace(startp, P_lcl)
    tempdry = np.interp(presdry, Pres[::-1], T_dry[::-1])
    tempiso = np.interp(presdry, Pres[::-1], T_iso[::-1])

    return (
        presdry * 100., tempdry + 273.15,
        tempiso + 273.15, T_lcl + 273.15, P_lcl * 100.)


def moist_ascent(startpp, starttk, ptop=100, nsteps=501):
    """
    Lift a parcel moist adiabatically from startp to endp.

    :param startpp: Pressure of parcel to lift in Pa
    :param starttk: Temperature of parcel at startp in K
    :param ptop: Top pressure of parcel to lift in Pa
    :param nsteps:
    :return:
    """

    startp = startpp / 100.  # convert to hPa
    startt = starttk - 273.15  # convert to deg C

    preswet = np.logspace(np.log10(startp), np.log10(ptop), nsteps)

    temp = startt
    tempwet = np.zeros(preswet.shape)
    tempwet[0] = startt
    for ii in range(preswet.shape[0] - 1):
        delp = preswet[ii] - preswet[ii + 1]
        temp = temp - 100. * delp * gamma_w(
            temp + 273.15, (preswet[ii] - delp / 2) * 100.)
        tempwet[ii + 1] = temp

    return preswet * 100., tempwet + 273.15


def pottemp_3D(temp, lev, zdim, punit=100., p0=100000.):
    """
    Calculate potential temperature.

    :param temp: array_like, temperature [K].
    :param lev: array_like, level [punit*Pa].
    :param zdim: vertical dimensions.
    :param punit: pressure level punit.
    :param p0: reference pressure.
    :return: ndarray.
    """

    temp = np.asarray(temp)
    ndim = temp.ndim
    p = arr.expand(lev, ndim, axis=zdim)

    out = temp * ((p0/p/punit)**kappa)

    return out


def stability(temp, lev, zdim, punit=100.):
    """
    P level coordinates stability (Brunt-Vaisala).

    :param temp: array_like, temperature.
    :param lev: array_like, pressure level.
    :param zdim: vertical dimension axis.
    :param punit: pressure unit.
    :return: ndarray.
    """

    temp = np.asarray(temp)
    ndim = temp.ndim
    p = arr.expand(lev, ndim, axis=zdim) * punit
    theta = pottemp_3D(temp, lev, zdim, punit=punit)
    alpha = Rd * temp / p
    N = -alpha * grid.dvardp(np.log(theta), lev, zdim, punit=punit)

    return N


def atmosphere(alt):
    """python-standard-atmosphere
    Python package for creating pressure and temperature profiles of
    the standard atmosphere for use with geophysical models. This
    package will only calcualate good values up to 86km.
    https://github.com/pcase13/python-standard-atmosphere/blob/master/standard.py
    Arguments:
        alt {scalar} -- altitude, hPa

    Returns:
        scalar -- standard-atmosphere
    """

    # Constants
    REARTH = 6369.0  # radius of earth
    GMR = 34.163195  # hydrostatic constant
    NTAB = 8  # number of entries in defining tables

    # Define defining tables
    htab = [0.0, 11.0, 20.0, 32.0, 47.0, 51.0, 71.0, 84.852]
    ttab = [288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65, 186.946]
    ptab = [
        1.0, 2.233611E-1, 5.403295E-2, 8.5666784E-3, 1.0945601E-3,
        6.6063531E-4, 3.9046834E-5, 3.68501E-6]
    gtab = [-6.5, 0.0, 1.0, 2.8, 0.0, -2.8, -2.0, 0.0]

    # Calculate
    h = alt*REARTH/(alt+REARTH)  # convert to geopotential alt
    i = 1
    j = NTAB

    while(j > i+1):
        k = int((i+j)/2)  # integer division
        if(h < htab[k]):
            j = k
        else:
            i = k
    print(i)
    tgrad = gtab[i]
    tbase = ttab[i]
    deltah = h-htab[i]
    tlocal = tbase + tgrad * deltah
    theta = tlocal/ttab[0]

    if(tgrad == 0.0):
        delta = ptab[i] * np.exp(-1*GMR*deltah/tbase)
    else:
        delta = ptab[i] * (tbase/tlocal)**(GMR/tgrad)

    sigma = delta/theta
    return sigma, delta, theta


def get_standard_atmosphere_1d(z):
    NZ = z.shape[0]
    p0 = 1.013250e5
    t0 = 288.15
    p = np.zeros(z.shape)
    t = np.zeros(z.shape)

    for i in np.arange(NZ):
        sigma, delta, theta = atmosphere(z[i]/1000.)  # convert to km
        p[i] = p0 * delta
        t[i] = t0 * theta
    return p, t


def get_standard_atmosphere_2d(z):
    NZ = z.shape[1]
    NY = z.shape[0]
    p0 = 1.013250e5
    t0 = 288.15
    p = np.zeros(z.shape)
    t = np.zeros(z.shape)

    for i in np.arange(NY):
        for j in np.arange(NZ):
            sigma, delta, theta = atmosphere(z[i, j]/1000.)  # convert to km
            p[i, j] = p0 * delta
            t[i, j] = t0 * theta
    return p, t


def get_standard_atmosphere_3d(z):
    NZ = z.shape[2]
    NX = z.shape[0]
    NY = z.shape[1]
    p0 = 1.013250e5
    t0 = 288.15
    p = np.zeros(z.shape)
    t = np.zeros(z.shape)

    for i in np.arange(NX):
        for j in np.arange(NY):
            for k in np.arange(NZ):
                # convert to km
                sigma, delta, theta = atmosphere(z[i, j, k]/1000.)
                p[i, j, k] = p0 * delta
                t[i, j, k] = t0 * theta
    return p, t
