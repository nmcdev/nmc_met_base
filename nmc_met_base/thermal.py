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


# -----------------------------
# Calculate web-bulb temperature
# refer to https://github.com/smartlixx/WetBulb/blob/master/WetBulb.py

def QSat_2(T_k, p_t):
    """
    [es_mb,rs,de_mbdT,dlnes_mbdT,rsdT,foftk,fdt]=QSat_2(T_k, p_t)
    DESCRIPTION:
      Computes saturation mixing ratio and the change in saturation
      mixing ratio with respect to temperature.  Uses Bolton eqn 10, 39.
      Davies-Jones eqns 2.3,A.1-A.10
      Reference:  Bolton: The computation of equivalent potential temperature. 
  	      Monthly Weather Review (1980) vol. 108 (7) pp. 1046-1053
 	      Davies-Jones: An efficient and accurate method for computing the 
          wet-bulb temperature along pseudoadiabats. Monthly Weather Review 
          (2008) vol. 136 (7) pp. 2764-2785
 
    INPUTS:
      T_k        temperature (K)
      p_t        surface atmospheric pressure (pa)
      T_k and p_t should be arrays of identical dimensions.
    OUTPUTS:
      es_mb      vapor pressure (pa)
      rs       	 humidity (kg/kg)
      de_mbdT    d(es)/d(T)
      dlnes_mbdT dln(es)/d(T)
      rsdT     	 d(qs)/d(T)
      foftk      Davies-Jones eqn 2.3
      fdT     	 d(f)/d(T)
    Ported from HumanIndexMod by Jonathan R Buzan 08/08/13
    MATLAB port by Robert Kopp
    Last updated by Robert Kopp, robert-dot-kopp-at-rutgers-dot-edu, Wed Sep 02 22:22:25 EDT 2015
    """

    lambd_a = 3.504    	# Inverse of Heat Capacity
    alpha = 17.67 	    # Constant to calculate vapour pressure
    beta = 243.5		# Constant to calculate vapour pressure
    epsilon = 0.6220	# Conversion between pressure/mixing ratio
    es_C = 6.112		# Vapour Pressure at Freezing STD (mb)
    vkp = 0.2854		# Heat Capacity
    y0 = 3036		    # constant
    y1 = 1.78		    # constant
    y2 = 0.448		    # constant
    Cf = 273.15         # Freezing Temp (K)
    refpres = 1000	    # Reference Pressure (mb)

# Constants used to calculate es(T)
# Clausius-Clapeyron
    p_tmb = p_t*0.01
    tcfbdiff = T_k - Cf + beta
    es_mb = es_C * np.exp(alpha*(T_k - Cf)/(tcfbdiff))
    dlnes_mbdT = alpha * beta/((tcfbdiff)*(tcfbdiff))
    pminuse = p_tmb - es_mb
    de_mbdT = es_mb * dlnes_mbdT
    d2e_mbdT2 = dlnes_mbdT * (de_mbdT - 2*es_mb/(tcfbdiff))

# Constants used to calculate rs(T)
    ndimpress = (p_tmb/refpres)**vkp
    p0ndplam = refpres * ndimpress**lambd_a
    rs = epsilon * es_mb/(p0ndplam - es_mb + np.spacing(1)) #eps)
    prersdt = epsilon * p_tmb/((pminuse)*(pminuse))
    rsdT = prersdt * de_mbdT
    d2rsdT2 = prersdt * (d2e_mbdT2 -de_mbdT*de_mbdT*(2/(pminuse)))

# Constants used to calculate g(T)
    rsy2rs2 = rs + y2*rs*rs
    oty2rs = 1 + 2.0*y2*rs
    y0tky1 = y0/T_k - y1
    goftk = y0tky1 * (rs + y2 * rs * rs)
    gdT = - y0 * (rsy2rs2)/(T_k*T_k) + (y0tky1)*(oty2rs)*rsdT
    d2gdT2 = 2.0*y0*rsy2rs2/(T_k*T_k*T_k) - 2.0*y0*rsy2rs2*(oty2rs)*rsdT + \
        y0tky1*2.0*y2*rsdT*rsdT + y0tky1*oty2rs*d2rsdT2

# Calculations for used to calculate f(T,ndimpress)
    foftk = ((Cf/T_k)**lambd_a)*(np.abs(1 - es_mb/p0ndplam))**(vkp*lambd_a)* \
        np.exp(-lambd_a*goftk)
    fdT = -lambd_a*(1.0/T_k + vkp*de_mbdT/pminuse + gdT)
    d2fdT2 = lambd_a*(1.0/(T_k*T_k) - vkp*de_mbdT*de_mbdT/(pminuse*pminuse) - \
        vkp*d2e_mbdT2/pminuse - d2gdT2)

# avoid bad numbers
    rs[rs>1]=np.nan
    rs[rs<0]=np.nan

    return es_mb,rs,de_mbdT,dlnes_mbdT,rsdT,foftk,fdT


def WetBulb(TemperatureC, Pressure, Humidity, HumidityMode=0):
    """ 
    INPUTS:
      TemperatureC	   2-m air temperature (degrees Celsius)
      Pressure	       Atmospheric Pressure (Pa)
      Humidity         Humidity -- meaning depends on HumidityMode
      HumidityMode
        0 (Default): Humidity is specific humidity (kg/kg)
        1: Humidity is relative humidity (#, max = 100)
      TemperatureC, Pressure, and Humidity should either be scalars or arrays of
        identical dimension.
    OUTPUTS:
      Twb	    wet bulb temperature (C)
      Teq	    Equivalent Temperature (K)
      epott 	Equivalent Potential Temperature (K)
    """
    SHR_CONST_TKFRZ = 273.15
    TemperatureK = TemperatureC + SHR_CONST_TKFRZ

    constA = 2675 	 # Constant used for extreme cold temparatures (K)
    grms = 1000 	 # Gram per Kilogram (g/kg)
    p0 = 1000   	 # surface pressure (mb)

    kappad = 0.2854	 # Heat Capacity

    C = SHR_CONST_TKFRZ		# Freezing Temperature
    pmb = Pressure*0.01   	# pa to mb
    T1 = TemperatureK		# Use holder for T

    es_mb,rs = QSat_2(TemperatureK, Pressure)[0:2] # first two returned values

    if HumidityMode==0:
        qin = Humidity                   # specific humidity
        relhum = 100.0 * qin/rs          # relative humidity (%)
        vapemb = es_mb * relhum * 0.01   # vapor pressure (mb) 
    elif HumidityMode==1:
        relhum = Humidity                # relative humidity (%)
        qin = rs * relhum * 0.01         # specific humidity
        vapemb = es_mb * relhum * 0.01   # vapor pressure (mb) 
    #end

    mixr = qin * grms          # change specific humidity to mixing ratio (g/kg)
   
    #-----------------------------------------------------------------------

    # Calculate Equivalent Pot. Temp (pmb, T, mixing ratio (g/kg), pott, epott)	
    # Calculate Parameters for Wet Bulb Temp (epott, pmb)
    pnd = (pmb/p0)**(kappad)
    D = 1.0/(0.1859*pmb/p0 + 0.6512)
    k1 = -38.5*pnd*pnd + 137.81*pnd - 53.737
    k2 = -4.392*pnd*pnd + 56.831*pnd - 0.384

    # Calculate lifting condensation level.  first eqn 
    # uses vapor pressure (mb)
    # 2nd eqn uses relative humidity.  
    # first equation: Bolton 1980 Eqn 21.
    #   tl = (2840/(3.5*log(T1) - log(vapemb) - 4.805)) + 55;
    # second equation: Bolton 1980 Eqn 22.  relhum = relative humidity
    tl = (1.0/((1.0/((T1 - 55))) - (np.log(relhum/100.0)/2840.0))) + 55.0

    # Theta_DL: Bolton 1980 Eqn 24.
    theta_dl = T1*((p0/(pmb-vapemb))**kappad) * ((T1/tl)**(mixr*0.00028))
    # EPT: Bolton 1980 Eqn 39.  
    epott = theta_dl * np.exp(((3.036/tl)-0.00178)*mixr*(1 + 0.000448*mixr))
    Teq = epott*pnd			 # Equivalent Temperature at pressure
    X = (C/Teq)**3.504

    # Calculates the regime requirements of wet bulb equations.
    invalid = (Teq > 600) + (Teq < 200)
    hot = (Teq > 355.15)
    cold = ((X>=1) * (X<=D))
    X[invalid==1] = np.nan 
    Teq[invalid==1] = np.nan

    # Calculate Wet Bulb Temperature, initial guess
    # Extremely cold regime if X.gt.D then need to 
    # calculate dlnesTeqdTeq 
    es_mb_teq,rs_teq,de_mbdTeq, dlnes_mbdTeq, rsdTeq, foftk_teq, fdTeq = QSat_2(Teq, Pressure)
    wb_temp = Teq - C - ((constA*rs_teq)/(1 + (constA*rs_teq*dlnes_mbdTeq)))
    sub=np.where(X<=D)
    wb_temp[sub] = (k1[sub] - 1.21 * cold[sub] - 1.45 * hot[sub] - (k2[sub] - 1.21 * cold[sub]) * X[sub] + (0.58 / X[sub]) * hot[sub])
    wb_temp[invalid==1]=np.nan

    # Newton-Raphson Method

    maxiter = 3
    iter = 0
    delta = 1e6*np.ones_like(wb_temp)

    while (np.max(delta)>0.01) and (iter<=maxiter):
        es_mb_wb_temp,rs_wb_temp,de_mbdwb_temp, dlnes_mbdwb_temp, rsdwb_temp, foftk_wb_temp, fdwb_temp = QSat_2(wb_temp + C, Pressure)
        delta = (foftk_wb_temp - X)/fdwb_temp  #float((foftk_wb_temp - X)/fdwb_temp)
        delta = np.where(delta<10., delta, 10.) #min(10,delta)
        delta = np.where(delta>-10., delta, -10.) #max(-10,delta)
        wb_temp = wb_temp - delta
        wb_temp[invalid==1] = np.nan
        Twb = wb_temp
        iter = iter+1
    #end
    
    # ! 04-06-16: Adding iteration constraint.  Commenting out original code.
    # but in the MATLAB code, for sake of speed, we only do this for the values
    # that didn't converge

    if 1: #ConvergenceMode:
        
        convergence = 0.00001
        maxiter = 20000

        es_mb_wb_temp,rs_wb_temp,de_mbdwb_temp, dlnes_mbdwb_temp, rsdwb_temp, foftk_wb_temp, fdwb_temp = QSat_2(wb_temp + C, Pressure)
        delta = (foftk_wb_temp - X)/fdwb_temp  #float((foftk_wb_temp - X)/fdwb_temp)
        subdo = np.where(np.abs(delta)>convergence) #find(abs(delta)>convergence)

        iter = 0
        while (len(subdo)>0) and (iter<=maxiter):
            iter = iter + 1
            wb_temp[subdo] = wb_temp[subdo] - 0.1*delta[subdo]
            es_mb_wb_temp,rs_wb_temp,de_mbdwb_temp, dlnes_mbdwb_temp, rsdwb_temp, foftk_wb_temp, fdwb_temp = QSat_2(wb_temp[subdo]+C, Pressure[subdo])
            delta = 0 * wb_temp
            delta[subdo] = (foftk_wb_temp - X[subdo])/fdwb_temp #float((foftk_wb_temp - X[subdo])/fdwb_temp)
            subdo = np.where(np.abs(delta)>convergence) #find(abs(delta)>convergence);

        Twb = wb_temp
        if any(map(len,subdo)): #len(subdo)>0:
            print(len(subdo))
            Twb[subdo] = TemperatureK[subdo]-C
            #print(subdo)
            for www in subdo[0]:
            #    print(www)
                print('WARNING-Wet_Bulb failed to converge. Setting to T: WB, P, T, RH, Delta: %0.2f, %0.2f, %0.1f, %0.2g, %0.1f'%(Twb[www], Pressure[www], \
                    TemperatureK[www], relhum[www], delta[www]))
    
    #Twb=float(Twb)
    return Twb,Teq,epott
