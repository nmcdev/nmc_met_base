# _*_ coding: utf-8 _*_

# Copyright (c) 2019 NMC Developers.
# Distributed under the terms of the GPL V3 License.

"""
Physical constants.

refer to:
http://carina.fcaglp.unlp.edu.ar/ida/archivos/tabla_constantes.pdf
"""

import sys
import datetime as _dt
import math as _math
import numpy as _np

# math constants
pi = _math.pi
d2r = pi / 180.                   # degree => radian
r2d = 180. / pi                   # radian => degree
eps = 1.e-7                       # small machine float
n_degree = 360.                   # Number of degrees in a circle
epsilon = sys.float_info.epsilon

# unit conversions
kt2ms = 0.515           # convert kts to m/s
dam2m = 10.             # convert dam to m
ms2kt = 1./0.515        # convert m/s to kts
m2dam = 0.1             # convert m to dam
km2nm = 0.54            # convert km to NM
mb2pa = 100.            # convert mb to Pa
pa2mb = 1./100.         # convert Pa to mb (*)

# Earth
Re = 6.371e+6               # earth's mean radius (Weast and Astle 1980, m)
Ae = 5.1e14                 # Area of the surface of the Earth m^2
daysec = 86400.0            # day seconds
omega = 7.2921159e-5        # earth angular velocity (radians/second)
g0 = 9.81                   # Acceleration due to gravity at sea level (N/kg)
f0 = 1.e-4                  # f-plane parameter (s**-1)
mass_e = 5.972e24           # Mass of Earth, From Serway 1992,  In kg
mass_a = 5.3e18             # Mass of the Earth’s atmosphere, kg
mass_ac = 1.017e4           # Mass of an atmospheric column kg/m^2

# Thermodynamic constants
rho0 = 1.25       # Typical density of air at sea level (kg/m**3)
md = 28.97        # Effective molecular mass for dry air (kg/kmol)
rd = 287.04       # dry gas constant (J/K/kg)
rv = 461.6        # gas constant of water vapour (J/K/kg)
cp = 1004.5       # Specific heat of dry air, constant pressure
cv = 717.5        # Specific heat of dry air, constant volume
gcp = 9.8e-3      # Dry adiabatic lapse rate K/m
K = 2.4e-2        # Thermal conductivity at 0 J m−1 s−1 K−1
kappa = rd/cp     # Poisson's constant
gamma = cp/cv
epsil = rd/rv
Talt = 288.15     # temperature at standard sea level
Tfrez = 273.15    # zero degree K
T0 = 300
P0 = 101325       # standard atmosphere pressure Pa
Pr = 1000.0
lapsesta = 6.5e-3  # lower atmosphere averaged T lapse rate (degree/m)

# water constants
rhow = 1000.    # Density of liquid water at 0C, In kg / m^3
rhoi = 9.17e+2  # Density of ice at 0C kg / m^3
mw = 18.016     # Molecular mass for H2O kg / kmol
meps = 0.622    # Molecular weight ratio of H2O to dry air
cpw = 1952.     # Specific heat of water vapor at constant pressure J/deg/kg
cvw = 1463.     # Specific heat of water vapor at constant volume J/deg/kg
cw = 4218.      # Specific heat of liquid water at 0C J/K/kg
ci = 2106.      # Specific heat of ice at 0C J/K/kg
Lv = 2.5e+6     # Latent heat of vaporization at 0 degree (J/kg)
Ls = 2.85e+6    # Latent heat of sublimation (H2O) (J/kg)
Lf = 3.34e+5    # Latent heat of fusion (H2O) (J/kg)
eo = 6.11

# time constants
base_time = _dt.datetime(1900, 1, 1, 0, 0, 0)
base_time_units = 'days since 1900-01-01 00 UTC'

# map region limit
limit_china = (73.6667, 135.042, 3.86667, 53.5500)
limit_china_continents = (73., 136., 18., 54.)


# functions for geophysical constants
#

def earth_f(lat):
    """
    Compute f parameters.
    f = 2 * earth_omega*sin(lat)

    :param lat: array_like, latitude (degrees) to be converted
    :return:

    >>> earth_f(_np.array([-30., 0., 30.))
    """
    return 2.*omega*_np.sin(pi/180.*lat)


def earth_beta(lat):
    """
    Compute beta parameter.
    beta = df/dy = 2*earth_omega/earth_radius*cos(lat)

    :param lat: array_like, latitude (degrees) to be converted
    :return: float or array of floats, beta parameter

    >>> earth_beta(__np.array([-30., 0., 30.))
    """
    return 2.*omega/Re*_np.cos(pi/180.*lat)


def dlon2dx(dlon, clat):
    """
    经度差在局地直角坐标系中的实际距离.
    """
    return _np.deg2rad(dlon)*Re*_np.cos(_np.deg2rad(clat))


def dx2dlon(dx, clat):
    """
    实际距离转换为经度差。
    """
    return _np.rad2deg(dx/Re/_np.cos(_np.deg2rad(clat)))


def dlat2dy(dlat):
    """
    纬度差在局地直角坐标系中的实际距离.
    """
    return _np.deg2rad(dlat)*Re


def dy2dlat(dy):
    """
    实际距离转换为纬度差。
    """
    return _np.rad2deg(dy/Re)
