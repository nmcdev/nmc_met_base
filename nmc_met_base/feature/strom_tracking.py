# _*_ coding: utf-8 _*_

# Copyright (c) 2022 NMC Developers.
# Distributed under the terms of the GPL V3 License.

"""
Automated detection and tracking of atmospheric storms (cyclones) 
and high-pressure systems (anticyclones), given a series of mean 
sea level pressure maps.

notes: This code as been applied to 6-hourly mean sea level pressure 
maps from NCEP Twentieth Century Reanalysis (20CR). The code at the 
top of storm_detection.py will need to be modified for use with another 
data source, as will various other function options as necessary 
(e.g. time step, grid resolution, etc). 

Adapted from https://github.com/ecjoliver/stormTracking
"""

import numpy as np
import scipy as sp
import scipy.ndimage as ndimage
from datetime import date
from itertools import repeat


def distance_matrix(lons,lats):
    '''Calculates the distances (in km) between any two cities based on the formulas
    c = sin(lati1)*sin(lati2)+cos(longi1-longi2)*cos(lati1)*cos(lati2)
    d = EARTH_RADIUS*Arccos(c)
    where EARTH_RADIUS is in km and the angles are in radians.
    Source: http://mathforum.org/library/drmath/view/54680.html
    This function returns the matrix.'''

    EARTH_RADIUS = 6378.1
    X = len(lons)
    Y = len(lats)
    assert X == Y, 'lons and lats must have same number of elements'

    d = np.zeros((X,X))

    #Populate the matrix.
    for i2 in range(len(lons)):
        lati2 = lats[i2]
        loni2 = lons[i2]
        c = np.sin(np.radians(lats)) * np.sin(np.radians(lati2)) + \
            np.cos(np.radians(lons-loni2)) * \
            np.cos(np.radians(lats)) * np.cos(np.radians(lati2))
        d[c<1,i2] = EARTH_RADIUS * np.arccos(c[c<1])

    return d


def detect_storms(field, lon, lat, res, Npix_min, cyc, globe=False):
    '''
    Detect storms present in field which satisfy the criteria.
    Algorithm is an adaptation of an eddy detection algorithm,
    outlined in Chelton et al., Prog. ocean., 2011, App. B.2,
    with modifications needed for storm detection.
    field is a 2D array specified on grid defined by lat and lon.
    res is the horizontal grid resolution in degrees of field
    Npix_min is the minimum number of pixels within which an
    extremum of field must lie (recommended: 9).
    cyc = 'cyclonic' or 'anticyclonic' specifies type of system
    to be detected (cyclonic storm or high-pressure systems)
    globe is an option to detect storms on a globe, i.e. with periodic
    boundaries in the West/East. Note that if using this option the 
    supplied longitudes must be positive only (i.e. 0..360 not -180..+180).
    Function outputs lon, lat coordinates of detected storms
    '''

    len_deg_lat = 111.325 # length of 1 degree of latitude [km]

    # Need to repeat global field to the West and East to properly detect around the edge
    if globe:
        dl = 20. # Degrees longitude to repeat on East and West of edge
        iEast = np.where(lon >= 360. - dl)[0][0]
        iWest = np.where(lon <= dl)[0][-1]
        lon = np.append(lon[iEast:]-360, np.append(lon, lon[:iWest]+360))
        field = np.append(field[:,iEast:], np.append(field, field[:,:iWest], axis=1), axis=1)

    llon, llat = np.meshgrid(lon, lat)

    lon_storms = np.array([])
    lat_storms = np.array([])
    amp_storms = np.array([])

    # ssh_crits is an array of ssh levels over which to perform storm detection loop
    # ssh_crits increasing for 'cyclonic', decreasing for 'anticyclonic'
    ssh_crits = np.linspace(np.nanmin(field), np.nanmax(field), 200)
    ssh_crits.sort()
    if cyc == 'anticyclonic':
        ssh_crits = np.flipud(ssh_crits)

    # loop over ssh_crits and remove interior pixels of detected storms from subsequent loop steps
    for ssh_crit in ssh_crits:
 
    # 1. Find all regions with eta greater (less than) than ssh_crit for anticyclonic (cyclonic) storms (Chelton et al. 2011, App. B.2, criterion 1)
        if cyc == 'anticyclonic':
            regions, nregions = ndimage.label( (field>ssh_crit).astype(int) )
        elif cyc == 'cyclonic':
            regions, nregions = ndimage.label( (field<ssh_crit).astype(int) )

        for iregion in range(nregions):
 
    # 2. Calculate number of pixels comprising detected region, reject if not within >= Npix_min
            region = (regions==iregion+1).astype(int)
            region_Npix = region.sum()
            storm_area_within_limits = (region_Npix >= Npix_min)
 
    # 3. Detect presence of local maximum (minimum) for anticylones (cyclones), reject if non-existent
            interior = ndimage.binary_erosion(region)
            exterior = region.astype(bool) - interior
            if interior.sum() == 0:
                continue
            if cyc == 'anticyclonic':
                has_internal_ext = field[interior].max() > field[exterior].max()
            elif cyc == 'cyclonic':
                has_internal_ext = field[interior].min() < field[exterior].min()
 
    # 4. Find amplitude of region, reject if < amp_thresh
            if cyc == 'anticyclonic':
                amp_abs = field[interior].max()
                amp = amp_abs - field[exterior].mean()
            elif cyc == 'cyclonic':
                amp_abs = field[interior].min()
                amp = field[exterior].mean() - amp_abs
            amp_thresh = np.abs(np.diff(ssh_crits)[0])
            is_tall_storm = amp >= amp_thresh
 
    # Quit loop if these are not satisfied
            if np.logical_not(storm_area_within_limits * has_internal_ext * is_tall_storm):
                continue
 
    # Detected storms:
            if storm_area_within_limits * has_internal_ext * is_tall_storm:
                # find centre of mass of storm
                storm_object_with_mass = field * region
                storm_object_with_mass[np.isnan(storm_object_with_mass)] = 0
                j_cen, i_cen = ndimage.center_of_mass(storm_object_with_mass)
                lon_cen = np.interp(i_cen, range(0,len(lon)), lon)
                lat_cen = np.interp(j_cen, range(0,len(lat)), lat)
                # Remove storms detected outside global domain (lon < 0, > 360)
                if globe * (lon_cen >= 0.) * (lon_cen <= 360.):
                    # Save storm
                    lon_storms = np.append(lon_storms, lon_cen)
                    lat_storms = np.append(lat_storms, lat_cen)
                    # assign (and calculated) amplitude, area, and scale of storms
                    amp_storms = np.append(amp_storms, amp_abs)
                # remove its interior pixels from further storm detection
                storm_mask = np.ones(field.shape)
                storm_mask[interior.astype(int)==1] = np.nan
                field = field * storm_mask

    return lon_storms, lat_storms, amp_storms


def storms_list(lon_storms_a, lat_storms_a, amp_storms_a, lon_storms_c, lat_storms_c, amp_storms_c):
    '''
    Creates list detected storms
    '''

    storms = []

    for ed in range(len(lon_storms_c)):
        storm_tmp = {}
        storm_tmp['lon'] = np.append(lon_storms_a[ed], lon_storms_c[ed])
        storm_tmp['lat'] = np.append(lat_storms_a[ed], lat_storms_c[ed])
        storm_tmp['amp'] = np.append(amp_storms_a[ed], amp_storms_c[ed])
        storm_tmp['type'] = list(repeat('anticyclonic',len(lon_storms_a[ed]))) + list(repeat('cyclonic',len(lon_storms_c[ed])))
        storm_tmp['N'] = len(storm_tmp['lon'])
        storms.append(storm_tmp)

    return storms


def storms_init(det_storms, year, month, day, hour):
    '''
    Initializes list of storms. The ith element of output is
    a dictionary of the ith storm containing information about
    position and size as a function of time, as well as type.
    '''

    storms = []

    for ed in range(det_storms[0]['N']):
        storm_tmp = {}
        storm_tmp['lon'] = np.array([det_storms[0]['lon'][ed]])
        storm_tmp['lat'] = np.array([det_storms[0]['lat'][ed]])
        storm_tmp['amp'] = np.array([det_storms[0]['amp'][ed]])
        storm_tmp['type'] = det_storms[0]['type'][ed]
        storm_tmp['year'] = np.array([year[0]])
        storm_tmp['month'] = np.array([month[0]])
        storm_tmp['day'] = np.array([day[0]])
        storm_tmp['hour'] = np.array([hour[0]])
        storm_tmp['exist_at_start'] = True
        storm_tmp['terminated'] = False
        storms.append(storm_tmp)

    return storms


def len_deg_lon(lat):
    '''
    Returns the length of one degree of longitude (at latitude
    specified) in km.
    '''

    R = 6371. # Radius of Earth [km]

    return (np.pi/180.) * R * np.cos( lat * np.pi/180. )


def len_deg_lat():
    '''
    Returns the length of one degree of latitude in km.
    '''
    return 111.325 # length of 1 degree of latitude [km]


def latlon2km(lon1, lat1, lon2, lat2):
    '''
    Returns the distance, in km, between (lon1, lat1) and (lon2, lat2)
    '''

    EARTH_RADIUS = 6371. # Radius of Earth [km]
    c = np.sin(np.radians(lat1)) * np.sin(np.radians(lat2)) + np.cos(np.radians(lon1-lon2)) * np.cos(np.radians(lat1)) * np.cos(np.radians(lat2))
    d = EARTH_RADIUS * np.arccos(c)

    return d


def track_storms(storms, det_storms, tt, year, month, day, hour, dt, prop_speed=80.):
    '''
    Given a set of detected storms as a function of time (det_storms)
    this function will update tracks of individual storms at time step
    tt in variable storms
    dt indicates the time step of the underlying data (in hours)
    prop_speed indicates the maximum storm propagation speed (in km/hour)
    '''

    # List of unassigned storms at time tt

    unassigned = range(det_storms[tt]['N'])

    # For each existing storm (t<tt) loop through unassigned storms and assign to existing storm if appropriate

    for ed in range(len(storms)):

        # Check if storm has already been terminated

        if not storms[ed]['terminated']:

            # Define search region around centroid of existing storm ed at last known position
    
            x0 = storms[ed]['lon'][-1] # [deg. lon]
            y0 = storms[ed]['lat'][-1] # [deg. lat]
    
            # Find all storm centroids in search region at time tt
    
            is_near = latlon2km(x0, y0, det_storms[tt]['lon'][unassigned], det_storms[tt]['lat'][unassigned]) <= prop_speed*dt
    
            # Check if storms' type is the same as original storm
    
            is_same_type = np.array([det_storms[tt]['type'][i] == storms[ed]['type'] for i in unassigned])
    
            # Possible storms are those which are near and of the same type
    
            possibles = is_near * is_same_type
            if possibles.sum() > 0:
    
                # Of all found storms, accept only the nearest one
    
                dist = latlon2km(x0, y0, det_storms[tt]['lon'][unassigned], det_storms[tt]['lat'][unassigned])
                nearest = dist == dist[possibles].min()
                next_storm = unassigned[np.where(nearest * possibles)[0][0]]
    
                # Add coordinatse and properties of accepted storm to trajectory of storm ed
    
                storms[ed]['lon'] = np.append(storms[ed]['lon'], det_storms[tt]['lon'][next_storm])
                storms[ed]['lat'] = np.append(storms[ed]['lat'], det_storms[tt]['lat'][next_storm])
                storms[ed]['amp'] = np.append(storms[ed]['amp'], det_storms[tt]['amp'][next_storm])
                storms[ed]['year'] = np.append(storms[ed]['year'], year[tt])
                storms[ed]['month'] = np.append(storms[ed]['month'], month[tt])
                storms[ed]['day'] = np.append(storms[ed]['day'], day[tt])
                storms[ed]['hour'] = np.append(storms[ed]['hour'], hour[tt])
    
                # Remove detected storm from list of storms available for assigment to existing trajectories
    
                unassigned.remove(next_storm)

            # Terminate storm otherwise

            else:

                storms[ed]['terminated'] = True

    # Create "new storms" from list of storms not assigned to existing trajectories

    if len(unassigned) > 0:

        for un in unassigned:

            storm_tmp = {}
            storm_tmp['lon'] = np.array([det_storms[tt]['lon'][un]])
            storm_tmp['lat'] = np.array([det_storms[tt]['lat'][un]])
            storm_tmp['amp'] = np.array([det_storms[tt]['amp'][un]])
            storm_tmp['type'] = det_storms[tt]['type'][un]
            storm_tmp['year'] = year[tt]
            storm_tmp['month'] = month[tt]
            storm_tmp['day'] = day[tt]
            storm_tmp['hour'] = hour[tt]
            storm_tmp['exist_at_start'] = False
            storm_tmp['terminated'] = False
            storms.append(storm_tmp)

    return storms


def strip_storms(tracked_storms, dt, d_tot_min=1000., d_ratio=0.6, dur_min=72):
    '''
    Following Klotzbach et al. (MWR, 2016) strip out storms with:
     1. A duration of less than dur_min (in hours). dt provides the
        time step of the track data (in hours).
     2. A total track length <= d_tot_min (short tracks)
     3. A start-to-end straight-line distance that is less than d_ratio
        times the total track length (meandering tracks).
    Use d_tot_min = 0, d_ratio = 0 and/or dur_min = 0 to avoid stripping out
    storms due to these criteria. It is recommended to use dur_min >= 6 or 12
    hours in order to remove a significant number of "storms" that appear due
    to high-frequency synoptic variability in the data.
    '''

    stripped_storms = []

    for ed in range(len(tracked_storms)):

        # 1. Remove storms which last less than dur_min hours
        if len(tracked_storms[ed]['lon']) <= dur_min/dt:
            continue

        # 2. Calculate total track length
        d_tot = 0
        for k in range(len(tracked_storms[ed]['lon'])-1):
            d_tot += latlon2km(tracked_storms[ed]['lon'][k], tracked_storms[ed]['lat'][k], tracked_storms[ed]['lon'][k+1], tracked_storms[ed]['lat'][k+1])

        # 3. Calcualate start-to-end straight-line track distance
        d_str = latlon2km(tracked_storms[ed]['lon'][0], tracked_storms[ed]['lat'][0], tracked_storms[ed]['lon'][-1], tracked_storms[ed]['lat'][-1])

        # Keep storms that satisfy the conditions 2 & 3
        if (d_tot >= d_tot_min) * ((d_str / d_tot) >= d_ratio):
            stripped_storms.append(tracked_storms[ed])

    return stripped_storms


def timevector(date_start, date_end):
    '''
    Generated daily time vector, along with year, month, day, day-of-year,
    and full date information, given start and and date. Format is a 3-element
    list so that a start date of 3 May 2005 is specified date_start = [2005,5,3]
    Note that day-of year (doy) is [0 to 59, 61 to 366] for non-leap years and [0 to 366]
    for leap years.
    returns: t, dates, T, year, month, day, doy
    '''
    # Time vector
    t = np.arange(date(date_start[0],date_start[1],date_start[2]).toordinal(),date(date_end[0],date_end[1],date_end[2]).toordinal()+1)
    T = len(t)
    # Date list
    dates = [date.fromordinal(tt.astype(int)) for tt in t]
    # Vectors for year, month, day-of-month
    year = np.zeros((T))
    month = np.zeros((T))
    day = np.zeros((T))
    for tt in range(T):
        year[tt] = date.fromordinal(t[tt]).year
        month[tt] = date.fromordinal(t[tt]).month
        day[tt] = date.fromordinal(t[tt]).day
    year = year.astype(int)
    month = month.astype(int)
    day = day.astype(int)
    # Leap-year baseline for defining day-of-year values
    year_leapYear = 2012 # This year was a leap-year and therefore doy in range of 1 to 366
    t_leapYear = np.arange(date(year_leapYear, 1, 1).toordinal(),date(year_leapYear, 12, 31).toordinal()+1)
    dates_leapYear = [date.fromordinal(tt.astype(int)) for tt in t_leapYear]
    month_leapYear = np.zeros((len(t_leapYear)))
    day_leapYear = np.zeros((len(t_leapYear)))
    doy_leapYear = np.zeros((len(t_leapYear)))
    for tt in range(len(t_leapYear)):
        month_leapYear[tt] = date.fromordinal(t_leapYear[tt]).month
        day_leapYear[tt] = date.fromordinal(t_leapYear[tt]).day
        doy_leapYear[tt] = t_leapYear[tt] - date(date.fromordinal(t_leapYear[tt]).year,1,1).toordinal() + 1
    # Calculate day-of-year values
    doy = np.zeros((T))
    for tt in range(T):
        doy[tt] = doy_leapYear[(month_leapYear == month[tt]) * (day_leapYear == day[tt])]
    doy = doy.astype(int)

    return t, dates, T, year, month, day, doy
