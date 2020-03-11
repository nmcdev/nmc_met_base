# _*_ coding: utf-8 _*_

# Copyright (c) 2019 NMC Developers.
# Distributed under the terms of the GPL V3 License.

"""
Cyclone identification and track methods.
"""

import numpy as np
from nmc_met_base.geographical import haversine_np
from nmc_met_base.math import extreme_2d


def _elim_mult_centers(in_press, in_lon, in_lat, search_rad=800e3, type=-1):
    """
    ;   Given a vector of pressures, and corresponding vectors of lon. and lat.
    ;   where those pressures are at, looks to see if any two points are "too"
    ;   close to each other.  If they are, then the one with the lower (or
    ;   higher, as set by the Type keyword) pressure is retained.  The 1-D
    ;   vector returned is of the locations (in terms of subscripting of the
    ;   original pressure vector) that have been retained.
    ;
    ;   This function is typically used for eliminating multiple high or low
    ;   centers that has been identified by an automated pressure center
    ;   finding algorithm.

    :param in_press: Pressure (in hPa) at locations defined by in_lon and
                     in_lat.  Floating or double array of any dimension.
                     Unchanged by procedure.
    :param in_lon: Longitude of points given by in_press (in decimal deg).
                   Floating or double array.  Same dimensions as in_press.
    :param in_lat: Latitude of points given by in_press (in decimal deg).
                   Floating or double array.  Same dimensions as in_press.
    :param search_rad: Radius defining the region from a point the procedure
                       searches to try and determine whether a given location
                       is too close to other locations.  In meters.  Not
                       changed by function.  This can either be a scalar
                       (which is applied to all locations) or a vector of the
                       same size as in_press that gives Search_Rad to use
                       for each location.  Default is 800e3 meters.
    :param type: Required.  If set to 1, then the function retains the
                 higher of the pressures; if set to -1, then the function
                 retains the lower of the pressures.
    :return: Vector of the locations of retained locations, as
             described above.  Created.  1-D integer vector of
             array indices, in terms of the input array in_press.
             If none of the pressures are "too close" to each other,
             out_loc will end up being just a vector of the indices
             of all the elements in in_press.
    """

    # protect input
    press = in_press
    lon = in_lon
    lat = in_lat
    npress = press.size

    '''
    ; --------------------- Find Multiple Center Situations -------------------
    ;
    ; Method:  All permutations of the values of press are tested pairwise
    ; against to see each other to see if they are less than Search_Rad apart.
    ; If so, it is assumed that they are not actually separate systems, and
    ; the value with the lowest (highest) value is retained as describing the
    ; true low (high) center.
    ;
    ; NB:  If a case exists where the min. (or max.) of the points that are
    ; within Search_Rad of each other applies to more than one point, it is
    ; assumed that both are centers, and a warning message is printed out.
    ; This should be an extremely rare situation, since press is floating pt.
    '''
    out_loc = np.array([], dtype=np.int64)
    for i in range(npress):
        dist_from_i = haversine_np(
            np.full(npress, lon[i]), np.full(npress, lat[i]), lon, lat)
        same_loc = np.flatnonzero(dist_from_i <= search_rad)

        if same_loc.size == 1:
            out_loc = np.append(out_loc, same_loc)

        if same_loc.size > 1:
            same_press = press[same_loc]
            if type > 0:
                keep_pts = np.argmax(same_press)
            else:
                keep_pts = np.argmin(same_press)
            out_loc = np.append(out_loc, same_loc[keep_pts])

    # ---------------------------- Clean-Up and Output ------------------------
    if out_loc.size == 0:
        out_loc = np.arange(npress)
    else:
        out_loc = np.unique(out_loc)

    return out_loc


def loc(in_press, in_lon, in_lat, edge_distance=800e3,
        lr_periodic=False, tb_periodic=False,
        search_rad_max=1200e3, search_rad_min=400e3,
        search_rad_ndiv=3, slp_diff_test=2, limit=None,
        ref_point=None, relax=1.0):
    """
    ;   Given a lat.-lon. grid of sea level pressure, fctn. finds where the
    ;   centers of cyclones are (using a form of the Serreze (1995) and Serreze
    ;   et al. (1997) algorithms) and returns a vector of the locations of
    ;   centers, in 1-D array index form.  This function supports "pseudo-
    ;   arbitrary" spacing:  For the purposes of calculating local maximum, it
    ;   is assumed that the grid is a 2-D grid where each internal point is
    ;   surrounded by 8 points).  The boundaries of the 2-D array are also
    ;   assumed to be the boundaries of the domain.  However, no other
    ;   assumptions, including in terms of grid size and spacing, are made.
    ;
    ;   If either your top/bottom or left/right boundaries are periodic, see
    ;   keyword list discussion of Lr_Periodic and Tb_Periodic below.  Note
    ;   although these keywords are included, I have not tested whether
    ;   specifying those keywords will properly detect cyclones at periodic
    ;   boundaries; I have only tested whether the specification of those
    ;   keywords will turn on or off the edge effect filter.
    http://www.johnny-lin.com/idl_code/cyclone_loc.pro

    - References:
    * Serreze, M. C. (1995), Climatological aspects of cyclone development
      and decay in the Arctic, Atmos.-Oc., v. 33, pp. 1-23;
    * Serreze, M. C., F. Carse, R. G. Barry, and J. C. Rogers (1997),
      Icelandic low cyclone activity:  Climatological features, linkages
      with the NAO, and relationships with recent changes in the Northern
      Hemisphere circulation, J. Clim., v. 10, pp. 453-464.

    Notices:
    1 参数的选择对于最后的结果非常重要, 最好将典型的气旋显示出来,
      主观测量要识别气旋的大小, 获得参数的值.
    2 search_rad_min和slp_diff_test的设置经验上更为重要一些.
    3 典型气旋中心常有多个低极值点, 因此search_rad_min设置太小会造成
      同一个气旋多个气旋被识别出来, search_rad_min最好能够覆盖
      气旋的中心部位.
    4 slp_diff_test要根据search_rad_min的距离来设置, 不能设置太大,
      会造成很难满足条件而无法识别出气旋, 也不能设置太小而把太多的
      弱气旋包含进来, 一般考虑0.25/100km.
    5 search_rad_max最好包含气旋的最外围, 但其主要作用是保证有4以上的点
      高于中心气压, 这个一般很好满足, 因此不太重要.
    6 search_rad_ndiv就用默认的3就行, 一般第一个圆环就能满足条件.

    :param in_press: Sea level pressure (in hPa) at grid defined by in_lon and
                     in_lat. 2-D floating or double array.
    :param in_lon: Longitude of grid given by in_press (in decimal deg),
                   1D array.
    :param in_lat: Latitude of grid given by in_press (in decimal deg),
                   1D array.
    :param edge_distance:  Distance defining how from the edge is the "good"
                           domain to be considered; if you're within this
                           distance of the edge (and you're boundary is
                           non-periodic), it's assumed that cyclone centers
                           cannot be detected there.  In meters.  Not changed
                           by function.  Scalar.  Default is 800 kilometers.
    :param lr_periodic:  If LR_PERIODIC is true, the left and right (i.e. col.
                         at rows 0 and NX_P-1) bound. are assumed periodic,
                         and the edge effect for IDing cyclones (i.e. that
                         cyclones found near the edge are not valid) is assumed
                         not to apply.
    :param tb_periodic:  If TB_PERIODIC is true, the top and bottom bound.
                         (rows at col. 0 and NY_P-1) are assumed periodic.  If
                         neither are true (default), none of the bound. are
                         assumed periodic.
    :param search_rad_max: Max. radius defining the region from a point the
                           procedure searches to try and determine whether a
                           given location is a low pressure center.  In meters.
                           Not changed by function. This can either be a scalar
                           (which is applied to all locations) or a vector of
                           the same size as in_press of Search_Rad_Max to use
                           for each location.  Default is 1200e3 meters.
    :param search_rad_min: Min. radius defining the region from a point the
                           procedure searches to determine whether a given
                           location is a low pressure center.  In meters.  Not
                           changed by function.  This can either be a scalar
                           (which is applied to all locations) or a vector of
                           the same size as in_press that gives Search_Rad_Min
                           to use for each location.  Default is 400e3 meters.
                           This value is also used to test for multiple lows
                           (see commenting below).
    :param search_rad_ndiv: Integer number of shells between Search_Rad_Min and
                            Search_Rad_Max to search.  Scalar.  Default is 3.
    :param slp_diff_test: A low pressure center is identified if it is entirely
                          surrounded by grid points in the region between
                          Search_Rad_Min and Search_Rad_Max that are all higher
                          in SLP than the point in question by a min. of
                          Slp_Diff_Test.  In hPa. Not changed by function. This
                          can either be a scalar (which is applied to all
                          locations) or a vector of the same size as in_press
                          of slp_diff_test to use for each location.
                          Default is 2 hPa.
    :param limit: give a region limit where cyclones can be identified,
                  format is [lonmin, lonmax, latmin, latmax].
                  if None, do not think limit region.
    :param ref_point: if is not None, will return the nearest cyclone to the
                      reference point.
    :param relax: value 0~1.0, the proportion of shell grid points which meet
                  the pressure slp_diff_test.

    :return: [ncyclones, 3] array, each cyclone
             [cent_lon, cent_lat,cent_pressure]

    """

    # protect input
    press = in_press.ravel()
    lons, lats = np.meshgrid(in_lon, in_lat)
    npress = press.size

    #
    # Start cycling through each point in Entire Domain
    tmp_loc = []
    for i in range(npress):
        # check limit region
        if limit is not None:
            if (lons.ravel()[i] < limit[0]) or (lons.ravel()[i] > limit[1]) \
                    or (lats.ravel()[i] < limit[2]) or \
                    (lats.ravel()[i] > limit[3]):
                continue

        '''
        ; ------ What Array Indices Surround Each Index for a Shell of Points -
        ;
        ; shell_loc_for_i is a vector of the subscripts of the points that
        ; are within the region defined by search_rad_min and search_rad_top of
        ; the element i, and are not i itself.
        ;
        ; For each point in the spatial domain, we search through a number of
        ; shells (where search_rad_top expands outwards by search_rad_ndiv
        ; steps until it reaches search_rad_max).  This enables more
        ; flexibility in  finding centers of various sizes.
        '''

        # distance of each point from i
        dist_from_i = haversine_np(
            np.full(npress, lons.ravel()[i]), np.full(npress, lats.ravel()[i]),
            lons.ravel(), lats.ravel())

        # make array of the lower limit of of the search shell
        incr = (search_rad_max - search_rad_min) / search_rad_ndiv
        search_rad_top = (np.arange(search_rad_ndiv) + 1.0) * incr + \
            search_rad_min

        # Cycle through each search_rad division
        for ndiv in range(search_rad_ndiv):
            shell_loc_for_i = np.flatnonzero(
                (dist_from_i <= search_rad_top[ndiv]) &
                (dist_from_i >= search_rad_min))
            npts_shell = shell_loc_for_i.size

            if npts_shell == 0:
                print("*** warning--domain may be too spread out ***")

            '''
            ; --------------- Find Locations That Pass the Low Pressure Test --
            ;
            ; Method:  For each location, check that the pressure of all the
            ; points in the shell around i, defined by search_rad_top and
            ; search_rad_min, is slp_diff_test higher.  If so, and the shell
            ; of points around that location is >= 4 (which is a test to help
            ; make sure the location isn't being examined on the basis of just
            ; a few points), then that location is labeled as passing the low
            ; pressure test.
            ;
            ; Note that since the shell is based upon distance which is based
            ; on lat/lon, this low pressure test automatically accommodates for
            ; periodic bound., if the bounds are periodic.  For non-periodic
            ; bounds, some edge points may pass this test, and thus must be
            ; removed later on in the edge effects removal section.
            '''
            if npts_shell > 0:
                slp_diff = press[shell_loc_for_i] - press[i]
                tmp = np.flatnonzero(slp_diff >= slp_diff_test)
                if (tmp.size >= npts_shell*relax) and (npts_shell >= 4):
                    tmp_loc.append(i)
                    break    # pass the low pressure test

    '''
    ; ----------------- Identify Low Pressure Centers Candidates --------------
    ;
    ; Method:  From the locations that pass the SLP difference test, we find
    ; which ones could be low pressure centers by finding the locations that
    ; are local minimums in SLP.  Note low_loc values are in units of indices
    ; of the orig. pressure array.
    '''
    if len(tmp_loc) == 0:
        return None

    tmp_loc = np.array(tmp_loc)
    test_slp = np.full(in_press.shape, 100000.0)
    test_slp.ravel()[tmp_loc] = press.ravel()[tmp_loc]

    # 会去掉几个相邻的低压中心候选点，找一个最低气压的低压中心.
    low_loc = extreme_2d(test_slp, -1, edge=True)

    '''
    ; ----- Test For Multiple Systems In a Region Defined By Search_Rad_Min --
    ;
    ; Method:  If two low centers identified in low_loc are less than
    ; Search_Rad_Min apart, it is assumed that they are not actually
    ; separate systems, and the value with the lowest SLP value is
    ; retained as describing the true low center.
    '''
    if low_loc is not None:
        test_slp_ll = test_slp.ravel()[low_loc]
        lon_ll = lons.ravel()[low_loc]
        lat_ll = lats.ravel()[low_loc]
        emc_loc = _elim_mult_centers(
            test_slp_ll, lon_ll, lat_ll, type=-1, search_rad=search_rad_min)
        out_loc = low_loc[emc_loc]
    else:
        return None

    '''
    ; --------------------------- Eliminate Edge Points -----------------------
    ;
    ; Method:  Eliminate all out_loc candidate points that are a distance
    ; Edge_Distance away from the edge, for the boundaries that are non-
    ; periodic.
    '''
    # Flag to elim. edge:  default is on (=1)
    ielim_flag = True

    if not lr_periodic and not tb_periodic:
        edge_lon = np.concatenate(
            (lons[0, :], lons[-1, :], lons[:, 0], lons[:, -1]))
        edge_lat = np.concatenate(
            (lats[0, :], lats[-1, :], lats[:, 0], lats[:, -1]))
    elif lr_periodic and not tb_periodic:
        edge_lon = np.concatenate((lons[:, 0], lons[:, -1]))
        edge_lat = np.concatenate((lats[:, 0], lats[:, -1]))
    elif not lr_periodic and tb_periodic:
        edge_lon = np.concatenate((lons[0, :], lons[-1, :]))
        edge_lat = np.concatenate((lats[0, :], lats[-1, :]))
    elif lr_periodic and tb_periodic:
        # set flag to elim. edge to off
        ielim_flag = False
    else:
        print('error--bad periodic keywords')

    # Case elim. at least some edges
    if ielim_flag:
        for i, iloc in np.ndenumerate(out_loc):
            dist_from_ol_i = haversine_np(
                np.full(edge_lon.size, lons.ravel()[iloc]),
                np.full(edge_lat.size, lats.ravel()[iloc]),
                edge_lon, edge_lat)

            tmp = np.flatnonzero(dist_from_ol_i <= edge_distance)
            if tmp.size > 0:
                out_loc[i] = -1

        # keep only those points not near edge:
        good_pts = np.flatnonzero(out_loc >= 0)
        if good_pts.size > 0:
            out_loc = out_loc[good_pts]
        else:
            return None

    # clean up and sort
    cent_lon = lons.ravel()[out_loc]
    cent_lat = lats.ravel()[out_loc]
    cent_press = press[out_loc]
    sort_idx = np.argsort(cent_press)
    cent_press = cent_press[sort_idx]
    cent_lon = cent_lon[sort_idx]
    cent_lat = cent_lat[sort_idx]
    if ref_point is None:
        return np.stack((cent_lon, cent_lat, cent_press), axis=1)
    else:
        dist_from_refer = haversine_np(
            np.full(cent_press.size, ref_point[0]),
            np.full(cent_press.size, ref_point[1]), cent_lon, cent_lat)
        idx = np.argmin(dist_from_refer)
        return np.array(
            [cent_lon[idx], cent_lat[idx], cent_press[idx]]).reshape([1, 3])
