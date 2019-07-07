# _*_ coding: utf-8 _*_

# Copyright (c) 2019 NMC Developers.
# Distributed under the terms of the GPL V3 License.

"""
Collections of utilities functions.
"""


def lon2txt(lon, fmt='%g'):
    """
    Format the longitude number with degrees.

    :param lon: longitude
    :param fmt:
    :return:

    :Examples:
    >>> lon2txt(135)
     '135\N{DEGREE SIGN}E'
    >>> lon2txt(-30)
     '30\N{DEGREE SIGN}W'
    >>> lon2txt(250)
     '110\N{DEGREE SIGN}W'
    """
    lon = (lon + 360) % 360
    if lon > 180:
        lonlabstr = u'%s\N{DEGREE SIGN}W' % fmt
        lonlab = lonlabstr % abs(lon - 360)
    elif lon < 180 and lon != 0:
        lonlabstr = u'%s\N{DEGREE SIGN}E' % fmt
        lonlab = lonlabstr % lon
    else:
        lonlabstr = u'%s\N{DEGREE SIGN}' % fmt
        lonlab = lonlabstr % lon
    return lonlab


def lat2txt(lat, fmt='%g'):
    """
    Format the latitude number with degrees.
    :param lat:
    :param fmt:
    :return:

    :Examples:
    >>> lat2txt(60)
     '60\N{DEGREE SIGN}N'
    >>> lat2txt(-30)
     '30\N{DEGREE SIGN}S'
    """
    if lat < 0:
        latlabstr = u'%s\N{DEGREE SIGN}S' % fmt
        latlab = latlabstr % abs(lat)
    elif lat > 0:
        latlabstr = u'%s\N{DEGREE SIGN}N' % fmt
        latlab = latlabstr % lat
    else:
        latlabstr = u'%s\N{DEGREE SIGN}' % fmt
        latlab = latlabstr % lat
    return latlab
