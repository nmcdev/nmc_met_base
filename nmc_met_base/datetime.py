# _*_ coding: utf-8 _*_

# Copyright (c) 2019 NMC Developers.
# Distributed under the terms of the GPL V3 License.

"""
  Date and time manipulate functions.
"""

import datetime as dt
import pandas as pd
from dateutil.relativedelta import relativedelta


__months__ = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
              'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']


def datetime_range(start, end, delta):
    """
    Generate a list of datetimes between an interval.
    https://stackoverflow.com/questions/10688006/generate-a-list-of-datetimes-between-an-interval

    :param start: start date time.
    :param end: end date time.
    :param delta: time delta.
    :return: datetime list.

    :example
    >>> start = dt.datetime(2015,1,1)
    >>> end = dt.datetime(2015,1,31)
    >>> for dt in datetime_range(start, end, {'days': 2, 'hours':12}):
    ...     print(dt)
    """
    current = start
    if not isinstance(delta, dt.timedelta):
        delta = dt.timedelta(**delta)
    result = []
    while current <= end:
        result.append(current)
        current += delta
    return result


def d2s(d, fmt='%HZ%d%b%Y'):
    """
    Convert datetime to grads time string.
    https://bitbucket.org/tmiyachi/pymet/src/8df8e3ff2f899d625939448d7e96755dfa535357/pymet/tools.py

    :param d: datetime object
    :param fmt: datetime format
    :return: string

    :Examples:
     >>> d2s(datetime(2009,10,13,12))
     '12Z13OCT2009'
     >>> d2s(datetime(2009,10,13,12), fmt='%H:%MZ:%d%b%Y')
     '12:00Z13OCT2009'
    """

    fmt = fmt.replace('%b', __months__[d.month - 1])
    if d.year < 1900:
        fmt = fmt.replace('%Y', '{:04d}'.format(d.year))
        d = d + relativedelta(year=1900)
    return d.strftime(fmt)


def s2d(datestring):
    """
    Convert GRADS time string to datetime object.
    https://bitbucket.org/tmiyachi/pymet/src/8df8e3ff2f899d625939448d7e96755dfa535357/pymet/tools.py

    :param datestring: GRADS time string
    :return: datetime object

    :Examples:
     >>> s2d('12:30Z13OCT2009')
     datetime(2009, 10, 13, 12, 30)
     >>> s2d('12Z13OCT2009')
     datetime(2009, 10, 13, 12)
    """

    time, date = datestring.upper().split('Z')
    if time.count(':') > 0:
        hh, mm = time.split(':')
    else:
        hh = time
        mm = 0
    dd = date[:-7]
    mmm = __months__.index(date[-7:-4])+1
    yyyy = date[-4:]
    return dt.datetime(int(yyyy), int(mmm), int(dd), int(hh), int(mm))


def np64toDate(np64):
    """
    Converts a Numpy datetime64 to a Python datetime.

    :param np64: Numpy datetime64 value.
    :return:
    """
    return pd.to_datetime(str(np64)).replace(tzinfo=None).to_pydatetime()
