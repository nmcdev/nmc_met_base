# _*_ coding: utf-8 _*_

# Copyright (c) 2019 NMC Developers.
# Distributed under the terms of the GPL V3 License.

"""
  Date and time manipulate functions.
"""

import calendar
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


def get_same_date_range(date_center,
                        years=[1991+i for i in range(30)],
                        period=15, freq="1D"):
    """
    Generate date series for the same period of each year.

    Args:
        date_center (list): the central date, [month, day]
        years (list): year list, defaults to  [1991+i for i in range(30)].
        period (int, optional): date range length. Defaults to 15.
        freq (str, optional): date frequency. Defaults to "1D".

    Returns:
        the same period date series, [date_center-int(period/2), date_center+int(period/2)]
    """

    times = pd.to_datetime([])

    for year in years:
        start = dt.datetime(year, date_center[0], date_center[1]) - dt.timedelta(days=int(period/2))
        times = times.append(pd.date_range(start=start, periods=period, freq=freq))

    return times


def get_same_date(mon_day, years=[1991+i for i in range(30)]):
    """
    Generate the same date for each year. If mon_day = [2,29]
    in not leap year, just set empty item.

    Args:
        mon_day (list): the [month, day] list
        years (list, optional): year list, defaults to  [1991+i for i in range(30)].
    """

    dates = []
    for year in years:
        if mon_day == [2, 29]:
            if not calendar.isleap(year):
                continue
        dates.append(dt.datetime(year, mon_day[0], mon_day[1]))

    return dates


def extract_same_date(data, middle_date=None, period=7, var_time="Datetime"):
    """
    该程序用于从多年的时间序列数据中获得历史同期的数据.

    Args:
        data (pandas dataframe): pandas数据表格, 其中一列为时间序列的时间戳.
        middle_date (date object, optional): 中间时间日期. Defaults to None.
        period (int, optional): 时间窗口半径. Defaults to 7, 表示在middle_date前后7天之内.
    """
    
    if middle_date is None:
        middle_date = dt.date.today()
    
    same_dates = pd.date_range(start=middle_date-pd.Timedelta(period, unit='day'),
                               periods=period*2+1)
    same_dates = same_dates.strftime('%m-%d').to_list()
    data['Date'] = data[var_time].dt.strftime("%m-%d")
    data = data.loc[data['Date'].isin(same_dates), :]
    data.drop('Date', axis=1, inplace=True)
    
    return data 


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

