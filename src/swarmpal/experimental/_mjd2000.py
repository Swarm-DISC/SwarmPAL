"""
Copied from https://github.com/ancklo/ChaosMagPy/blob/afd8883c992faefed373dc0cbf63186daa8c01b4/chaosmagpy/data_utils.py#L384
"""

import datetime as dt
import numpy as np


def mjd2000(year, month=1, day=1, hour=0, minute=0, second=0, microsecond=0,
            nanosecond=0):
    """
    Computes the modified Julian date as floating point number (epoch 2000).

    It assigns 0 to 0h00 January 1, 2000. Leap seconds are not accounted for.

    Parameters
    ----------
    time : :class:`datetime.datetime`, ndarray, shape (...)
        Datetime class instance, `OR ...`
    year : int, ndarray, shape (...)
    month : int, ndarray, shape (...), optional
        Month of the year `[1, 12]` (defaults to 1).
    day : int, ndarray, shape (...), optional
        Day of the corresponding month (defaults to 1).
    hour : int , ndarray, shape (...), optional
        Hour of the day (default is 0).
    minute : int, ndarray, shape (...), optional
        Minutes (default is 0).
    second : int, ndarray, shape (...), optional
        Seconds (default is 0).
    microsecond : int, ndarray, shape (...), optional
        Microseconds (default is 0).
    nanosecond : int, ndarray, shape (...), optional
        Nanoseconds (default is 0).

    Returns
    -------
    time : ndarray, shape (...)
        Modified Julian date (units of days).

    Examples
    --------
    >>> a = np.array([datetime.datetime(2000, 1, 1), \
datetime.datetime(2002, 3, 4)])
    >>> mjd2000(a)
        array([  0., 793.])

    >>> mjd2000(2003, 5, 3, 13, 52, 15)  # May 3, 2003, 13:52:15 (hh:mm:ss)
        1218.5779513888888

    >>> mjd2000(np.arange(2000, 2005))  # January 1 in each year
        array([   0.,  366.,  731., 1096., 1461.])

    >>> mjd2000(np.arange(2000, 2005), 2, 1)  # February 1 in each year
        array([  31.,  397.,  762., 1127., 1492.])

    >>> mjd2000(np.arange(2000, 2005), 2, np.arange(1, 6))
        array([  31.,  398.,  764., 1130., 1496.])

    """

    year = np.asarray(year)

    if (np.issubdtype(year.dtype, np.dtype(dt.datetime).type) or
            np.issubdtype(year.dtype, np.datetime64)):
        datetime = year.astype('datetime64[ns]')

    else:
        # build iso datetime string with str_ (supported in NumPy >= 2.0)
        year = np.asarray(year, dtype=np.str_)
        month = np.char.zfill(np.asarray(month, dtype=np.str_), 2)
        day = np.char.zfill(np.asarray(day, dtype=np.str_), 2)

        year_month = np.char.add(np.char.add(year, '-'), month)
        datetime = np.char.add(np.char.add(year_month, '-'), day)

        datetime = datetime.astype('datetime64[ns]')

        # not use iadd here because it doesn't broadcast arrays
        datetime = (datetime + np.asarray(hour, dtype='timedelta64[h]')
                    + np.asarray(minute, dtype='timedelta64[m]')
                    + np.asarray(second, dtype='timedelta64[s]')
                    + np.asarray(microsecond, dtype='timedelta64[us]')
                    + np.asarray(nanosecond, dtype='timedelta64[ns]'))

    nanoseconds = datetime - np.datetime64('2000-01-01', 'ns')

    return nanoseconds / np.timedelta64(1, 'D')  # fraction of days
