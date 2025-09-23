"""Utilities for builtin distributions and fitters."""

import datetime as dt

import numpy as np


def convert_numpy_datetime(time_obj: np.datetime64) -> dt.datetime:
    """Convert numpy datetime to python stdlib datetime.

    Parameters
    ----------
    time_obj:
        datetime to be converted.

    Returns
    -------
    datetime.datetime:
        Converted datetime.
    """
    unix_epoch = np.datetime64(0, 's')
    one_second = np.timedelta64(1, 's')
    seconds_since_epoch = (time_obj - unix_epoch) / one_second
    return dt.datetime.fromtimestamp(float(seconds_since_epoch))

