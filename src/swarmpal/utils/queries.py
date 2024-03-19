from __future__ import annotations

from viresclient import SwarmRequest


def last_available_time(collection):
    """UTC of last available data for a collection, e.g. SW_FAST_MAGA_LR_1B

    Parameters
    ----------
    collection : str

    Returns
    -------
    datetime
    """
    request = SwarmRequest()
    availability = request.available_times(collection)
    return availability["endtime"].iloc[-1].to_pydatetime().replace(tzinfo=None)
