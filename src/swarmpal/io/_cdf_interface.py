from __future__ import annotations

from viresclient._data_handling import FileReader


def cdf_to_xarray(cdf_file):
    with open(cdf_file, "rb") as f:
        with FileReader(f) as fr:
            return fr.as_xarray_dataset()
