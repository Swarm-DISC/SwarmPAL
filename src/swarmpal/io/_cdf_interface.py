from __future__ import annotations

import logging

import pycdfpp
from viresclient._data_handling import FileReader
from xarray import Dataset

logger = logging.getLogger(__name__)


def cdf_to_xarray_viresclient(
    cdf_file,
):
    """Using viresclient FileReader"""
    with open(cdf_file, "rb") as f:
        with FileReader(f) as fr:
            return fr.as_xarray_dataset()


def cdf_to_xarray(cdf_file):
    """Using pycdfpp"""
    cdf = pycdfpp.load(str(cdf_file))
    timevars = ("Timestamp",)  # NB hardcoded for now, but should be made flexible
    ds = Dataset()
    # Add all the variables
    for varname, data in cdf.items():
        timevar = timevars[0]
        num_dims = len(data.shape)
        cdf_type = data.type
        dim_names = [timevar] + [f"{varname}_dim_{i}" for i in range(1, num_dims)]
        if str(cdf_type).endswith("CDF_EPOCH"):
            ds[varname] = dim_names, pycdfpp.to_datetime64(data)
        else:
            ds[varname] = dim_names, data
        attrs = {}
        for _, attr in dict(data.attributes).items():
            attrs[attr.name] = attr.value
        ds[varname].attrs = attrs
    # Add global attributes
    for attr_name, attr_value in cdf.attributes.items():
        _attr_value = attr_value[0] if len(attr_value) == 1 else list(attr_value)
        ds.attrs[attr_name] = _attr_value
    return ds


def xarray_to_cdf(ds, file_name):
    cdf = pycdfpp.CDF()
    # Global attributes
    for attr_name, attr_value in ds.attrs.items():
        _attr_value = [attr_value] if not isinstance(attr_value, list) else attr_value
        try:
            cdf.add_attribute(attr_name, _attr_value)
        except Exception as e:
            logger.warning(f"Error adding attribute {attr_name}: {e}")
    # # Coordinates
    # for var_name, var_data in ds.coords.items():
    #     cdf.add_variable(var_name, var_data.values, compression=pycdfpp.CompressionType.gzip_compression)
    #     for attr_name, attr_value in var_data.attrs.items():
    #         cdf[var_name].add_attribute(attr_name, attr_value)
    # Only add Timestamp coordinate (typically the only coordinate in the dataset)
    # Fix Timestamp to CDF_EPOCH to match source data
    cdf.add_variable(
        "Timestamp",
        ds["Timestamp"].values,
        pycdfpp.DataType.CDF_EPOCH,
        compression=pycdfpp.CompressionType.gzip_compression,
    )
    # Data variables
    for var_name, var_data in ds.data_vars.items():
        _var_data = var_data.astype("str") if var_name == "Spacecraft" else var_data
        try:
            cdf.add_variable(
                var_name,
                _var_data.values,
                compression=pycdfpp.CompressionType.gzip_compression,
            )
            for attr_name, attr_value in _var_data.attrs.items():
                cdf[var_name].add_attribute(attr_name, attr_value)
        except Exception as e:
            logger.warning(f"Error adding variable {var_name}: {e}")
    pycdfpp.save(cdf, file_name)
