from __future__ import annotations

import re

import numpy as np
import pandas as pd
from xarray import DataArray, DataTree

from swarmpal.io import PalProcess
from swarmpal.utils.exceptions import PalError

# Dataset name patterns, grouped by product family. A dataset's group
# determines which entry of `DEFAULT_REJECT_BITS` applies to it, since the
# same flag_name can need different treatment in different products (e.g. a
# future non-MAGX_LR_1B product might filter `Flags_F` differently).
DATASET_GROUPS = {
    "MAGX_LR_1B": re.compile(r"^SW_(OPER|FAST)_MAG[ABC]_LR_1B$"),
}

# Default "bad" bits per flag field, keyed first by dataset group (see
# `DATASET_GROUPS`) and curated from flag_definitions.md. Bits that are
# purely informational (mode/source indicators, not defects) are excluded
# from the defaults; callers can override via `reject_bits`.
DEFAULT_REJECT_BITS = {
    "MAGX_LR_1B": {
        "Flags_F": [1, 2, 3, 4, 5, 6],
        "Flags_B": [1, 2, 3, 4],
        "Flags_Platform": [1, 2, 3, 4, 6, 7],
        # Flags_q is not a clean bitfield (bits 4-5 select a category, bits
        # 0-2 are an enumerated sub-code within that category), so it has no
        # safe default - callers must pass explicit `reject_bits` to get any
        # masking from it.
        "Flags_q": [],
    },
}


def _dataset_group(dataset: str) -> str | None:
    """Identify which `DEFAULT_REJECT_BITS` group a dataset name belongs to"""
    for group, pattern in DATASET_GROUPS.items():
        if pattern.match(dataset):
            return group
    return None


class FlagClean(PalProcess):
    """Mask a variable to NaN using bitwise checks on one or more flag fields

    Scoped to the dataset groups in ``DATASET_GROUPS`` (currently just
    ``MAGX_LR_1B``, i.e. SW_{OPER,FAST}_MAG{A,B,C}_LR_1B). Supported
    flag_names and their default reject bits are looked up per group via
    ``DEFAULT_REJECT_BITS``, since the same flag_name can need different
    treatment in a different product. See
    ``swarmpal/experimental/flag_definitions.md`` for the meaning of each bit
    in ``Flags_F``, ``Flags_B`` and ``Flags_Platform``. ``Flags_q`` is also
    accepted, but since it is not a clean bitfield it has no default reject
    bits - pass `reject_bits` explicitly to mask anything with it.
    """

    @property
    def process_name(self) -> str:
        return "FlagClean"

    def set_config(
        self,
        dataset: str = "SW_OPER_MAGA_LR_1B",
        variable: str = "",
        flags: list | None = None,
        output_dataset: str = "",
    ) -> None:
        """Set the process configuration

        Parameters
        ----------
        dataset : str
            Selects this dataset from the datatree. Must match one of the
            patterns in `DATASET_GROUPS` (currently just
            SW_{OPER,FAST}_MAG{A,B,C}_LR_1B).
        variable : str
            Name of the variable to clean (e.g. "F", "B_NEC")
        flags : list of dict
            Each entry is ``{"flag_name": str, "reject_bits": list[int]}``.
            ``flag_name`` must be a key in `DEFAULT_REJECT_BITS` for the
            group `dataset` belongs to (e.g. for MAGX_LR_1B: "Flags_F",
            "Flags_B", "Flags_Platform", "Flags_q"). ``reject_bits`` is
            optional; when omitted, the default bits for that group and
            flag_name (`DEFAULT_REJECT_BITS`) are used - for "Flags_q" this
            default is empty, so it rejects nothing unless `reject_bits` is
            given explicitly. Bad masks from all entries are OR-combined.
        output_dataset : str
            Sets the name of the dataset in the data tree that this process
            writes results to, by default the same as `dataset`
        """
        super().set_config(
            dataset=dataset,
            variable=variable,
            flags=flags or [],
            output_dataset=output_dataset or dataset,
        )

    def _call(self, datatree: DataTree) -> DataTree:
        dataset = self.config["dataset"]
        variable = self.config["variable"]
        self._validate(datatree)
        group_defaults = DEFAULT_REJECT_BITS[_dataset_group(dataset)]
        subtree = datatree[dataset]
        ds = subtree.ds
        bad = np.zeros(ds[variable].shape[0], dtype=bool)
        for entry in self.config["flags"]:
            flag_name = entry["flag_name"]
            reject_bits = entry.get("reject_bits") or group_defaults[flag_name]
            mask = sum(1 << b for b in reject_bits)
            bad |= (ds[flag_name].data.astype(int) & mask) != 0
        ds[variable].data[bad, ...] = np.nan
        if self.output_dataset != dataset:
            datatree[self.output_dataset] = subtree
        return datatree

    def _validate(self, datatree: DataTree) -> None:
        dataset = self.config["dataset"]
        variable = self.config["variable"]
        flags = self.config["flags"]
        group = _dataset_group(dataset)
        if group is None:
            raise PalError(
                "FlagClean only supports datasets matching one of the "
                f"patterns for groups {sorted(DATASET_GROUPS)}, got {dataset!r}"
            )
        group_defaults = DEFAULT_REJECT_BITS[group]
        ds = datatree[dataset].ds
        if variable not in ds.data_vars:
            raise PalError(f"FlagClean: {variable=} not available in {dataset}")
        if not flags:
            raise PalError("FlagClean: at least one entry in `flags` is required")
        for entry in flags:
            flag_name = entry.get("flag_name")
            if flag_name not in group_defaults:
                raise PalError(
                    f"FlagClean: unsupported flag_name {flag_name!r} for "
                    f"{dataset!r} (group {group!r}), must be one of "
                    f"{sorted(group_defaults)}"
                )
            if flag_name not in ds.data_vars:
                raise PalError(f"FlagClean: {flag_name=} not available in {dataset}")


def _outlier_mask(
    x: np.ndarray, window_size: int, method: str, multiplier: float
) -> np.ndarray:
    """Detect outliers using a centered moving-window statistic

    Independent of `tfalib.outliers`: this uses pandas' native
    ``rolling(..., center=True)`` rather than a trailing window manually
    re-centered with `np.roll`. `np.roll` wraps circularly, so points near
    the end of the series would otherwise be compared against a window
    built from the *start* of the series instead of correctly falling back
    to NaN (no detection) like the equivalent points at the start do - see
    the note in `tfalib.moving_q25_and_q75`, which has this bug.
    """
    original_shape = x.shape
    x2d = x.reshape(original_shape[0], -1)
    window = pd.DataFrame(x2d).rolling(
        window_size, min_periods=window_size // 2, center=True
    )
    if method == "iqr":
        q25 = window.quantile(0.25, interpolation="linear").to_numpy()
        q75 = window.quantile(0.75, interpolation="linear").to_numpy()
        spread = q75 - q25
        lower, upper = q25 - multiplier * spread, q75 + multiplier * spread
    elif method == "normal":
        mean = window.mean().to_numpy()
        std = window.std().to_numpy()
        lower, upper = mean - multiplier * std, mean + multiplier * std
    else:
        raise PalError(f"Despike: unknown method {method!r}, must be 'iqr' or 'normal'")
    bad = (x2d < lower) | (x2d > upper)
    return bad.reshape(original_shape)


class Despike(PalProcess):
    """Detect and mask outliers in a variable using a moving-window statistic"""

    @property
    def process_name(self) -> str:
        return "Despike"

    def set_config(
        self,
        dataset: str = "SW_OPER_MAGA_LR_1B",
        variable: str = "",
        window_size: int = 10,
        method: str = "iqr",
        multiplier: float = 0.5,
        output_dataset: str = "",
    ) -> None:
        """Set the process configuration

        Parameters
        ----------
        dataset : str
            Selects this dataset from the datatree
        variable : str
            Name of the variable to despike
        window_size : int, optional
            The size (number of points) of the rolling window, by default 10
        method : str, optional
            "normal" or "iqr", by default "iqr"
        multiplier : float, optional
            Indicates the spread of the zone of accepted values, by default 0.5
        output_dataset : str
            Sets the name of the dataset in the data tree that this process
            writes results to, by default the same as `dataset`
        """
        super().set_config(
            dataset=dataset,
            variable=variable,
            window_size=window_size,
            method=method,
            multiplier=multiplier,
            output_dataset=output_dataset or dataset,
        )

    def _call(self, datatree: DataTree) -> DataTree:
        dataset = self.config["dataset"]
        variable = self.config["variable"]
        subtree = datatree[dataset]
        ds = subtree.ds
        if variable not in ds.data_vars:
            raise PalError(f"Despike: {variable=} not available in {dataset}")
        inds = _outlier_mask(
            ds[variable].data,
            self.config["window_size"],
            self.config["method"],
            self.config["multiplier"],
        )
        ds[variable].data[inds] = np.nan
        if self.output_dataset != dataset:
            datatree[self.output_dataset] = subtree
        return datatree


class InterpolateGaps(PalProcess):
    """Linearly interpolate over NaN gaps in a variable"""

    @property
    def process_name(self) -> str:
        return "InterpolateGaps"

    def set_config(
        self,
        dataset: str = "SW_OPER_MAGA_LR_1B",
        variable: str = "",
        output_dataset: str = "",
    ) -> None:
        """Set the process configuration

        Parameters
        ----------
        dataset : str
            Selects this dataset from the datatree
        variable : str
            Name of the variable to interpolate
        output_dataset : str
            Sets the name of the dataset in the data tree that this process
            writes results to, by default the same as `dataset`
        """
        super().set_config(
            dataset=dataset,
            variable=variable,
            output_dataset=output_dataset or dataset,
        )

    def _call(self, datatree: DataTree) -> DataTree:
        dataset = self.config["dataset"]
        variable = self.config["variable"]
        subtree = datatree[dataset]
        ds = subtree.ds
        if variable not in ds.data_vars:
            raise PalError(f"InterpolateGaps: {variable=} not available in {dataset}")
        self._interpolate(ds[variable])
        if self.output_dataset != dataset:
            datatree[self.output_dataset] = subtree
        return datatree

    @staticmethod
    def _interpolate(da: DataArray) -> None:
        shape = da.data.shape
        if len(shape) == 1:
            n = shape[0]
            t_ind = np.arange(n)
            x = da.data
            not_nan = ~np.isnan(x)
            da.data[:] = np.interp(t_ind, t_ind[not_nan], x[not_nan])
        else:
            n, d = shape
            t_ind = np.arange(n)
            for i in range(d):
                x = np.reshape(da.data[:, i], (n,))
                not_nan = ~np.isnan(x)
                da.data[:, i] = np.interp(t_ind, t_ind[not_nan], x[not_nan])
