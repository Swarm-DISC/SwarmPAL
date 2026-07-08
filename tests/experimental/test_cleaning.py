from __future__ import annotations

import numpy as np
import pytest
from xarray import Dataset, DataTree

import swarmpal
from swarmpal import apply_processes
from swarmpal.experimental import Despike, FlagClean, InterpolateGaps
from swarmpal.utils.exceptions import PalError

DATASET_NAME = "SW_OPER_MAGA_LR_1B"


def _make_datatree(n=20):
    timestamps = np.arange(n)
    F = np.full(n, 45000.0)
    B_NEC = np.full((n, 3), 45000.0)
    ds = Dataset(
        data_vars={
            "F": ("Timestamp", F),
            "B_NEC": (("Timestamp", "NEC"), B_NEC),
            "Flags_F": ("Timestamp", np.zeros(n, dtype=int)),
            "Flags_B": ("Timestamp", np.zeros(n, dtype=int)),
            "Flags_Platform": ("Timestamp", np.zeros(n, dtype=int)),
            "Flags_q": ("Timestamp", np.zeros(n, dtype=int)),
        },
        coords={"Timestamp": timestamps, "NEC": ["N", "E", "C"]},
    )
    return DataTree.from_dict({DATASET_NAME: ds})


def test_cleaning_by_name():
    """New experimental cleaning processes are added to the process lookup"""
    processes = {
        "FlagClean": FlagClean,
        "Despike": Despike,
        "InterpolateGaps": InterpolateGaps,
    }
    for process_name, process in processes.items():
        new_process = swarmpal.make_process(process_name)
        assert isinstance(new_process, process)
        assert new_process.process_name == process_name


def test_flagclean_default_bits():
    """Default reject bits mask known-bad values and leave clean ones alone"""
    data = _make_datatree()
    ds = data[DATASET_NAME].ds
    # bit 1 set -> bad (outlier/gap); index 0
    ds["Flags_F"].data[0] = 2
    # bit 0 only (mode indicator) -> not bad by default; index 1
    ds["Flags_F"].data[1] = 1
    # clean; index 2 stays 0
    process = FlagClean(
        config={
            "dataset": DATASET_NAME,
            "variable": "F",
            "flags": [{"flag_name": "Flags_F"}],
        }
    )
    data = process(data)
    F = data[DATASET_NAME]["F"].data
    assert np.isnan(F[0])
    assert not np.isnan(F[1])
    assert not np.isnan(F[2])


def test_flagclean_explicit_reject_bits():
    """Explicit reject_bits overrides the default preset"""
    data = _make_datatree()
    ds = data[DATASET_NAME].ds
    ds["Flags_B"].data[0] = 1  # bit 0, excluded from default, but requested here
    process = FlagClean(
        config={
            "dataset": DATASET_NAME,
            "variable": "B_NEC",
            "flags": [{"flag_name": "Flags_B", "reject_bits": [0]}],
        }
    )
    data = process(data)
    B_NEC = data[DATASET_NAME]["B_NEC"].data
    assert np.all(np.isnan(B_NEC[0, :]))
    assert not np.any(np.isnan(B_NEC[1, :]))


def test_flagclean_combines_multiple_flags():
    """Multiple flag fields in one call OR-combine their bad masks"""
    data = _make_datatree()
    ds = data[DATASET_NAME].ds
    ds["Flags_B"].data[0] = 2  # bad via Flags_B
    ds["Flags_Platform"].data[1] = 2  # bad via Flags_Platform (thrusters activated)
    process = FlagClean(
        config={
            "dataset": DATASET_NAME,
            "variable": "B_NEC",
            "flags": [
                {"flag_name": "Flags_B"},
                {"flag_name": "Flags_Platform"},
            ],
        }
    )
    data = process(data)
    B_NEC = data[DATASET_NAME]["B_NEC"].data
    assert np.all(np.isnan(B_NEC[0, :]))
    assert np.all(np.isnan(B_NEC[1, :]))
    assert not np.any(np.isnan(B_NEC[2, :]))


def test_flagclean_rejects_non_maglr1b_dataset():
    """FlagClean is scoped to SW_{OPER,FAST}_MAGx_LR_1B datasets"""
    ds = Dataset(
        data_vars={
            "F": ("Timestamp", np.zeros(5)),
            "Flags_F": ("Timestamp", np.zeros(5, dtype=int)),
        },
        coords={"Timestamp": np.arange(5)},
    )
    data = DataTree.from_dict({"SW_OPER_EFIA_TCT02": ds})
    process = FlagClean(
        config={
            "dataset": "SW_OPER_EFIA_TCT02",
            "variable": "F",
            "flags": [{"flag_name": "Flags_F"}],
        }
    )
    with pytest.raises(PalError):
        process(data)


def test_flagclean_rejects_unsupported_flag_name():
    """Only Flags_F, Flags_B, Flags_Platform, Flags_q are supported"""
    data = _make_datatree()
    process = FlagClean(
        config={
            "dataset": DATASET_NAME,
            "variable": "F",
            "flags": [{"flag_name": "Flags_bogus"}],
        }
    )
    with pytest.raises(PalError):
        process(data)


def test_flagclean_q_default_rejects_nothing():
    """Flags_q has no default reject bits, since it isn't a clean bitfield"""
    data = _make_datatree()
    ds = data[DATASET_NAME].ds
    ds["Flags_q"].data[:] = 255  # sentinel: not enough STR data
    process = FlagClean(
        config={
            "dataset": DATASET_NAME,
            "variable": "F",
            "flags": [{"flag_name": "Flags_q"}],
        }
    )
    data = process(data)
    F = data[DATASET_NAME]["F"].data
    assert not np.any(np.isnan(F))


def test_flagclean_q_explicit_reject_bits():
    """Flags_q can still mask via explicitly requested reject_bits"""
    data = _make_datatree()
    ds = data[DATASET_NAME].ds
    ds["Flags_q"].data[0] = 8  # bit 3: on-ground aberrational correction applied
    process = FlagClean(
        config={
            "dataset": DATASET_NAME,
            "variable": "F",
            "flags": [{"flag_name": "Flags_q", "reject_bits": [3]}],
        }
    )
    data = process(data)
    F = data[DATASET_NAME]["F"].data
    assert np.isnan(F[0])
    assert not np.isnan(F[1])


def test_despike_masks_spike():
    """Despike detects an injected spike and masks it as NaN"""
    data = _make_datatree(n=30)
    ds = data[DATASET_NAME].ds
    ds["F"].data[15] += 1000  # inject a spike
    process = Despike(
        config={
            "dataset": DATASET_NAME,
            "variable": "F",
            "window_size": 10,
            "method": "iqr",
            "multiplier": 0.5,
        }
    )
    data = process(data)
    F = data[DATASET_NAME]["F"].data
    assert np.isnan(F[15])
    assert not np.any(np.isnan(np.delete(F, 15)))


def test_despike_no_spurious_edge_flags():
    """A level shift at the start of the series must not flag the tail

    Regression test for a wraparound bug in tfalib.moving_q25_and_q75 (uses
    np.roll to re-center a trailing rolling window, which wraps circularly
    and leaks quantiles from the start of the series onto the last few
    points). Despike is implemented independently of tfalib to avoid this.
    """
    data = _make_datatree(n=200)
    ds = data[DATASET_NAME].ds
    rng = np.random.default_rng(0)
    F = 45000 + rng.normal(scale=0.5, size=200)
    F[:20] += 50  # baseline offset only at the start, no real spike anywhere
    ds["F"].data[:] = F
    process = Despike(
        config={
            "dataset": DATASET_NAME,
            "variable": "F",
            "window_size": 20,
            "method": "iqr",
            "multiplier": 0.5,
        }
    )
    data = process(data)
    result = data[DATASET_NAME]["F"].data
    assert not np.isnan(result[0])
    assert not np.isnan(result[-1])


def test_interpolategaps_1d():
    """InterpolateGaps fills NaN gaps in a 1-D variable"""
    data = _make_datatree(n=10)
    ds = data[DATASET_NAME].ds
    ds["F"].data[:] = np.arange(10, dtype=float)
    ds["F"].data[3] = np.nan
    ds["F"].data[4] = np.nan
    process = InterpolateGaps(config={"dataset": DATASET_NAME, "variable": "F"})
    data = process(data)
    F = data[DATASET_NAME]["F"].data
    assert not np.any(np.isnan(F))
    np.testing.assert_allclose(F, np.arange(10, dtype=float))


def test_interpolategaps_2d():
    """InterpolateGaps fills NaN gaps in a 2-D (vector) variable"""
    data = _make_datatree(n=10)
    ds = data[DATASET_NAME].ds
    ramp = np.tile(np.arange(10, dtype=float)[:, None], (1, 3))
    ds["B_NEC"].data[:] = ramp
    ds["B_NEC"].data[5, :] = np.nan
    process = InterpolateGaps(config={"dataset": DATASET_NAME, "variable": "B_NEC"})
    data = process(data)
    B_NEC = data[DATASET_NAME]["B_NEC"].data
    assert not np.any(np.isnan(B_NEC))
    np.testing.assert_allclose(B_NEC, ramp)


def test_chained_cleaning_processes():
    """FlagClean -> Despike -> InterpolateGaps chain via apply_processes"""
    data = _make_datatree(n=30)
    ds = data[DATASET_NAME].ds
    ds["F"].data[:] = 45000.0
    ds["Flags_F"].data[5] = 2  # bad by flag
    ds["F"].data[15] += 1000  # spike
    process_params = [
        dict(
            process_name="FlagClean",
            dataset=DATASET_NAME,
            variable="F",
            flags=[{"flag_name": "Flags_F"}],
        ),
        dict(
            process_name="Despike",
            dataset=DATASET_NAME,
            variable="F",
            window_size=10,
            method="iqr",
            multiplier=0.5,
        ),
        dict(
            process_name="InterpolateGaps",
            dataset=DATASET_NAME,
            variable="F",
        ),
    ]
    data = apply_processes(data, process_params)
    F = data[DATASET_NAME]["F"].data
    assert not np.any(np.isnan(F))
    np.testing.assert_allclose(F, 45000.0, atol=1e-6)
