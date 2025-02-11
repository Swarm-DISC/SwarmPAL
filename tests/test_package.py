from __future__ import annotations

import pytest

import swarmpal as m


def test_version():
    assert m.__version__


def test_imports():
    from swarmpal.io import PalDataItem, create_paldata
    from swarmpal.toolboxes import fac, tfa

    for module in (PalDataItem, create_paldata, fac, tfa):
        print(module.__doc__)


@pytest.mark.dsecs()
def test_imports_dsecs():
    from swarmpal.toolboxes import dsecs

    for module in (dsecs,):
        print(module.__doc__)
