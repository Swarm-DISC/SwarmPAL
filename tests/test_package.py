from __future__ import annotations

import swarmpal as m


def test_version():
    assert m.__version__


def test_imports():
    from swarmpal.toolboxes.tfa import tfa_processor, tfalib

    for module in (tfa_processor, tfalib):
        print(module.__doc__)
