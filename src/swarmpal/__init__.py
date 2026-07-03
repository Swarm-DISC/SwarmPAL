from __future__ import annotations

from swarmpal import express, io, toolboxes, utils
from swarmpal._api import (
    apply_process,
    apply_processes,
    fetch_data,
    make_process,
    quicklook,
)

try:
    from swarmpal._version import __version__
except ModuleNotFoundError:
    # _version.py is generated at build time; fall back to installed metadata
    from importlib.metadata import PackageNotFoundError, version

    try:
        __version__ = version("swarmpal")
    except PackageNotFoundError:
        __version__ = "unknown"

__all__ = (
    "__version__",
    "apply_process",
    "apply_processes",
    "express",
    "fetch_data",
    "io",
    "make_process",
    "quicklook",
    "toolboxes",
    "utils",
)
