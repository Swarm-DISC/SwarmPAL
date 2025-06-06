from __future__ import annotations

import pooch
import yaml
from xarray import open_datatree

# from swarmpal import version
from swarmpal.io._paldata import PalDataItem

POOCH = pooch.create(
    # Download location. Defaults to ~/.cache/swarmpal_test_data on Linux.
    path=pooch.os_cache("swarmpal_test_data"),
    base_url="https://raw.githubusercontent.com/Swarm-DISC/SwarmPal-test-data/refs/heads/main",
    # version='main',
    version_dev="main",
    registry={
        "registry.txt": "md5:f93b9c46046ed30bd9e8f7fe94441eaf",
    },
)
POOCH.load_registry(POOCH.fetch("registry.txt"))

# git-lfs files are not 'in' the git repository, but can be downloaded from slightly different URL.
# See: https://stackoverflow.com/questions/45117476/access-download-on-git-lfs-file-via-raw-githubusercontent-com
POOCH_LFS = pooch.create(
    path=pooch.os_cache("swarmpal_test_data"),
    base_url="https://media.githubusercontent.com/media/Swarm-DISC/SwarmPal-test-data/refs/heads/main",
    # version='main',
    version_dev="main",
    registry=None,
)
POOCH_LFS.load_registry(POOCH.fetch("registry.txt"))


def get_local_filename(filename):
    """Returns the absolute path to a filename in the test set."""
    data_filename = f"data/{filename}"
    return POOCH_LFS.fetch(data_filename)


def load_test_dataset(filename, group=""):
    """Returns a test dataset as a SwarmPal PalDataItem"""
    local_filename = get_local_filename(filename)
    return PalDataItem.from_file(local_filename, group=group)


def load_test_datatree(filename, group=""):
    """Returns a test dataset as an xarray.DataTree"""
    local_filename = get_local_filename(filename)
    return open_datatree(local_filename, group=group)


def load_test_config(dataset_name):
    """Returns the configuration of a test dataset as dictionary"""
    config_filename = f"config/{dataset_name}.yaml"
    filename = POOCH.fetch(config_filename)
    with open(filename) as f:
        datasets = yaml.safe_load(f)
        return datasets
