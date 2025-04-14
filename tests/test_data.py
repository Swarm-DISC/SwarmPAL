from __future__ import annotations

import pooch
import yaml

# from swarmpal import version
from swarmpal.io._paldata import PalDataItem

POOCH = pooch.create(
    # Download location. Defaults to ~/.cache/swarmpal_test_data on Linux.
    path=pooch.os_cache("swarmpal_test_data"),
    base_url="https://raw.githubusercontent.com/dawiedotcom/SwarmPal-test-data/refs/heads/main",
    # version='main',
    version_dev="main",
    registry={
        "registry.txt": "md5:48d2b7b830b9fa7c2eaec28cb0bc1c07",
        "config.yaml": "md5:f5b6ca36ee46a61b956bc14dc3db8abb",
    },
)

# git-lfs files are not 'in' the git repository, but can be downloaded from slightly different URL.
# See: https://stackoverflow.com/questions/45117476/access-download-on-git-lfs-file-via-raw-githubusercontent-com
POOCH_LFS = pooch.create(
    path=pooch.os_cache("swarmpal_test_data/data"),
    base_url="https://media.githubusercontent.com/media/dawiedotcom/SwarmPal-test-data/refs/heads/main/data",
    # version='main',
    version_dev="main",
    registry=None,
)

POOCH_LFS.load_registry(POOCH.fetch("registry.txt"))


def load_test_dataset(filename, group=""):
    """Returns a test dataset as a SwarmPal PalDataItem"""
    local_filename = POOCH_LFS.fetch(filename)
    return PalDataItem.from_file(local_filename, group=group)


def load_test_config(dataset_name):
    """Returns the configuration of a test dataset as dictionary"""
    filename = POOCH.fetch("config.yaml")
    with open(filename) as f:
        datasets = yaml.safe_load(f)
        return datasets[dataset_name]
