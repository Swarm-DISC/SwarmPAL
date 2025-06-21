# Installation

## Swarm Virtual Research Environment

The easiest way to use SwarmPAL is in the Swarm Virtual Research Environment (read more [here](https://notebooks.vires.services/)). To get started with the SwarmPAL demo tool (which includes the examples given on these pages, as interactive notebooks), follow this link: [![Swarm-VRE](https://img.shields.io/badge/%F0%9F%9A%80%20launch-Swarm--VRE-blue)](https://vre.vires.services/hub/user-redirect/git-pull?repo=https%3A%2F%2Fgithub.com%2Fsmithara%2Fswarmpal-demo&urlpath=lab%2Ftree%2Fswarmpal-demo%2FREADME.ipynb&branch=main)

## Install latest release

The package is available from PyPI:

If you *do not need* the DSECS toolbox:
```bash
pip install swarmpal
```

If you *do need* the DSECS toolbox:
```bash
pip install swarmpal[dsecs]
```
which includes [apexpy](https://github.com/aburrell/apexpy), which is needed for the DSECS toolbox. There can be some trouble installing this so you might need to manually install apexpy first.

To include extra experimental features:
```bash
pip install swarmpal[dsecs,experimental]
```

:::{admonition} New to Python?

To setup Python on your system, check guidance on the [viresclient installation notes](https://viresclient.readthedocs.io/en/latest/installation.html#recommended-setup-if-starting-without-python-already)

:::

## Install latest development version

Assuming you have a compatible system with git, a fortran compiler, and a Python>=3.10 installation with a recent version of pip, you can install the latest development version from the `staging` branch with:

```bash
pip install git+https://github.com/Swarm-DISC/SwarmPAL@staging#egg=swarmpal[dsecs,experimental,test]
```

:::{admonition} Fortran compiler?

If you are using conda, you can get one from:
``` bash
conda install conda-forge::fortran-compiler
```

:::

The fortran compiler is required in order to install the dependency, apexpy. It may be better to try installing apexpy first and debugging that if you run into trouble.

To bypass installation of apexpy (so disabling usage of the DSECS toolbox), you can use pip without the `[dsecs]` option:

```bash
pip install git+https://github.com/Swarm-DISC/SwarmPAL@staging#egg=swarmpal
```

## Install for local development

Clone the repository and install it in editable mode :

```bash
git clone https://github.com/Swarm-DISC/SwarmPAL
pip install -e SwarmPAL[dsecs,experimental,test,dev,docs]
pytest SwarmPAL/
```

````{tip}
Instructions for using [uv](https://docs.astral.sh/uv/reference/cli/) will be provided soon. e.g.:
```bash
uv venv
uv sync --all-extras
uv run pytest
```
````

For more information check the [development Guide on HackMD](https://hackmd.io/@swarm/dev/%2Ff6YIHfqxT9yL0giWJzhr_Q)
