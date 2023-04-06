# Installation

## ESA Virtual Research Environment

> To be updated

## Install latest release

> To be updated

<!-- The package is available from PyPI:

```bash
pip install swarmpal
``` -->

:::{admonition} New to Python?
:class: note

To setup Python on your system, check guidance on the [viresclient installation notes](https://viresclient.readthedocs.io/en/latest/installation.html#recommended-setup-if-starting-without-python-already)
:::

## Install latest development version

Assuming you have a compatible system with git, a fortran compiler, and a Python>=3.8 installation with a recent version of pip, you can install the latest development version from the `staging` branch with:

```bash
pip install git+https://github.com/Swarm-DISC/SwarmPAL@staging#egg=swarmpal[dsecs]
```

:::{admonition} Fotran compiler?
:class: note

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

Clone the repository and install it in editable mode:

```bash
git clone https://github.com/Swarm-DISC/SwarmPAL
pip install -e SwarmPAL/
```

For more information check the [development Guide on HackMD](https://hackmd.io/@swarm/dev/%2Ff6YIHfqxT9yL0giWJzhr_Q)
