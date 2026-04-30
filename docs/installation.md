# Installation

## Swarm Virtual Research Environment

The easiest way to use SwarmPAL is in the Swarm Virtual Research Environment (read more [here](https://notebooks.vires.services/)). To get started with the SwarmPAL demo tool (which includes the examples given on these pages, as interactive notebooks), follow this link: [![Swarm-VRE](https://img.shields.io/badge/%F0%9F%9A%80%20launch-Swarm--VRE-blue)](https://vre.vires.services/hub/user-redirect/git-pull?repo=https%3A%2F%2Fgithub.com%2Fsmithara%2Fswarmpal-demo&urlpath=lab%2Ftree%2Fswarmpal-demo%2FREADME.ipynb&branch=main)

## Install latest release

The package is available from PyPI:

If you *do not need* the DSECS toolbox:
```bash
pip install swarmpal[experimental]
```

If you *do need* the DSECS toolbox:
```bash
pip install swarmpal[dsecs,experimental]
```
which includes [apexpy](https://github.com/aburrell/apexpy), which is needed for the DSECS toolbox. There can be some trouble installing this so you might need to manually install apexpy first.

:::{admonition} New to Python?

To setup Python on your system, check guidance on the [viresclient installation notes](https://viresclient.readthedocs.io/en/latest/installation.html#recommended-setup-if-starting-without-python-already)

:::

## Install for development

(using uv)

```
git clone git@github.com:Swarm-DISC/SwarmPAL.git
cd SwarmPAL
uv venv --python 3.11
uv sync --frozen --all-groups --all-extras
```

You can also use nox to run tests and build docs using ephemeral environments (they live in the `.nox` directory), e.g.:
```
uvx nox -s tests
uvx nox -s docs -- no-exec -- serve
```
