# Installation

:::{tip}
Try the [SwarmPAL dashboards](https://dev.swarmdisc.org/swarmpal-processor/)
:::

:::{note}
SwarmPAL is available in the Swarm Virtual Research Environment - [read more here](https://notebooks.vires.services/)
:::

## Install latest release

The package is available from PyPI:

::::::{tab-set}

:::::{tab-item} pip

::::{tab-set}

:::{tab-item} Full installation
```bash
pip install swarmpal[dsecs,experimental]
```
:::

:::{tab-item} Minimal installation
```bash
pip install swarmpal
```
:::

::::

:::::

:::::{tab-item} uv

::::{tab-set}

:::{tab-item} Full installation
```bash
uv add swarmpal[dsecs,experimental]
```
:::

:::{tab-item} Minimal installation
```bash
uv add swarmpal
```
:::

::::

:::::

::::::

:::{admonition} New to Python?

To setup Python on your system, check guidance on the [viresclient installation notes](https://viresclient.readthedocs.io/en/latest/installation.html#recommended-setup-if-starting-without-python-already)

:::

## Install for development

(using [uv](https://docs.astral.sh/uv/))

```bash
git clone git@github.com:Swarm-DISC/SwarmPAL.git
cd SwarmPAL
uv venv --python 3.11
uv sync --frozen --all-groups --all-extras
```

(You might need to omit ``--frozen`` or instead use ``--locked``)

You can also use nox to run tests and build docs using ephemeral environments (they live in the `.nox` directory), e.g.:

```bash
uvx nox -s tests
uvx nox -s docs -- no-exec -- serve
```
