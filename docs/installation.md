# Installation

The package is in early development so is not recommended for use yet. The following instructions are a guide how to set up for testing and evaluation. For development, see the [Development Guide on HackMD](https://hackmd.io/@swarm/dev/%2Ff6YIHfqxT9yL0giWJzhr_Q).

(If you want to begin with a reasonable conda environment file, you can check [this one used for development](https://raw.githubusercontent.com/Swarm-DISC/SwarmPAL/staging/environment.yml))

Assuming you have a compatible system with git, a fortran compiler, and a Python>=3.8 installation with a recent version of pip, you can install the latest development version from the `staging` branch with:

```
pip install git+https://github.com/Swarm-DISC/SwarmPAL@staging#egg=swarmpal[dsecs]
```

The fortran compiler is required in order to install the dependency, apexpy. It may be better to try installing apexpy first and debugging that if you run into trouble.

To bypass installation of apexpy (so disabling usage of the DSECS toolbox), you can use pip without the `[dsecs]` option:

```
pip install git+https://github.com/Swarm-DISC/SwarmPAL@staging#egg=swarmpal
```
