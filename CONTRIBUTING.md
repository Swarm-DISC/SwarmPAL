See the SwarmPAL [documentation](https://swarmpal.readthedocs.io/en/latest/contributing.html)  as well as the [development notes on HackMD](https://hackmd.io/@swarm/dev/%2Ff6YIHfqxT9yL0giWJzhr_Q) for development of this package.

Useful references:
- https://github.com/scientific-python/cookie
- https://www.pyopensci.org/python-package-guide/

---

# Quick development

This project uses [uv](https://docs.astral.sh/uv/) for environment and dependency
management and [nox](https://nox.thea.codes/) as a task runner. Install uv (see the uv
docs), then let nox drive the common tasks — it builds isolated environments for you via
uv, so you don't have to manage them by hand. If you don't have nox installed, `uvx nox`
runs it without installing.

```console
$ uvx nox -s lint             # Run pre-commit (lint + format) on all files
$ uvx nox -s tests            # Run the test suite on all installed Python versions
$ uvx nox -s tests-3.10       # Run tests on a specific Python version
$ uvx nox -s docs             # Build the docs (fast mode: skips notebook execution)
$ uvx nox -s docs -- --full   # Full docs build (executes notebooks + autoapi)
$ uvx nox -s docs -- serve    # Build and serve the docs locally
```

# Setting up a development environment manually

Create and sync an environment with uv (this reads `pyproject.toml` and `uv.lock` and
installs SwarmPAL in editable mode):

```bash
uv sync --group test --group apexpy_wheels --extra experimental
```

Dependency groups (`dev`, `test`, `docs`) and optional extras (`experimental`, `dsecs`)
are defined in `pyproject.toml`. The `apexpy_wheels` group provides prebuilt `apexpy`
wheels so you don't need a Fortran compiler locally. Run commands inside the environment
with `uv run`, e.g. `uv run pytest`.

# Testing

Run the unit checks with pytest:

```bash
uv run pytest
```

Tests that reach remote servers (VirES) are marked `remote`. To skip them:

```bash
uv run pytest -m "not remote"
```

# Building docs

Use nox (see above) — `uvx nox -s docs` for a fast build, or `-- serve` to preview:

```bash
uvx nox -s docs -- serve
```

# Pre-commit

This project uses pre-commit for all style checking. While you can run it through nox
(`uvx nox -s lint`), it's worth installing on its own and enabling the git hook so checks
run automatically on each commit:

```bash
uv tool install pre-commit  # or: pipx install pre-commit / brew install pre-commit
pre-commit install          # install the hook into this repo
```

You can also run it against all files at any time:

```bash
pre-commit run -a
```
