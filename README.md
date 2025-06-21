# SwarmPAL

---
[![Swarm-VRE](https://img.shields.io/badge/%F0%9F%9A%80%20launch-Swarm--VRE-blue)](https://vre.vires.services/hub/user-redirect/git-pull?repo=https%3A%2F%2Fgithub.com%2Fsmithara%2Fswarmpal-demo&urlpath=lab%2Ftree%2Fswarmpal-demo%2FREADME.ipynb&branch=main)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/smithara/swarmpal-demo/HEAD)

[![PyPI](https://img.shields.io/pypi/v/swarmpal)]( https://pypi.org/project/swarmpal/)
[![Documentation](https://img.shields.io/badge/docs-online-success)][rtd-link]
[![GitHub Discussion][github-discussions-badge]][github-discussions-link]
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7826899.svg)](https://doi.org/10.5281/zenodo.7826899)

[![Actions Status][actions-badge]][actions-link]
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Code style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/format.json)](https://github.com/astral-sh/ruff)

---

[actions-badge]:            https://github.com/Swarm-DISC/SwarmPAL/workflows/CI/badge.svg
[actions-link]:             https://github.com/Swarm-DISC/SwarmPAL/actions
[black-badge]:              https://img.shields.io/badge/code%20style-black-000000.svg
[black-link]:               https://github.com/psf/black
[conda-badge]:              https://img.shields.io/conda/vn/conda-forge/swarmpal
[conda-link]:               https://github.com/conda-forge/swarmpal-feedstock
[contribute-badge]:         https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg
[contribute-link]:          CODE_OF_CONDUCT.md
[github-discussions-badge]: https://img.shields.io/static/v1?label=Discussions&message=Ask&color=blue&logo=github
[github-discussions-link]:  https://github.com/Swarm-DISC/SwarmPAL/discussions
[pypi-link]:                https://pypi.org/project/swarmpal/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/swarmpal
[pypi-version]:             https://badge.fury.io/py/swarmpal.svg
[rtd-badge]:                https://img.shields.io/badge/docs-online-success
[rtd-link]:                 https://swarmpal.readthedocs.io/

SwarmPAL (*Swarm Product Algorithm Laboratory*) provides tools to perform higher level processing of data from [*Swarm*](https://earth.esa.int/eogateway/missions/swarm) and beyond, supported by [Swarm DISC](https://earth.esa.int/eogateway/activities/swarm-disc). To access data, SwarmPAL connects to VirES and HAPI servers (using [viresclient](https://github.com/ESA-VirES/VirES-Python-Client/) & [hapiclient](https://github.com/hapi-server/client-python/)). Processing code is organised within *toolboxes* that are contributed by different teams.

![SwarmPAL mermaid diagram](https://mermaid.ink/img/pako:eNqVlG1r2zAQgP-K0L60YGduEteOKWVeXthgHWUJG6wuRbHPiagiBVnu4ob0t-9kN0mzQtcYjKTTc686aU1TlQGNqOu6iUyVzPksSiQhglWqNBEBcZ_IejMX6k86Z9qQbz8sQUhRTmeaLedkwAy7SagdSKFKnUKR0NsGyriG1HAlyeRzI7HfT66HY1SpR2Q_rUkxZ0uISDbdPD097ckv8fVXBO3wJjfiApDrD0bkI_kOBifIHyBXTJZMINRMdiGCzBL5T0oTpcRUrTCRN7LY0aO4v5fW0cT9Fv53BZczAXcFM_vIBTDpis1eofb_yuZgPOyPD63Wota1hqVWWOPilc2DdPcasWSiKnhxdAyTUXxoDwVH-bd8326-D8VDNKDfx_5iDyDg-LrCymALDFfoCMtCTHPQu2bYfkxWN7GsyHOqt-_1g6dOMN6m8vUMg61HdPyq37jEMHKWwsnJhb2JlyumNasupvrS3qaJBjg93cL1_XLdy70WubDLF7363MWmEmD9kZwLEX3I89wpjFb34GYMs7AuIuIT_yWOCf8Ppw5dgF4wnuGLsbbKCTVzWEBCI5xmkLNSmIQ6zVaupBnzx3q37SVyg_qsNGpcyZRGRpfgUK3K2ZxGORMFrsplxgwMOMOjWuykSyZ_K7XYqsy09f-sjrUE3VelNDRqe0EN02hNVzTqeq3wLOh0vdD3wo7nhQ6tEOr5rY7XPkOhH_q9TuBvHPpYm0f-PDwPvLPzdi8IvG6n61DIuFH6qnkk67dy8xcI0Jd6?type=png)
<!--
I had trouble with getting sphinxcontrib-mermaid to play nice with this diagram, so access it here instead:

https://mermaid.live/edit#pako:eNqVlG1r2zAQgP-K0L60YGduEteOKWVeXthgHWUJG6wuRbHPiagiBVnu4ob0t-9kN0mzQtcYjKTTc686aU1TlQGNqOu6iUyVzPksSiQhglWqNBEBcZ_IejMX6k86Z9qQbz8sQUhRTmeaLedkwAy7SagdSKFKnUKR0NsGyriG1HAlyeRzI7HfT66HY1SpR2Q_rUkxZ0uISDbdPD097ckv8fVXBO3wJjfiApDrD0bkI_kOBifIHyBXTJZMINRMdiGCzBL5T0oTpcRUrTCRN7LY0aO4v5fW0cT9Fv53BZczAXcFM_vIBTDpis1eofb_yuZgPOyPD63Wota1hqVWWOPilc2DdPcasWSiKnhxdAyTUXxoDwVH-bd8326-D8VDNKDfx_5iDyDg-LrCymALDFfoCMtCTHPQu2bYfkxWN7GsyHOqt-_1g6dOMN6m8vUMg61HdPyq37jEMHKWwsnJhb2JlyumNasupvrS3qaJBjg93cL1_XLdy70WubDLF7363MWmEmD9kZwLEX3I89wpjFb34GYMs7AuIuIT_yWOCf8Ppw5dgF4wnuGLsbbKCTVzWEBCI5xmkLNSmIQ6zVaupBnzx3q37SVyg_qsNGpcyZRGRpfgUK3K2ZxGORMFrsplxgwMOMOjWuykSyZ_K7XYqsy09f-sjrUE3VelNDRqe0EN02hNVzTqeq3wLOh0vdD3wo7nhQ6tEOr5rY7XPkOhH_q9TuBvHPpYm0f-PDwPvLPzdi8IvG6n61DIuFH6qnkk67dy8xcI0Jd6

-->

![SwarmPAL diagram](https://swarmdisc.org/wp-content/uploads/2024/02/SwarmPAL-diagrams-overview.png)

For more information see:

- <https://swarmpal.readthedocs.io/>
- <https://swarmdisc.org/lab/>
