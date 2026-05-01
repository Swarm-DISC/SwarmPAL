# Changelog

## 0.3 (2026-05-01)

- Refreshed development process to use `uv`
- Refactored `pal_meta` to contain enough information to reconstruct the data and be more flexible
- Can be driven from CLI using YAML configuration files
- Top level quicklook function that displays a quicklook plot defined by the applied toolbox - also provides an xarray accessor (so you can access like `data.swarmpal.quicklook()`)

## 0.2 (2025-06-21)

- Requires Python ≥ 3.10
- Update to switch from xarray-datatree package (now deprecated) to Xarray ≥ 2024.10
- Basic CLI
- Tweaks to FAC, TFA, and DSECS tools
- New approach to testing via cached data at <https://github.com/Swarm-DISC/SwarmPAL-test-data>
- Now available also as a containerised distribution (i.e. Docker image): <https://github.com/Swarm-DISC/SwarmPAL-processor>
- Accessible through dashboards under development: <https://dev.swarmdisc.org/swarmpal-processor/>

## 0.1 (2023-04-13)

- First release ✨
- Reasonable implementation of using `DataTree` and `PalProcess` base class
- Includes first iteration of TFA and DSECS Swarm projects
