# Quickstart

```{tip}
Try the [SwarmPAL dashboards](https://dev.swarmdisc.org/swarmpal-processor/)
```

There are two main concepts to understand in SwarmPAL, *data* and *processes*:
- *Data* live within a [xarray DataTree](https://docs.xarray.dev/en/latest/user-guide/data-structures.html#datatree) and can hold multiple datasets (both the inputs and outputs of a *process*)
- *Processes* behave like functions. They act on data to transform them by adding derived parameters into the data object. Such processes are organised together within thematic *toolboxes* (FAC, TFA, DSECS, ...).

We logically separate a workflow into two steps:
- *fetching data*: data are downloaded from a [VirES](https://vires.services) server or any [HAPI](https://hapi-server.org/) server, or loaded from disk
- *applying processes*: apply a "PalProcess" to your data to perform a given analysis routine

SwarmPAL offers different interfaces to run these steps:

````{eval-rst}
.. tabs::

   .. tab:: Granular usage

      The :mod:`swarmpal.io` module contains utilities to fetch and load data. The FAC toolbox (:mod:`swarmpal.toolboxes.fac`) contains the ``FAC_single_sat`` process to perform the FAC analysis.

      .. code-block:: python

         from swarmpal.io import create_paldata, PalDataItem
         from swarmpal.toolboxes.fac.processes import FAC_single_sat

         # Fetch magnetic data from the VirES server
         data = create_paldata(
            PalDataItem.from_vires(
               server_url="https://vires.services/ows",
               collection="SW_OPER_MAGA_LR_1B",
               measurements=["B_NEC"],
               models=["CHAOS"],
               start_time="2025-04-10T00:00:00",
               end_time="2025-04-11T00:00:00",
               options={"asynchronous": False, "show_progress": False},
            )
         )
         # Apply the FAC_single_sat process from the FAC toolbox
         process = FAC_single_sat(
            config=dict(
               dataset="SW_OPER_MAGA_LR_1B",
               model_varname="B_NEC_CHAOS",
               measurement_varname="B_NEC",
            )
         )
         data = process(data)


   .. tab:: Generic process application

      The generic ``fetch_data`` and ``apply_process`` functions can be used to operate any toolbox which is known to SwarmPAL.

      .. code-block:: python

         from swarmpal import fetch_data, apply_process

         data = fetch_data(
            dict(
               data_params=[
                     dict(
                        provider="vires",
                        server_url="https://vires.services/ows",
                        collection="SW_OPER_MAGA_LR_1B",
                        measurements=["B_NEC"],
                        models=["CHAOS"],
                        start_time="2025-04-10T00:00:00",
                        end_time="2025-04-11T00:00:00",
                        options={"asynchronous": False, "show_progress": False},
                     )
               ],
            )
         )
         data = apply_process(
            data=data,
            process_name="FAC_single_sat",
            config={
               "dataset": "SW_OPER_MAGA_LR_1B",
               "model_varname": "B_NEC_CHAOS",
               "measurement_varname": "B_NEC",
            },
         )

   .. tab:: CLI / batch operation

      Command line utilities are provided to allow running toolbox code using configuration files.

      .. code-block:: bash

         $ swarmpal batch FAC_CONFIG.yml TEST_FAC_OUTPUTS.NC

      .. code-block:: yaml

         # FAC_CONFIG.yml

         data_params:
           - provider: vires
             collection: SW_OPER_MAGA_LR_1B
             measurements:
               - B_NEC
             models:
               - CHAOS
             start_time: "2025-04-10T00:00:00"
             end_time: "2025-04-11T00:00:00"
             server_url: https://vires.services/ows
             options:
               asynchronous: false
               show_progress: false

         process_params:
           - process_name: FAC_single_sat

````

The resulting data contains both the input data (in this case ``SW_OPER_MAGA_LR_1B``) and outputs that have been generated (``PAL_FAC_single_sat``):

```
<xarray.DataTree>
Group: /
│   Attributes:
│       PAL_meta:  {"FAC_single_sat": {"dataset": "SW_OPER_MAGA_LR_1B", "model_va...
├── Group: /SW_OPER_MAGA_LR_1B
│       Dimensions:      (Timestamp: 86400, NEC: 3)
│       Coordinates:
│         * Timestamp    (Timestamp) datetime64[ns] 691kB 2025-04-10 ... 2025-04-10T2...
│         * NEC          (NEC) <U1 12B 'N' 'E' 'C'
│       Data variables:
│           Spacecraft   (Timestamp) category 86kB A A A A A A A A A ... A A A A A A A A
│           B_NEC        (Timestamp, NEC) float64 2MB 1.554e+04 -1.437e+03 ... 3.531e+04
│           Latitude     (Timestamp) float64 691kB -13.45 -13.38 -13.32 ... 49.42 49.36
│           Radius       (Timestamp) float64 691kB 6.826e+06 6.826e+06 ... 6.815e+06
│           Longitude    (Timestamp) float64 691kB 16.42 16.42 16.42 ... -167.4 -167.4
│           B_NEC_CHAOS  (Timestamp, NEC) float64 2MB 1.555e+04 -1.429e+03 ... 3.532e+04
│       Attributes:
│           Sources:         ['CHAOS-8.1_static.shc', 'SW_OPER_MAGA_LR_1B_20250409T00...
│           MagneticModels:  ["CHAOS = 'CHAOS-Core'(max_degree=20,min_degree=1) + 'CH...
│           AppliedFilters:  []
│           PAL_meta:        {"analysis_window": ["2025-04-10T00:00:00", "2025-04-11T...
└── Group: /PAL_FAC_single_sat
        Dimensions:    (Timestamp: 86399)
        Coordinates:
          * Timestamp  (Timestamp) datetime64[ns] 691kB 2025-04-10T00:00:00.500000 .....
        Data variables:
            FAC        (Timestamp) float64 691kB -0.004585 -0.01044 ... -0.02231
            IRC        (Timestamp) float64 691kB -0.003524 -0.008009 ... 0.01995
            Latitude   (Timestamp) float64 691kB -13.45 -13.38 -13.32 ... 49.49 49.42
            Radius     (Timestamp) float64 691kB 6.826e+06 6.826e+06 ... 6.815e+06
            Longitude  (Timestamp) float64 691kB 16.42 16.42 16.42 ... -167.4 -167.4
        Attributes:
            Sources:  ['CHAOS-8.1_static.shc', 'SW_OPER_MAGA_LR_1B_20250409T000000_20...
```

For visualisation tools, refer to the toolbox guides on the following pages.
