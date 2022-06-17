# Introduction

SwarmX is a Python package containing analysis and visualisation tools to interact with Swarm products. We rely on the VirES service (via [viresclient](https://viresclient.readthedocs.io/)) to provide access to data and models as well as frameworks from the Python ecosystem (e.g. Xarray) and specific utilities from the scientific community (e.g. apexpy).

```{mermaid}
graph LR

    subgraph SwarmX: Analysis Tools
        direction LR
        v2[viresclient]
        algvis(Algorithms<br>Visualisations)
        v2 -->|Pre-configured<br>inputs| algvis
        pack[swarmx package]
        algvis === pack
        eco[[Python<br>ecosystem]] --> algvis
    end

    scidev((Expert<br>scientists)) -.->|Project work| algvis
    eng((Software<br>engineers)) -.->|Maintain<br>Improve| pack

    subgraph Data Access
        direction LR
        VS[(VirES Server:<br>One product = one time series<br>Subsampling & subsetting<br>Geomagnetic model evaluation<br>+ more, e.g. conjunctions)]
        VS -->|VirES<br>API| v[viresclient]
        v -->|Python<br>datatypes| du([Direct usage<br>Notebooks<br>Other packages])
    end
```

The package is structured so that researchers can run high level analysis routines with just a few lines of code which seemlessly pulls in Swarm data underneath, while the steps of the routines are also exposed so that they can be run more flexibly (e.g. with other data).

```{mermaid}
graph LR
subgraph SwarmX package
    direction TB
    npf[Algorithms<br>NumPy/SciPy etc]
    viz[Visualisation<br>Matplotlib etc]
    sds[SwarmX<br>data structures]
    sds --> npf --> sds --> viz
    sds --- intf[Convenient interface]
end

subgraph VirES connection
    viresclient -->|Swarm<br>products| sds
end
npf & viz -.-> usem[Use manually<br>with other data etc]
intf -.-> used[Use directly]
```
