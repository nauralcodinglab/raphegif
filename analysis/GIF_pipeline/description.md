# GIF fitting pipeline

## Pipeline for single cell models

Goal is to take in raw recordings and output fitted models, benchmarks, and
example traces.

```mermaid
graph LR

raw(Raw data) --> qc{QC and preprocessing}
qc --> good(Good recordings)
qc --> bad(Bad recordings)

good --> fit{Fit models}
fit --> mod(Models)
fit --> bench(Benchmarks)
fit --> ex(Example traces)
```

Notes:
- Quality control step should accept parameters for rejection thresholds.
- Each model probably needs its own fitting script.

## Detailed pipeline

From single cells to networks.

```mermaid
graph TB

subgraph ser[5HT]

rawser[Raw recordings] -- preprocess_experiments.py --> proser[Cleaned recordings]

proser --> serGIF[GIF]
proser --> serKGIF[KGIF]
proser --> seriGIF[iGIF]

serKGIF --> latency[Spike latency simulations]

end

subgraph pyr[mPFC]

rawpyr[Raw recordings] -- preprocess_experiments.py --> propyr[Cleaned recordings]

propyr --> pyrGIF[GIF]
propyr --> pyriGIF[iGIF]

end

subgraph som[SOM]

rawsom[Raw recordings] -- preprocess_experiments.py --> prosom[Cleaned recordings]

prosom --> somGIF[GIF]
prosom --> somiGIF[iGIF]

end

serGIF --> Benchmarks
serKGIF --> Benchmarks
seriGIF --> Benchmarks
pyrGIF --> Benchmarks
pyriGIF --> Benchmarks
somGIF --> Benchmarks
somiGIF --> Benchmarks


serKGIF --> netmod[Network model]
somGIF --> netmod
netmod --> netsim[Network simulations]
netin[Network input] --> netsim

```
