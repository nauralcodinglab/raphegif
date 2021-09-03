# Fitting single-cell models to DRN and mPFC neurons

## Pipeline for single cell models
```mermaid
graph LR

raw(Raw data) --> qc{QC and preprocessing}
qc --> good(Good recordings)
qc --> bad(Bad recordings)

good -- Training data --> fit{Fit models}
fit --> GIF(GIF)
fit --> AugmentedGIF(AugmentedGIF)
fit --> othermod(Other models...)

GIF --> bench{Run benchmarks}
AugmentedGIF --> bench
othermod --> bench
good -- Test data --> bench

bench --> ex(Example traces)
bench --> md(Spike train prediction)
bench --> r2V(Var. explained on V)
bench --> r2dV(Var. explained on dV)
```

The analysis pipeline ingests raw electrophysiological data in Axon Binary
Format, uses it to train single cell models from `grr`, and outputs various
benchmarks computed on held-out data. These three stages are implemented by the
`AEC_QC.py`, `fit_mods.py`, and `run_all_benchmarks.sh` scripts in this
directory. The scripts are designed to be run from the command line to process
experiments/models in batches. Running `make all` from this directory reproduces
my GIF-based models of DRN neurons and mPFC pyramidal cells along with their
associated benchmarks.

## Detailed pipeline

From single cells to networks.

```mermaid
graph TB

subgraph ser[5HT]

rawser[Raw recordings] -- preprocess_experiments.py --> proser[Cleaned recordings]

proser --> serGIF[GIF]
proser --> serKGIF[KGIF]
proser --> seriGIF[iGIF]

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
