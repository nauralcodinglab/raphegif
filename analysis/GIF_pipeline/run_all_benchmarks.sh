#! /bin/sh

PROCPATH=../../data/processed
MODPATH=../../data/models

# Benchmark normal models.
echo "Benchmarking regular old models."
for celltype in 5HT mPFC GABA; do
    python ./run_benchmarks.py \
        $PROCPATH/${celltype}_fastnoise/${celltype}_goodcells.ldat \
        $MODPATH/${celltype}/${celltype}_GIFs.lmod \
        $MODPATH/${celltype}/${celltype}_AugmentedGIFs.lmod \
        $MODPATH/${celltype}/${celltype}_iGIF_NPs.lmod \
        $MODPATH/${celltype}/${celltype}_iGIF_VRs.lmod \
        -o $MODPATH/$celltype --precision 8. -v &
done
wait

# Benchmark models fitted to smaller datasets.
echo "Benchmarking models fitted to small datasets."
for celltype in 5HT mPFC GABA; do
    python ./run_benchmarks.py \
        $PROCPATH/${celltype}_fastnoise/${celltype}_minified_goodcells.ldat \
        $MODPATH/${celltype}/${celltype}_minified_GIFs.lmod \
        $MODPATH/${celltype}/${celltype}_minified_AugmentedGIFs.lmod \
        $MODPATH/${celltype}/${celltype}_minified_iGIF_NPs.lmod \
        $MODPATH/${celltype}/${celltype}_minified_iGIF_VRs.lmod \
        -o $MODPATH/$celltype --precision 8. -v &
done
wait

