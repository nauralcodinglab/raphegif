#! /bin/sh

PROCPATH=../../data/processed
MODPATH=../../data/models

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
