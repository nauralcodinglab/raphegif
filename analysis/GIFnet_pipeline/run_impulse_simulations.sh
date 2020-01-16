# Set environment variables.
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export OMP_NUM_THREADS=4

# Define constants.
SERMODS=../../data/models/5HT/5HT_AugmentedGIFs.lmod
GABAMODS=../../data/models/GABA/GABA_GIFs.lmod
REPEATS=10
PROCESSES=3

# Generate input.
full_network_input=../../data/simulations/GIF_network/input/impulse_full.dat
null_gaba_network_input=../../data/simulations/GIF_network/input/impulse_5HT_only.dat
python ../GIF_network/input_generators/impulse.py impulse_full_opts.json $full_network_input || exit 999
python ../GIF_network/input_generators/impulse.py impulse_nogaba_opts.json $null_gaba_network_input || exit 999

# Run simulations
for modtype in base noIA fixedIA; do
    for condition in GABA noGABA; do
        echo "Starting $modtype $condition simulations"
        for i in $(seq 0 $[$REPEATS - 1]); do
            python ./run_simulation.py \
                $(if [ $i == 0 ]; then echo "-v"; else echo "--num-ser-examples 0 --num-gaba-examples 0"; fi) \
                ../../data/models/GIF_network/GIFnet_${i}_subsample_${modtype}.mod \
                $(if [ $condition == GABA ]; then echo "$full_network_input"; elif [ $condition == noGABA ]; then echo "$null_gaba_network_input"; fi) \
                ../../data/simulations/GIF_network/subsample_${modtype}/${modtype}_impulse_${condition}_rep${i}.hdf5 \
                --seed-background ${i} &
            if [ $[($i + 1) % $PROCESSES] == 0 ]; then
                wait
            fi
        done
    done
done

