# Set environment variables.
export MKL_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export OMP_NUM_THREADS=2

# Define constants.
SERMODS=../../data/models/5HT/5HT_AugmentedGIFs.lmod
GABAMODS=../../data/models/GABA/GABA_GIFs.lmod
REPEATS=20
PROCESSES=10

synaptic_burst_input=../../data/simulations/GIF_network/synaptic_burst_input/synaptic_burst_input.dat
python ../GIF_network/input_generators/synaptic_bursts.py $synaptic_burst_input -v || exit 999

# DRN WITH GABA
echo "Starting DRN GABA control simulations."
for i in $(seq 0 $[$REPEATS - 1]); do
    python ./run_simulation.py \
        $(if [ $i == 0 ]; then echo "-v"; else echo "--num-ser-examples 0 --num-gaba-examples 0"; fi) \
        ../../data/models/GIF_network/GIFnet_${i}_subsample_base.mod \
        $synaptic_burst_input \
        ../../data/simulations/GIF_network/synaptic_burst_input/DRN_subsample_base/GABA_base/rep${i}.hdf5 \
        --seed-background ${i} --sigma-background 0.002 &
    if [ $[($i + 1) % $PROCESSES] == 0 ]; then
        wait
    fi
done
wait

# GABA KNOCKOUT
echo "Starting DRN GABA KO simulations."
for i in $(seq 0 $[$REPEATS - 1]); do
    python ./run_simulation.py \
        $(if [ $i == 0 ]; then echo "-v"; else echo "--num-ser-examples 0 --num-gaba-examples 0"; fi) \
        ../../data/models/GIF_network/GIFnet_${i}_subsample_base.mod \
        $synaptic_burst_input \
        ../../data/simulations/GIF_network/synaptic_burst_input/DRN_subsample_base/GABA_KO/rep${i}.hdf5 \
        --seed-background ${i} --sigma-background 0.002 \
        --no-gaba &
    if [ $[($i + 1) % $PROCESSES] == 0 ]; then
        wait
    fi
done
wait

# mPFC
echo "Starting mPFC simulations."
for i in $(seq 0 $[$REPEATS - 1]); do
    python ./run_simulation.py \
        $(if [ $i == 0 ]; then echo "-v"; else echo "--num-ser-examples 0 --num-gaba-examples 0"; fi) \
        ../../data/models/GIF_network/mPFC/GIFnet_${i}_subsample_base.mod \
        $synaptic_burst_input \
        ../../data/simulations/GIF_network/synaptic_burst_input/mPFC_base/GABA_KO/rep${i}.hdf5 \
        --seed-background ${i} --sigma-background 0.002 \
        --no-gaba &
    if [ $[($i + 1) % $PROCESSES] == 0 ]; then
        wait
    fi
done
wait
