#!/bin/sh

# Set multiprocessing parameters
export MKL_NUM_THREADS=10
export NUMEXPR_NUM_THREADS=10
export OMP_NUM_THREADS=10
PROCESSES=3  # Number of networks to simulate in parallel
REPEATS=20  # Total number of networks to simulate

_SIMULATION_PATH_PREFIX=$DATA_PATH/simulations/GIF_network

_run_deriv_simulations() {
    # Takes one argument: temperature

    _simulation_path=$_SIMULATION_PATH_PREFIX/deriv_input_"$1"_ser_only
    for baseline_ in "-0.080" "-0.060" "-0.040" "-0.020" "0.000" "0.020" "0.040" "0.060" "0.080"; do
        _deriv_input=$_simulation_path/"$baseline_"_baseline/input.dat
        mkdir -p $(dirname "$_deriv_input")  # Create dir to hold output if it doesn't exist
        rm -f $(dirname "$_deriv_input")/*.hdf5  # Clear any data from previous runs
        # Generate network input
        python ./input_generators/deriv.py \
            -v \
            $_deriv_input \
            --baseline "$baseline_" \
            || exit 999

        # Run simulations for several networks in parallel
        echo "Starting derivative simulations with $_deriv_input baseline ($1)"
        for i in $(seq 0 $[$REPEATS - 1]); do
            python ./run_simulation.py \
                $(if [ $i == 0 ]; then echo "-v"; else echo "--num-ser-examples 0 --num-gaba-examples 0"; fi) \
                $DATA_PATH/models/GIF_network/5HT_$1/GIFnet_${i}_subsample_base.mod \
                $_deriv_input \
                $(dirname "$_deriv_input")/rep${i}.hdf5 \
                --seed-background ${i} \
                --sigma-background 0.002 \
                --no-gaba \
                &
            if [ $[($i + 1) % $PROCESSES] == 0 ]; then
                wait
            fi
        done
        wait
    done
}

_run_deriv_simulations "room_temp"
_run_deriv_simulations "heated"
