# Set environment variables.
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export OMP_NUM_THREADS=4

# Define constants.
SERMODS=../../data/models/5HT/5HT_AugmentedGIFs.lmod
GABAMODS=../../data/models/GABA/GABA_GIFs.lmod
REPEATS=10
PROCESSES=4

# Generate GIFnet models.
python ./generate_models.py --sermods $SERMODS --gabamods $GABAMODS --prefix ../../data/models/GIF_network/GIFnet --opts gifnetmod_opts.json -r $REPEATS -v || exit 999

# Generate input.
full_network_input=../../data/simulations/GIF_network/step_input/square_step_full.dat
null_gaba_network_input=../../data/simulations/GIF_network/step_input/square_step_5HT_only.dat
python ../GIF_network/input_generators/current_step.py $full_network_input \
	--baseline-ser 0. --min-ser 0.010 --max-ser 0.060 \
	--baseline-gaba 0. --min-gaba 0.010 --max-gaba 0.060 \
	|| exit 999
python ../GIF_network/input_generators/current_step.py $null_gaba_network_input \
	--baseline-ser 0. --min-ser 0.010 --max-ser 0.060 \
	--baseline-gaba 0. --min-gaba 0. --max-gaba 0. \
	|| exit 999

# Run simulations
for modtype in base noIA fixedIA; do

    # GABA CONTROL
    echo "Starting $modtype GABA control simulations."
    for i in $(seq 0 $[$REPEATS - 1]); do
        python ./run_simulation.py \
            $(if [ $i == 0 ]; then echo "-v"; else echo "--num-ser-examples 0 --num-gaba-examples 0"; fi) \
            ../../data/models/GIF_network/GIFnet_${i}_subsample_${modtype}.mod \
            $full_network_input \
            ../../data/simulations/GIF_network/step_input/DRN_$modtype/GABA_base/rep${i}.hdf5 \
            --seed-background ${i} --sigma-background 0.001 &
        if [ $[($i + 1) % $PROCESSES] == 0 ]; then
            wait
        fi
    done
    wait

    # GABA KNOCKOUT
    echo "Starting $modtype GABA KO simulations."
    for i in $(seq 0 $[$REPEATS - 1]); do
        python ./run_simulation.py \
            $(if [ $i == 0 ]; then echo "-v"; else echo "--num-ser-examples 0 --num-gaba-examples 0"; fi) \
            ../../data/models/GIF_network/GIFnet_${i}_subsample_${modtype}.mod \
            $full_network_input \
            ../../data/simulations/GIF_network/step_input/DRN_$modtype/GABA_KO/rep${i}.hdf5 \
            --seed-background ${i} --sigma-background 0.001 \
            --no-gaba &
        if [ $[($i + 1) % $PROCESSES] == 0 ]; then
            wait
        fi
    done
    wait

    echo "Starting $modtype GABA no input simulations."
    for i in $(seq 0 $[$REPEATS - 1]); do
        python ./run_simulation.py \
            $(if [ $i == 0 ]; then echo "-v"; else echo "--num-ser-examples 0 --num-gaba-examples 0"; fi) \
            ../../data/models/GIF_network/GIFnet_${i}_subsample_${modtype}.mod \
            $null_gaba_network_input \
            ../../data/simulations/GIF_network/step_input/DRN_$modtype/GABA_noinput/rep${i}.hdf5 \
            --seed-background ${i} --sigma-background 0.001 &
        if [ $[($i + 1) % $PROCESSES] == 0 ]; then
            wait
        fi
    done
    wait

done

