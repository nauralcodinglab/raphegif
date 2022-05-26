#!/bin/sh

# Control number of threads used for numpy parallelism.
export MKL_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export OMP_NUM_THREADS=2

SERMODS_ROOM_TEMP=$DATA_PATH/models/5HT/5HT_AugmentedGIFs.lmod
SERMODS_HEATED=$DATA_PATH/models/5HT_heated/5HT_AugmentedGIFs.lmod
REPEATS=20
PROCESSES=10

# Generate GIFnet models. Use generate_mpfc_models.py because GABA models aren't required.
echo "Generating GIFnet models."
mkdir -p $DATA_PATH/models/GIF_network/{5HT_room_temp,5HT_heated} \
&& python ./generate_sernet_models.py -v \
    --mods $SERMODS_ROOM_TEMP \
    --prefix $DATA_PATH/models/GIF_network/5HT_room_temp/GIFnet \
    --opts sernetmod_room_temp_opts.json \
    -r $REPEATS \
&& python ./generate_mpfc_models.py -v \
    --mods $SERMODS_HEATED \
    --prefix $DATA_PATH/models/GIF_network/5HT_heated/GIFnet \
    --opts sernetmod_heated_opts.json \
    -r $REPEATS \
&& _MODELS_GENERATED=1 \
|| _MODELS_GENERATED=0

# Generate input.
_INPUT_PATH=$DATA_PATH/simulations/GIF_network/step_input/square_step_principal_only.dat
echo "Generating network input."
mkdir -p $(dirname $_INPUT_PATH) \
&& python ./input_generators/current_step.py \
    $_INPUT_PATH \
    --baseline-ser 0.00 \
    --min-ser 0.005 \
    --max-ser 0.150 \
    --baseline-gaba 0. \
    --min-gaba 0. \
    --max-gaba 0. \
|| exit 101

_OUTPUT_PATH_PREFIX=$DATA_PATH/simulations/GIF_network/step_input
if [ [ $_MODELS_GENERATED -eq 1 ] -a [ -e $_INPUT_PATH ] ]; then
    echo "Starting room temperature simulations."
    mkdir -p $_OUTPUT_PATH_PREFIX/5HT_base/GABA_KO
    for i in $(seq 0 $[$REPEATS - 1]); do
        python ./run_simulation.py \
            $(if [ $i == 0 ]; then echo "-v"; else echo "--num-ser-examples 0 --num-gaba-examples 0") \
            $DATA_PATH/models/GIF_network/5HT/GIFnet_${i}_subsample_base.mod \
            $_INPUT_PATH \
            $_OUTPUT_PATH_PREFIX/5HT_base/GABA_KO/rep${i}.hdf5 \
            --seed-background ${i} \
            --sigma-bacground 0.002 \
            --no-gaba \
            &
        if [ $[($i + 1) % $PROCESSES] == 0 ]; then
            wait
        fi
    done
    wait

    echo "Starting heated bath simulations."
    mkdir -p $_OUTPUT_PATH_PREFIX/5HT_heated_base/GABA_KO
    for i in $(seq 0 $[$REPEATS - 1]); do
        python ./run_simulation.py \
            $(if [ $i == 0 ]; then echo "-v"; else echo "--num-ser-examples 0 --num-gaba-examples 0") \
            $DATA_PATH/models/GIF_network/5HT_heated/GIFnet_${i}_subsample_base.mod \
            $_INPUT_PATH \
            $_OUTPUT_PATH_PREFIX/5HT_heated_base/GABA_KO/rep${i}.hdf5 \
            --seed-background ${i} \
            --sigma-bacground 0.002 \
            --no-gaba \
            &
        if [ $[($i + 1) % $PROCESSES] == 0 ]; then
            wait
        fi
    done
    wait
fi
