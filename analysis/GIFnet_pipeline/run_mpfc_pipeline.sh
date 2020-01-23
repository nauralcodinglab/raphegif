# Set environment variables.
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export OMP_NUM_THREADS=4

# Define constants.
MPFCMODS=../../data/models/mPFC/mPFC_GIFs.lmod
REPEATS=10
PROCESSES=5

# Generate GIFnet models.
python ./generate_mpfc_models.py --mods $MPFCMODS  --prefix ../../data/models/GIF_network/mPFC/GIFnet --opts mpfcmod_opts.json -r $REPEATS -v || exit 999

# Generate input.
mpfc_network_input=../../data/simulations/GIF_network/step_input/square_step_mPFC.dat
python ../GIF_network/input_generators/current_step.py $mpfc_network_input \
	--baseline-ser 0.00 --min-ser 0.010 --max-ser 0.250 \
	--baseline-gaba 0. --min-gaba 0. --max-gaba 0. \
	|| exit 999

# Run simulations
echo "Starting simulations."
for i in $(seq 0 $[$REPEATS - 1]); do
    python ./run_simulation.py \
        $(if [ $i == 0 ]; then echo "-v"; else echo "--num-ser-examples 0 --num-gaba-examples 0"; fi) \
        ../../data/models/GIF_network/mPFC/GIFnet_${i}_subsample_base.mod \
        $mpfc_network_input \
        ../../data/simulations/GIF_network/step_input/mPFC_base/GABA_KO/rep${i}.hdf5 \
        --seed-background ${i} --sigma-background 0.001 \
        --no-gaba &
    if [ $[($i + 1) % $PROCESSES] == 0 ]; then
        wait
    fi
done
wait

