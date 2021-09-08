# Set environment variables.
export MKL_NUM_THREADS=10
export NUMEXPR_NUM_THREADS=10
export OMP_NUM_THREADS=10

# Define constants.
REPEATS=20
PROCESSES=3

# DRN WITH GABA
for baseline_ in "-0.080" "-0.060" "-0.040" "-0.020" "0.000" "0.020" "0.040" "0.060" "0.080"; do
    deriv_input=../../data/simulations/GIF_network/deriv_input/"$baseline_"_baseline/input.dat
    mkdir -p $(dirname "$deriv_input")
    rm -f $(dirname "$deriv_input")/*.hdf5
    python ./input_generators/deriv.py $deriv_input -v --baseline "$baseline_" || exit 999

    echo "Starting derivative simulations with $deriv_input baseline"
	for i in $(seq 0 $[$REPEATS - 1]); do
	    python ./run_simulation.py \
		$(if [ $i == 0 ]; then echo "-v"; else echo "--num-ser-examples 0 --num-gaba-examples 0"; fi) \
		../../data/models/GIF_network/GIFnet_${i}_subsample_base.mod \
		$deriv_input \
		$(dirname "$deriv_input")/rep${i}.hdf5 \
		--seed-background ${i} --sigma-background 0.002 &
	    if [ $[($i + 1) % $PROCESSES] == 0 ]; then
		wait
	    fi
	done
	wait
done
