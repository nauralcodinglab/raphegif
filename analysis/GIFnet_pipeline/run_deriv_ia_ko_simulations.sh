# Set multiprocessing parameters
export MKL_NUM_THREADS=10
export NUMEXPR_NUM_THREADS=10
export OMP_NUM_THREADS=10
PROCESSES=3  # Number of networks to simulate in parallel
REPEATS=20  # Total number of networks to simulate

# Run derivative simulations in DRN with IA KO and several input baselines
for baseline_ in "-0.080" "-0.060" "-0.040" "-0.020" "0.000" "0.020" "0.040" "0.060" "0.080"; do
	deriv_input=../../data/simulations/GIF_network/deriv_input_ia_ko/"$baseline_"_baseline/input.dat
	mkdir -p $(dirname "$deriv_input")  # Create dir to hold output if it doesn't exist
	rm -f $(dirname "$deriv_input")/*.hdf5  # Clear any data from previous runs
	# Generate network input
	python ../GIF_network/input_generators/deriv.py $deriv_input -v --baseline "$baseline_" || exit 999

	# Run simulations for several networks in parallel
	echo "Starting derivative simulations with $deriv_input baseline"
	for i in $(seq 0 $[$REPEATS - 1]); do
	    python ./run_simulation.py \
		$(if [ $i == 0 ]; then echo "-v"; else echo "--num-ser-examples 0 --num-gaba-examples 0"; fi) \
		../../data/models/GIF_network/GIFnet_${i}_subsample_noIA.mod \
		$deriv_input \
		$(dirname "$deriv_input")/rep${i}.hdf5 \
		--seed-background ${i} --sigma-background 0.002 &
	    if [ $[($i + 1) % $PROCESSES] == 0 ]; then
		wait
	    fi
	done
	wait
done
