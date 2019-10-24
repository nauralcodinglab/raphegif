# Define constants.
SERMODS=../../data/models/5HT/5HT_AugmentedGIFs.lmod
GABAMODS=../../data/models/GABA/GABA_GIFs.lmod
REPEATS=10

# Generate GIFnet models.
python ./generate_models.py --sermods $SERMODS --gabamods $GABAMODS --prefix ../../data/models/subsample_GIFnet --opts gifnetmod_opts.json -r $REPEATS -v

# Generate input.
python ../GIF_network/input_generators/current_step.py ../../simulations/GIF_network/input/square_step.dat \
	--baseline-ser 0.01 --min-ser 0.02 --max-ser 0.070 \
	--baseline-gaba 0. --min-gaba 0.01 --max-gaba 0.050

# Run simulations
for modtype in base noIA fixedIA; do
	for condition in GABA noGABA; do
		echo "Starting $modtype $condition simulations"
		for i in {1..$REPEATS}; do
			python ./run_simulation.py $(if [ $i == 1 ]; then echo "-v"; fi) \
				../../data/models/GIFnet_${i}_subsample_${modtype}.mod \
				../../data/simulations/GIF_network/input/square_step.dat \
				../../data/simulations/GIF_network/subsample_${modtype}/${modtype}_${condition}_rep${i}.hdf5 \
				--seed-background $i \
				$(if [ $condition == noGABA ]; then echo "--no-gaba --no-feedforward"; fi) &
		done
		wait
	done
done

