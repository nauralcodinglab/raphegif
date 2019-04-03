# Preprocess GABA noisy current data.
# Must be run from project root with appropriate conda environment active!

DATA_PATH = data/processed/GABA_fastnoise
SCRIPT_PATH = figs/scripts/gaba_neurons

.PHONY : all
all : $(DATA_PATH)/gaba_goodcells.ldat 

$(DATA_PATH)/gaba_goodcells.ldat : data/GABA_cells/index.csv $(SCRIPT_PATH)/preprocess_fast_noise.py
	python $(SCRIPT_PATH)/preprocess_fast_noise.py

.PHONY : clean
clean :
	rm $(DATA_PATH)/gaba_goodcells.ldat

