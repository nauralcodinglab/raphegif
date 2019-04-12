# Preprocess 5HT noisy current data.
# Must be run from project root with appropriate conda environment active!

PROJECT_ROOT = $(shell pwd)
DATA_PATH = data/processed/5HT_fastnoise
SCRIPT_PATH = analysis/regression_tinkering

.PHONY : all
all : $(DATA_PATH)/5HT_goodcells.ldat 

$(DATA_PATH)/5HT_goodcells.ldat : data/fast_noise_5HT/index.csv $(SCRIPT_PATH)/preprocess_fast_noise.py
	PYTHONPATH="$(PROJECT_ROOT)" python $(SCRIPT_PATH)/preprocess_fast_noise.py

.PHONY : clean
clean :
	rm $(DATA_PATH)/5HT_goodcells.ldat

