# Preprocess 5HT noisy current data.
# Must be run from project root with appropriate conda environment active!

PROJECT_ROOT = $(shell pwd)
DATA_PATH = data/processed/5HT_fastnoise
MODEL_PATH = data/models/5HT
SCRIPT_PATH = analysis/regression_tinkering
SRC_PATH = src

.PHONY : all
all : $(MODEL_PATH)/sergifs.lmod $(MODEL_PATH)/serkgifs.lmod

$(MODEL_PATH)/sergifs.lmod : $(DATA_PATH)/5HT_goodcells.ldat $(SCRIPT_PATH)/fit_sergifs.py \
$(SRC_PATH)/GIF.py $(SRC_PATH)/Filter_Exps.py | $(MODEL_PATH)
	PYTHONPATH="$(PROJECT_ROOT)" python $(SCRIPT_PATH)/fit_sergifs.py

$(MODEL_PATH)/serkgifs.lmod : $(DATA_PATH)/5HT_goodcells.ldat \
$(SCRIPT_PATH)/fit_serkgifs.py $(SCRIPT_PATH)/model_evaluation.py \
$(SRC_PATH)/AugmentedGIF.py $(SRC_PATH)/Filter_Exps.py | $(MODEL_PATH)
	PYTHONPATH="$(PROJECT_ROOT)" python $(SCRIPT_PATH)/fit_serkgifs.py

$(MODEL_PATH) :
	mkdir -p $(MODEL_PATH)

$(DATA_PATH)/5HT_goodcells.ldat : data/fast_noise_5HT/index.csv $(SCRIPT_PATH)/preprocess_fast_noise.py | $(DATA_PATH)
	PYTHONPATH="$(PROJECT_ROOT)" python $(SCRIPT_PATH)/preprocess_fast_noise.py

$(DATA_PATH) :
	mkdir -p $(DATA_PATH)

.PHONY : clean
clean :
	rm $(DATA_PATH)/5HT_goodcells.ldat

