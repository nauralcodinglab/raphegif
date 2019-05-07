PROJECT_ROOT=$(shell pwd)
DATA_PATH=./data/processed/exclusion
SCRIPT_PATH=./analysis/exclusion

.PHONY : all
all : $(DATA_PATH)/intrinsic_reliabilities.ldat 

$(DATA_PATH)/intrinsic_reliabilities.ldat : $(SCRIPT_PATH)/assess_reliability.py $(DATA_PATH)/experiments.ldat | $(DATA_PATH)
	PYTHONPATH="$(PROJECT_ROOT)" python $< -v $(DATA_PATH)/experiments.ldat $@

$(DATA_PATH)/experiments.ldat : $(SCRIPT_PATH)/preprocess_experiments.py | $(DATA_PATH)
	PYTHONPATH="$(PROJECT_ROOT)" python $< -v $@

$(DATA_PATH) :
	mkdir -p $(DATA_PATH)
